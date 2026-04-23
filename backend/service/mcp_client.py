"""
Author: ai-business-hql qingli.hql@alibaba-inc.com
Date: 2025-06-16 16:50:17
LastEditors: ai-business-hql ai.bussiness.hql@gmail.com
LastEditTime: 2025-12-24 19:03:58
FilePath: /comfyui_copilot/backend/service/mcp-client.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

from ..service.workflow_rewrite_tools import get_current_workflow
from ..utils.globals import (
    BACKEND_BASE_URL,
    get_comfyui_copilot_api_key,
    DISABLE_WORKFLOW_GEN,
)
import asyncio
from contextlib import AsyncExitStack
import json
import os
import traceback
from typing import List, Dict, Any, Optional

try:
    from agents.mcp import MCPServerSse
    from agents import handoff, RunContextWrapper, HandoffInputData
    from agents.tool_context import ToolContext
    from agents.usage import Usage

    if not hasattr(__import__("agents"), "Agent"):
        raise ImportError
except Exception:
    raise ImportError(
        "Detected incorrect or missing 'agents' package while loading MCP components. "
        "Please install 'openai-agents' and ensure this plugin prefers it. Commands:\n"
        "  python -m pip uninstall -y agents gym tensorflow\n"
        "  python -m pip install -U openai-agents"
    )

from ..agent_factory import create_agent, diagnose_image, search_workflows
from ..service.workflow_rewrite_tools import (
    update_workflow,
    search_node_local,
    get_node_infos,
    remove_node,
)
from ..service.workflow_rewrite_agent import get_rewrite_expert_by_name

LOCAL_TOOLS_REGISTRY = {
    "search_workflows": search_workflows,
    "diagnose_image": diagnose_image,
    "get_current_workflow": get_current_workflow,
    "update_workflow": update_workflow,
    "search_node_local": search_node_local,
    "get_node_infos": get_node_infos,
    "remove_node": remove_node,
    "get_rewrite_expert_by_name": get_rewrite_expert_by_name,
}
from ..service.workflow_rewrite_agent import create_workflow_rewrite_agent
from ..service.message_memory import message_memory_optimize
from ..utils.request_context import get_rewrite_context, get_session_id, get_config
from ..utils.logger import log
from openai.types.responses import ResponseTextDeltaEvent
from openai import APIError, RateLimitError
from pydantic import BaseModel
from .rag_agent_client import pass_through_rag_agent

# Budget for MCP session initialization — wraps the full streamable-http
# handshake (SSE endpoint event + initialize RPC + tool-list RPC) performed
# by MCPServerSse inside exit_stack.enter_async_context(server).
#
# Measured on 2026-04-13 via scripts/measure_mcp_init.py:
#   Mac Copilot MCP (192.168.10.2:7002) → ~110-140ms for full init + list_tools
#   on a WARM server.
#
# Why 30s (not 8) — 2026-04-13 Mac-side handoff:
# On the very first MCP connection after a Mac MCP server restart, the Mac
# side lazy-loads the Korean→English translator + RAG embedding models used
# by recall_workflow / gen_workflow. This cold path was pushing the initial
# tools/list round-trip well past the old 8s budget, which killed the chat
# handler before any tool could run. Raising to 30s absorbs that cold start;
# warm reconnects still take <200ms so there is no steady-state cost.
# External MCP (Bing/ModelScope over WAN) inherits the same budget and is
# opt-in via BING_MCP_ENABLED.
MCP_CONNECT_TIMEOUT_SECONDS = float(
    os.getenv("COPILOT_MCP_CONNECT_TIMEOUT_SECONDS", "30")
)
BING_MCP_DEFAULT_URL = "https://mcp.api-inference.modelscope.net/8c9fe550938e4f/sse"


class ImageData:
    """Image data structure to match reference implementation"""

    def __init__(self, filename: str, data: str, url: str = None):
        self.filename = filename
        self.data = data  # base64 data
        self.url = url  # uploaded URL


async def _open_available_mcp_servers(server_defs):
    """
    Open only the MCP servers that are currently reachable.

    Hosted MCP endpoints are useful, but they should not take down the whole
    chat path when one remote server is temporarily unavailable.
    """
    exit_stack = AsyncExitStack()
    active_servers = []

    try:
        for server_name, server in server_defs:
            try:
                await asyncio.wait_for(
                    exit_stack.enter_async_context(server),
                    timeout=MCP_CONNECT_TIMEOUT_SECONDS,
                )
                active_servers.append(server)
                log.info(f"[MCP] Connected to {server_name} server")
            except Exception as server_error:
                log.warning(
                    f"[MCP] Skipping unavailable {server_name} server: {server_error}"
                )

        return exit_stack, active_servers
    except Exception:
        await exit_stack.aclose()
        raise


def _content_to_list(content: Any) -> List[Any]:
    if content is None:
        return []
    if isinstance(content, list):
        return list(content)
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [{"type": "text", "text": str(content)}]


def _merge_message_content(existing: Any, incoming: Any) -> Any:
    if isinstance(existing, str) and isinstance(incoming, str):
        return f"{existing}\n\n{incoming}" if existing else incoming

    merged_parts = _content_to_list(existing)
    incoming_parts = _content_to_list(incoming)
    if merged_parts and incoming_parts:
        last_part = merged_parts[-1]
        first_part = incoming_parts[0]
        if (
            isinstance(last_part, dict)
            and isinstance(first_part, dict)
            and last_part.get("type") == "text"
            and first_part.get("type") == "text"
        ):
            last_part = dict(last_part)
            last_part["text"] = (
                f"{last_part.get('text', '')}\n\n{first_part.get('text', '')}".strip()
            )
            merged_parts[-1] = last_part
            incoming_parts = incoming_parts[1:]
    return merged_parts + incoming_parts


def _normalize_message_roles(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive user/assistant turns so OpenAI-compatible backends that
    require alternating roles do not reject the conversation history.
    """
    normalized: List[Dict[str, Any]] = []

    for msg in msgs:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            normalized.append(msg)
            continue

        if (
            normalized
            and normalized[-1].get("role") == role
            and role in {"user", "assistant"}
        ):
            merged_msg = dict(normalized[-1])
            merged_msg["content"] = _merge_message_content(
                normalized[-1].get("content"),
                msg.get("content"),
            )
            normalized[-1] = merged_msg
            continue

        normalized.append(dict(msg))

    return normalized


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item.get("text")))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts).strip()
    if content is None:
        return ""
    return str(content)


def _latest_user_message_text(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return _content_to_text(message.get("content"))
    return ""


def _should_passthrough_rag_agent(
    messages: List[Dict[str, Any]], images: Optional[List[ImageData]] = None
) -> bool:
    """Only route low-risk conversational chat to the remote /agent endpoint."""
    if images:
        return False

    latest_user_text = _latest_user_message_text(messages)
    if not latest_user_text:
        return False

    lowered = latest_user_text.lower()
    local_handling_markers = (
        "create",
        "generate",
        "make",
        "draw",
        "render",
        "workflow",
        "node",
        "prompt",
        "image",
        "video",
        "lora",
        "controlnet",
        "control net",
        "canvas",
        "current workflow",
        "modify",
        "update",
        "rewrite",
        "fix",
        "change",
        "search_workflows",
        "diagnose_image",
        "recall_workflow",
        "gen_workflow",
        "워크플로우",
        "이미지",
        "생성",
        "만들",
        "그려",
        "렌더",
        "노드",
        "수정",
        "업데이트",
        "바꿔",
        "고쳐",
        "현재 워크플로우",
        "검색",
        "画布",
        "工作流",
        "生成",
        "创建",
        "修改",
        "更新",
        "添加",
        "修复",
    )
    return not any(marker in lowered for marker in local_handling_markers)


def _is_explicit_web_search_request(messages: List[Dict[str, Any]]) -> bool:
    latest_user_text = _latest_user_message_text(messages)
    if not latest_user_text:
        return False

    lower_text = latest_user_text.lower()
    markers = (
        "web search",
        "search the web",
        "search online",
        "browse the web",
        "browse web",
        "internet search",
        "bing search",
        "use bing",
        "웹검색",
        "웹 검색",
        "웹에서",
        "인터넷 검색",
        "인터넷에서",
        "빙 검색",
        "bing으로 검색",
    )
    return any(marker in lower_text for marker in markers)


def _should_attach_external_mcp(messages: List[Dict[str, Any]]) -> bool:
    enabled = _env_flag("BING_MCP_ENABLED", False)
    if not enabled:
        return False

    require_explicit = _env_flag("BING_MCP_REQUIRE_EXPLICIT_WEB_SEARCH", True)
    if not require_explicit:
        return True

    return _is_explicit_web_search_request(messages)


def _extract_json_object_slice(text: str, start_index: int) -> tuple[str, int]:
    if start_index >= len(text) or text[start_index] != "{":
        raise ValueError("Expected JSON object starting with '{'")

    depth = 0
    in_string = False
    escape = False

    for index in range(start_index, len(text)):
        char = text[index]

        if escape:
            escape = False
            continue

        if char == "\\" and in_string:
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_index : index + 1], index + 1

    raise ValueError("Unterminated JSON object in pseudo tool call")


def _extract_pseudo_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    cursor = 0

    while cursor < len(text):
        position = text.find("[ARGS]", cursor)
        if position == -1:
            break

        # Find the tool name preceding [ARGS]
        start_idx = position - 1
        while start_idx >= 0 and (text[start_idx].isalnum() or text[start_idx] == '_'):
            start_idx -= 1

        tool_name = text[start_idx+1:position]
        if not tool_name:
            cursor = position + 6
            continue

        marker_end = position + 6
        json_start = marker_end
        while json_start < len(text) and text[json_start].isspace():
            json_start += 1

        if json_start >= len(text) or text[json_start] != "{":
            cursor = marker_end
            continue

        try:
            raw_json, call_end = _extract_json_object_slice(text, json_start)
            calls.append(
                {
                    "name": tool_name,
                    "args": json.loads(raw_json),
                    "start": start_idx + 1,
                    "end": call_end,
                }
            )
            cursor = call_end
        except Exception as parse_error:
            log.warning(
                f"[MCP] Failed to parse pseudo tool call for {tool_name}: {parse_error}"
            )
            cursor = marker_end

    return calls


def _strip_pseudo_tool_calls(text: str, calls: List[Dict[str, Any]]) -> str:
    if not calls:
        return text

    parts: List[str] = []
    cursor = 0
    for call in calls:
        parts.append(text[cursor : call["start"]])
        cursor = call["end"]
    parts.append(text[cursor:])
    return "".join(parts).strip()


async def _build_mcp_tool_server_map(
    server_list: List[MCPServerSse],
) -> Dict[str, MCPServerSse]:
    tool_server_map: Dict[str, MCPServerSse] = {}

    for server in server_list:
        try:
            tools = await server.list_tools()
            for tool in tools:
                tool_name = getattr(tool, "name", None) or (
                    tool.get("name") if isinstance(tool, dict) else None
                )
                if tool_name and tool_name not in tool_server_map:
                    tool_server_map[tool_name] = server
        except Exception as tool_error:
            log.warning(f"[MCP] Failed to list tools from {server.name}: {tool_error}")

    return tool_server_map


def _extract_text_from_mcp_result(call_result: Any) -> str:
    parts: List[str] = []
    for item in getattr(call_result, "content", []) or []:
        item_type = getattr(item, "type", None)
        if item_type == "text":
            item_text = getattr(item, "text", "")
            if item_text:
                parts.append(item_text)
    return "\n".join(parts).strip()


def _parse_tool_result_payload(
    tool_name: str, tool_output_data: Any
) -> tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]:
    workflow_update_ext = None

    if isinstance(tool_output_data, dict):
        raw_keys = sorted(tool_output_data.keys())
        log.info(f"[MCP] Raw bridged payload keys for '{tool_name}': {raw_keys}")

        # FastMCP commonly wraps tool output as {"result": "<json-or-text>"}.
        # If we do not unwrap this envelope, successful tools look like
        # data=None/ext=None and get misclassified as failures.
        if "result" in tool_output_data and not any(
            key in tool_output_data for key in ("answer", "data", "ext", "text")
        ):
            wrapped_result = tool_output_data.get("result")
            if isinstance(wrapped_result, str):
                try:
                    tool_output_data = json.loads(wrapped_result)
                    log.info(
                        f"[MCP] Unwrapped FastMCP result envelope for '{tool_name}' "
                        f"into {type(tool_output_data).__name__}"
                    )
                except json.JSONDecodeError:
                    tool_output_data = {"answer": wrapped_result}
                    log.info(
                        f"[MCP] Treated non-JSON FastMCP result envelope for '{tool_name}' as answer text"
                    )
            else:
                tool_output_data = wrapped_result
                log.info(
                    f"[MCP] Unwrapped non-string FastMCP result envelope for '{tool_name}' "
                    f"into {type(tool_output_data).__name__}"
                )

        tool_ext = tool_output_data.get("ext")
        if isinstance(tool_ext, list):
            for ext_item in tool_ext:
                if ext_item.get("type") in {
                    "workflow_update",
                    "param_update",
                    "workflow",
                }:
                    workflow_update_ext = tool_ext
                    break

        if "text" in tool_output_data and tool_output_data.get("text") is not None:
            candidate_output = tool_output_data.get("text")
        else:
            candidate_output = tool_output_data
    else:
        tool_ext = None
        candidate_output = tool_output_data

    if isinstance(candidate_output, str):
        try:
            parsed_output = json.loads(candidate_output)
        except json.JSONDecodeError:
            parsed_output = {"answer": candidate_output}
    else:
        parsed_output = candidate_output

    if isinstance(parsed_output, dict):
        result = {
            "answer": parsed_output.get("answer")
            or (candidate_output if isinstance(candidate_output, str) else None),
            "data": parsed_output.get("data"),
            "ext": parsed_output.get("ext") or tool_ext,
            "content_dict": parsed_output,
        }
    elif isinstance(parsed_output, list):
        result = {
            "answer": None,
            "data": parsed_output,
            "ext": tool_ext,
            "content_dict": parsed_output,
        }
    else:
        result = {
            "answer": str(parsed_output),
            "data": None,
            "ext": tool_ext,
            "content_dict": None,
        }

    log.info(
        f"[MCP] Parsed bridged tool result for '{tool_name}': "
        f"data={len(result['data']) if result['data'] else 0}, ext={result['ext'] is not None}"
    )
    return result, workflow_update_ext


async def _bridge_pseudo_tool_calls(
    current_text: str,
    server_list: List[MCPServerSse],
    tool_results: Dict[str, Dict[str, Any]],
    workflow_tools_called: set,
    workflow_update_ext: Optional[List[Dict[str, Any]]],
) -> tuple[str, Optional[List[Dict[str, Any]]], bool]:
    pseudo_calls = _extract_pseudo_tool_calls(current_text)
    if not pseudo_calls:
        return current_text, workflow_update_ext, False

    log.info(f"[MCP] Found {len(pseudo_calls)} pseudo tool call(s) in assistant text")
    tool_server_map = await _build_mcp_tool_server_map(server_list)
    bridged_any = False
    bridged_answer_texts: List[str] = []

    for pseudo_call in pseudo_calls:
        tool_name = pseudo_call["name"]
        tool_args = pseudo_call["args"]

        if tool_name in tool_results:
            continue

        try:
            if tool_name in LOCAL_TOOLS_REGISTRY:
                log.info(f"[MCP] Bridging pseudo local tool '{tool_name}' with args: {tool_args}")
                local_tool = LOCAL_TOOLS_REGISTRY[tool_name]
                local_json = json.dumps(tool_args, ensure_ascii=False)
                local_ctx = ToolContext(
                    context=None,
                    usage=Usage(),
                    tool_name=tool_name,
                    tool_call_id=f"pseudo-{tool_name}",
                    tool_arguments=local_json,
                )
                raw_payload = await local_tool.on_invoke_tool(local_ctx, local_json)
            else:
                server = tool_server_map.get(tool_name)
                if server is None:
                    log.warning(
                        f"[MCP] No connected MCP server exposes tool '{tool_name}'"
                    )
                    continue

                workflow_tools_called.add(tool_name)
                log.info(
                    f"[MCP] Bridging pseudo tool call '{tool_name}' with args: {tool_args}"
                )
                call_result = await server.call_tool(tool_name, tool_args)
                raw_payload = getattr(call_result, "structuredContent", None)
                if raw_payload is None:
                    raw_payload = _extract_text_from_mcp_result(call_result)

                if (
                    getattr(call_result, "isError", False)
                    and isinstance(raw_payload, str)
                    and raw_payload
                ):
                    raw_payload = {"answer": raw_payload}

            parsed_result, bridged_workflow_update_ext = _parse_tool_result_payload(
                tool_name, raw_payload
            )
            tool_results[tool_name] = parsed_result
            if bridged_workflow_update_ext:
                workflow_update_ext = bridged_workflow_update_ext
            answer_text = parsed_result.get("answer")
            if isinstance(answer_text, str) and answer_text.strip():
                bridged_answer_texts.append(answer_text.strip())
            bridged_any = True
        except Exception as bridge_error:
            log.error(
                f"[MCP] Failed to bridge pseudo tool call '{tool_name}': {bridge_error}"
            )
            log.error(f"[MCP] Bridge traceback: {traceback.format_exc()}")
            tool_results[tool_name] = {
                "answer": f"Bridged tool call failed: {bridge_error}",
                "data": None,
                "ext": None,
                "content_dict": None,
            }

    cleaned_text = _strip_pseudo_tool_calls(current_text, pseudo_calls)
    if not cleaned_text and bridged_answer_texts:
        cleaned_text = "\n\n".join(bridged_answer_texts)
    return cleaned_text, workflow_update_ext, bridged_any


import os
import time
import pathlib

_LEGACY_AGENT_INVOKE = os.getenv("LEGACY_AGENT_INVOKE") == "1"

async def comfyui_agent_invoke(
    messages: List[Dict[str, Any]], images: List[ImageData] = None
):
    log.debug(f"[comfyui_agent_invoke] ENTER messages={len(messages)} images={len(images) if images else 0}")

    capture_dir = pathlib.Path(__file__).parent.parent.parent / "tests" / "regression" / "golden" / "capture"
    capture_dir.mkdir(parents=True, exist_ok=True)
    fn = capture_dir / f"{int(time.time()*1000)}.json"

    capture = {
        "input": {
            "kwargs": {
                "messages": messages,
                "images": [{"url": getattr(img, "url", ""), "b64": "..." if getattr(img, "b64", None) else None} for img in images] if images else None
            }
        },
        "expected_output": []
    }

    try:
        if _LEGACY_AGENT_INVOKE:
            generator = _legacy_comfyui_agent_invoke(messages, images)
        else:
            generator = _new_comfyui_agent_invoke(messages, images)

        async for chunk in generator:
            capture["expected_output"].append(chunk)
            yield chunk

        fn.write_text(json.dumps(capture, default=str, indent=2))
        log.debug("[comfyui_agent_invoke] EXIT success")
    except Exception as e:
        log.error(f"[comfyui_agent_invoke] ERROR {e}")
        capture["error"] = str(e)
        fn.write_text(json.dumps(capture, default=str, indent=2))
        raise

# 1. Input Validation
def _validate_invoke_input() -> tuple[str, dict]:
    session_id = get_session_id()
    if not session_id:
        raise ValueError("No session_id found in request context")
    config = get_config()
    if not config:
        raise ValueError("No config found in request context")
    return session_id, config

# 2. Context Preparation
async def _prepare_invoke_context(messages: List[Dict[str, Any]], session_id: str) -> tuple[List[Dict[str, Any]], bool, Any]:
    last_user_msg = _latest_user_message_text(messages)

    if last_user_msg:
        gen_keywords = ["만들", "생성", "그려", "추가", "바꿔", "수정", "create", "generate", "draw", "추천", "workflow", "워크플로우", "변경", "해줘"]
        is_generation_intent = any(kw in last_user_msg.lower() for kw in gen_keywords)

        if not is_generation_intent:
            try:
                log.info(f"[MCP] Routing simple chat directly to RAG agent for query: {last_user_msg}")
                response_text = await pass_through_rag_agent(last_user_msg, session_id=session_id)
                ext_with_finished = {"data": None, "finished": True}
                return messages, True, (response_text, ext_with_finished)
            except Exception as e:
                log.error(f"[MCP] Error passing to RAG agent: {e}")
        else:
            log.info(f"[MCP] Keeping local tool path for action-oriented query: {last_user_msg[:160]}")

    def _strip_trailing_whitespace(msgs):
        cleaned = []
        for msg in msgs:
            role = msg.get("role")
            if role != "assistant":
                cleaned.append(msg)
                continue
            msg_copy = dict(msg)
            content = msg_copy.get("content")
            if isinstance(content, str):
                msg_copy["content"] = content.rstrip()
            elif isinstance(content, list):
                new_content = []
                for part in content:
                    if isinstance(part, dict):
                        part_copy = dict(part)
                        text_val = part_copy.get("text")
                        if isinstance(text_val, str):
                            part_copy["text"] = text_val.rstrip()
                        new_content.append(part_copy)
                    else:
                        new_content.append(part)
                msg_copy["content"] = new_content
            cleaned.append(msg_copy)
        return cleaned

    messages = _strip_trailing_whitespace(messages)
    log.info(f"[MCP] Original messages count: {len(messages)}")
    messages = message_memory_optimize(session_id, messages)
    messages = _normalize_message_roles(messages)
    log.info(f"[MCP] Optimized messages count: {len(messages)}")
    return messages, False, None

# 3. Setup Tool Loop (MCP + Handoff)
async def _setup_tool_loop(messages: List[Dict[str, Any]], session_id: str, config: dict) -> tuple[Any, Any, Any]:
    copilot_mcp_url = os.getenv("COPILOT_MCP_URL") or (BACKEND_BASE_URL + "/mcp-server/mcp")
    copilot_headers = {"X-Session-Id": session_id}
    internal_api_key = get_comfyui_copilot_api_key()
    if internal_api_key:
        copilot_headers["Authorization"] = f"Bearer {internal_api_key}"

    mcp_server = MCPServerSse(
        params={"url": copilot_mcp_url, "timeout": 300.0, "headers": copilot_headers},
        cache_tools_list=True, client_session_timeout_seconds=300.0,
    )
    server_defs = [("copilot_mcp", mcp_server)]

    if _should_attach_external_mcp(messages):
        bing_mcp_url = (os.getenv("BING_MCP_URL") or BING_MCP_DEFAULT_URL).strip()
        bing_mcp_api_key = (os.getenv("BING_MCP_API_KEY") or "").strip()
        if bing_mcp_api_key:
            bing_server = MCPServerSse(
                params={"url": bing_mcp_url, "timeout": 300.0, "headers": {"Authorization": f"Bearer {bing_mcp_api_key}"}},
                cache_tools_list=True, client_session_timeout_seconds=300.0,
            )
            server_defs.append(("bing_mcp", bing_server))

    exit_stack, server_list = await _open_available_mcp_servers(server_defs)

    workflow_rewrite_agent_instance = create_workflow_rewrite_agent()
    class HandoffRewriteData(BaseModel):
        latest_rewrite_intent: str

    async def on_handoff(ctx, input_data: HandoffRewriteData):
        get_rewrite_context().rewrite_intent = input_data.latest_rewrite_intent
        log.info(f"Rewrite agent called with intent: {input_data.latest_rewrite_intent}")

    def rewrite_handoff_input_filter(data: HandoffInputData) -> HandoffInputData:
        intent = get_rewrite_context().rewrite_intent
        new_history = ()
        try:
            for item in data.input_history:
                if hasattr(item, "role") and getattr(item, "role") == "user":
                    if hasattr(item, "model_copy"):
                        new_history = (item.model_copy(update={"content": intent}),)
                        break
                    elif hasattr(item, "copy"):
                        new_history = (item.copy(update={"content": intent}),)
                        break
        except Exception:
            pass
        return HandoffInputData(
            input_history=new_history,
            pre_handoff_items=(),
            new_items=tuple(data.new_items),
        )

    handoff_rewrite = handoff(
        agent=workflow_rewrite_agent_instance,
        input_type=HandoffRewriteData,
        input_filter=rewrite_handoff_input_filter,
        on_handoff=on_handoff,
    )

    if DISABLE_WORKFLOW_GEN:
        w_inst = "**CASE 3: SEARCH WORKFLOW**\nIF the user wants to find or generate a NEW workflow.\n- Keywords: \"create\", \"generate\", \"search\", \"find\", \"recommend\", \"生成\", \"查找\", \"推荐\".\n- Action: Use `recall_workflow`.\n"
        w_cons = "- [Critical!] When the user's intent is to get workflows or generate images with specific requirements, you MUST call `recall_workflow` tool to find existing similar workflows.\n"
    else:
        w_inst = "**CASE 3: CREATE NEW / SEARCH WORKFLOW**\nIF the user wants to find or generate a NEW workflow from scratch.\n- Keywords: \"create\", \"generate\", \"search\", \"find\", \"recommend\", \"生成\", \"查找\", \"推荐\".\n- Action: Use `recall_workflow` AND `gen_workflow`.\n"
        w_cons = "- [Critical!] When the user's intent is to get workflows or generate images with specific requirements, you MUST ALWAYS call BOTH recall_workflow tool AND gen_workflow tool to provide comprehensive workflow options. Never call just one of these tools - both are required for complete workflow assistance. First call recall_workflow to find existing similar workflows, then call gen_workflow to generate new workflow options.\n"

    # WO-COPILOT-IMAGE-GEN-HOTFIX-MCP-CLIENT §4.A — action-path model override.
    # Why: chat dropdown model_select (e.g. qwen-coder) otherwise leaks into image/workflow
    # action path and overrides workflow_llm_model, triggering context-length failures.
    # HOTFIX scope: on action path, workflow_llm_model unconditionally wins. Proper 2-key
    # split (chat_model_select vs workflow_llm_model) lives in WO-COPILOT-MODEL-CONTRACT-SPLIT.
    effective_config = dict(config or {})
    _last_user_for_path = _latest_user_message_text(messages)
    if _last_user_for_path:
        _gen_kw = ["만들", "생성", "그려", "추가", "바꿔", "수정", "create", "generate", "draw", "추천", "workflow", "워크플로우", "변경", "해줘"]
        _is_action_path = any(k in _last_user_for_path.lower() for k in _gen_kw)
    else:
        _is_action_path = False
    if _is_action_path and effective_config.get("workflow_llm_model"):
        _orig_sel = effective_config.get("model_select")
        effective_config["model_select"] = effective_config["workflow_llm_model"]
        log.info(
            f"[MCP] Action-path model override: model_select {_orig_sel!r} -> workflow_llm_model {effective_config['model_select']!r} [WO-HOTFIX-A]"
        )

    agent = create_agent(
        name="ComfyUI-Copilot",
        instructions=f"""You are a powerful AI assistant for designing image processing workflows, capable of automating problem-solving using tools and commands.

When handing off to workflow rewrite agent or other agents, this session ID should be used for workflow data management.

### PRIMARY DIRECTIVE: INTENT CLASSIFICATION & HANDOFF
You act as a router. Your FIRST step is to classify the user's intent.

### TOOL-CALL RELIABILITY OVERRIDE (CONTEXT-TRIM SAFE)
The conversation history may be truncated for brevity and may contain ZERO tool calls/tool results.
- You MUST NOT treat "no prior tool message" as a reason to skip tool usage.
- If a CASE below requires a tool call or handoff, you MUST execute it even if you think you already know the answer.
- If a CASE below requires a tool call or handoff, your IMMEDIATE next assistant turn MUST be that tool call/handoff (do not output any natural-language explanation first).

**CASE 1: MODIFY/UPDATE/FIX CURRENT WORKFLOW (HIGHEST PRIORITY)**
IF the user wants to:
- Modify, enhance, update, or fix the CURRENT workflow/canvas.
- Add nodes/features to the CURRENT workflow (e.g., "add LoRA", "add controlnet", "fix the error").
- Change parameters in the CURRENT workflow.
- Keywords: "modify", "update", "add", "change", "fix", "current", "canvas", "修改", "更新", "添加", "画布", "加一个", "换一个", "调一下".

**ACTION:**
- You MUST IMMEDIATELY handoff to the `Workflow Rewrite Agent`.
- DO NOT call any other tools (like search_node, gen_workflow).
- DO NOT ask for more details. Just handoff.

**CASE 2: ANALYZE CURRENT WORKFLOW**
IF the user wants to:
- Analyze, explain, or understand the current workflow structure/logic.
- Ask questions about the current workflow (e.g., "how does this work?", "explain the workflow").
- Keywords: "analyze", "explain", "understand", "how it works", "workflow structure", "分析", "解释", "怎么工作的", "解读".

**ACTION:
- You MUST call `get_current_workflow` to retrieve the workflow details.
- Then, based on the returned workflow data, provide a detailed analysis or explanation to the user.

{w_inst}

### CONSTRAINT CHECKLIST
You must adhere to the following constraints to complete the task:

- **Tool compliance is mandatory**: If the selected CASE requires a tool/handoff, you MUST perform it. Do not answer directly without performing the required tool/handoff.
- [Important!] Respond must in the language used by the user in their question. Regardless of the language returned by the tools being called, please return the results based on the language used in the user's query. For example, if user ask by English, you must return
- Ensure that the commands or tools you invoke are within the provided tool list.
- If the execution of a command or tool fails, try changing the parameters or their format before attempting again.
- Your generated responses must follow the factual information given above. Do not make up information.
- If the result obtained is incorrect, try rephrasing your approach.
- Do not query for already obtained information repeatedly. If you successfully invoked a tool and obtained relevant information, carefully confirm whether you need to invoke it again.
- Ensure that the actions you generate can be executed accurately. Actions may include specific methods and target outputs.
- When you encounter a concept, try to obtain its precise definition and analyze what inputs can yield specific values for it.
- When generating a natural language query, include all known information in the query.
- Before performing any analysis or calculation, ensure that all sub-concepts involved have been defined.
- Printing the entire content of a file is strictly prohibited, as such actions have high costs and can lead to unforeseen consequences.
- Ensure that when you call a tool, you have obtained all the input variables for that tool, and do not fabricate any input values for it.
- Respond with markdown, using a minimum of 3 heading levels (H3, H4, H5...), and when including images use the format ![alt text](url),
{w_cons}
- When the user's intent is to query, return the query result directly without attempting to assist the user in performing operations.
- When the user's intent is to get prompts for image generation (like Stable Diffusion). Use specific descriptive language with proper weight modifiers (e.g., (word:1.2)), prefer English terms, and separate elements with commas. Include quality terms (high quality, detailed), style specifications (realistic, anime), lighting (cinematic, golden hour), and composition (wide shot, close up) as needed. When appropriate, include negative prompts to exclude unwanted elements. Return words divided by commas directly without any additional text.
- If you cannot find the information needed to answer a query, consider using bing_search to obtain relevant information. For example, if search_node tool cannot find the node, you can use bing_search to obtain relevant information about those nodes or components.
- If search_node tool cannot find the node, you MUST use bing_search to obtain relevant information about those nodes or components.

- **ERROR MESSAGE ANALYSIS** - When a user pastes specific error text/logs (containing terms like "Failed", "Error", "Traceback", or stack traces), prioritize providing troubleshooting help rather than invoking search tools. Follow these steps:
  1. Analyze the error to identify the root cause (error type, affected component, missing dependencies, etc.)
  2. Explain the issue in simple terms
  3. Provide concrete, executable solutions including:
     - Specific shell commands to fix the issue (e.g., `git pull`, `pip install`, file path corrections)
     - Code snippets if applicable
     - Configuration file changes with exact paths and values
  4. If the error relates to a specific ComfyUI extension or node, include instructions for:
     - Updating the extension (`cd path/to/extension && git pull`)
     - Reinstalling dependencies
     - Alternative approaches if the extension is problematic
        """,
        mcp_servers=server_list,
        handoffs=[handoff_rewrite],
        tools=[get_current_workflow],
        config=effective_config,
    )

    return exit_stack, server_list, agent

# 4. Stream LLM Response
class InvokeState:
    def __init__(self):
        self.current_text = ""
        self.ext = None
        self.tool_results = {}
        self.workflow_tools_called = set()
        self.last_yield_length = 0
        self.tool_call_queue = []
        self.current_tool_call = None
        self.workflow_update_ext = None
        self.handoff_occurred = False
        # WO-COPILOT-IMAGE-GEN-HOTFIX-MCP-CLIENT §4.C: terminal error surfacing
        self.all_retries_failed = False
        self.last_error_info = None  # dict: {error_class, stage, model, detail}


def _hotfix_extract_model_name(agent, config_hint=None) -> str:
    """WO-HOTFIX §4.C helper: best-effort model name for user-facing error text."""
    import os as _os
    for attr_path in ("config", "model", "model_select"):
        try:
            val = getattr(agent, attr_path, None)
            if isinstance(val, dict):
                m = val.get("model_select") or val.get("workflow_llm_model") or val.get("model")
                if m:
                    return str(m)
            elif isinstance(val, str) and val:
                return val
        except Exception:
            pass
    if isinstance(config_hint, dict):
        m = config_hint.get("model_select") or config_hint.get("workflow_llm_model")
        if m:
            return str(m)
    return _os.getenv("WORKFLOW_LLM_MODEL") or _os.getenv("OPENAI_MODEL") or "unknown"

async def _stream_llm_response(agent, messages: List[Dict[str, Any]]):
    from agents import Runner, set_tracing_disabled, set_default_openai_api
    set_tracing_disabled(False)
    set_default_openai_api("chat_completions")

    state = InvokeState()

    async def process_stream_events(stream_result):
        try:
            async for event in stream_result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    delta_text = event.data.delta
                    if delta_text:
                        state.current_text += delta_text
                        if len(state.current_text) > state.last_yield_length:
                            state.last_yield_length = len(state.current_text)
                            yield (state.current_text, None)

                elif event.type == "agent_updated_stream_event":
                    new_agent_name = event.new_agent.name
                    log.info(f"Handoff to: {new_agent_name}")
                    if state.handoff_occurred:
                        handoff_text = f"\n▸ **Switching to {new_agent_name}**\n\n"
                        state.current_text += handoff_text
                        state.last_yield_length = len(state.current_text)
                        yield (state.current_text, None)
                    state.handoff_occurred = True

                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        tool_name = getattr(event.item.raw_item, "name", "unknown_tool")
                        state.tool_call_queue.append(tool_name)
                        log.info(f"-- Tool '{tool_name}' was called")
                        if tool_name in ["recall_workflow", "gen_workflow"]:
                            state.workflow_tools_called.add(tool_name)

                    elif event.item.type == "tool_call_output_item":
                        log.info(f"-- Tool output: {event.item.output}")
                        tool_output_data_str = str(event.item.output)

                        tool_name = state.tool_call_queue.pop(0) if state.tool_call_queue else "unknown_tool"

                        try:
                            if isinstance(event.item.output, dict):
                                tool_output_data = event.item.output
                            elif isinstance(event.item.output, str):
                                tool_output_data = json.loads(event.item.output)
                            else:
                                tool_output_data = json.loads(str(event.item.output))

                            if "result" in tool_output_data and isinstance(tool_output_data["result"], str):
                                try:
                                    tool_output_data = json.loads(tool_output_data["result"])
                                except json.JSONDecodeError:
                                    tool_output_data = {"text": tool_output_data["result"]}

                            if "ext" in tool_output_data and tool_output_data["ext"]:
                                tool_ext_items = tool_output_data["ext"]
                                for ext_item in tool_ext_items:
                                    if ext_item.get("type") in {"workflow_update", "param_update", "workflow"}:
                                        state.workflow_update_ext = tool_ext_items
                                        log.info(f"-- Captured workflow tool ext from tool output: {len(tool_ext_items)} items")
                                        break

                            if "text" in tool_output_data and tool_output_data.get("text"):
                                parsed_output = json.loads(tool_output_data["text"])
                                if isinstance(parsed_output, dict):
                                    answer = parsed_output.get("answer")
                                    data = parsed_output.get("data")
                                    tool_ext = parsed_output.get("ext")
                                else:
                                    answer = None
                                    data = parsed_output if isinstance(parsed_output, list) else None
                                    tool_ext = None

                                state.tool_results[tool_name] = {
                                    "answer": answer, "data": data, "ext": tool_ext, "content_dict": parsed_output,
                                }
                        except (json.JSONDecodeError, TypeError):
                            state.tool_results[tool_name] = {
                                "answer": tool_output_data_str, "data": None, "ext": None, "content_dict": None,
                            }
        except Exception as e:
            log.error(f"Unexpected streaming error: {e}")
            raise e

    retry_count = 0
    max_retries = 3
    while retry_count <= max_retries:
        # WO-COPILOT-IMAGE-GEN-HOTFIX-MCP-CLIENT §4.B — reset per iteration so the
        # initial Handoff event of each retry does NOT emit a "▸ Switching to ..."
        # banner. handoff_occurred is the only per-attempt flag; current_text /
        # tool_results / workflow_update_ext must stay cumulative for partial progress.
        state.handoff_occurred = False
        try:
            result = Runner.run_streamed(agent, input=messages, max_turns=30)
            async for stream_chunk in process_stream_events(result):
                yield stream_chunk, None
            break
        except RateLimitError as rl_err:
            error_body = getattr(rl_err, "body", {})
            error_msg = error_body.get("message") if isinstance(error_body, dict) else None
            final_error_msg = error_msg or "Rate limit exceeded, please try again later."
            # WO-HOTFIX §4.C
            state.all_retries_failed = True
            state.last_error_info = {
                "error_class": "rate_limit",
                "stage": "llm_router_call",
                "model": _hotfix_extract_model_name(agent),
                "detail": final_error_msg,
            }
            yield (final_error_msg, None), None
            yield None, state  # WO-HOTFIX §4.C — propagate state to caller before exit
            return
        except (AttributeError, TypeError, ConnectionError, OSError) as stream_error:
            error_msg = str(stream_error)
            should_retry = "'NoneType' object has no attribute 'strip'" in error_msg or "Connection broken" in error_msg or "socket hang up" in error_msg
            retry_count += 1
            if should_retry and retry_count <= max_retries:
                if state.current_text:
                    yield (state.current_text, None), None
                await asyncio.sleep(min(2 ** (retry_count - 1), 10))
            else:
                # WO-HOTFIX §4.C — terminal error: record so _build_invoke_response can surface it
                _err_class = type(stream_error).__name__
                if "context_length" in error_msg.lower() or "context length" in error_msg.lower():
                    _err_class = "context_length_exceeded"
                elif "connection" in error_msg.lower():
                    _err_class = "connection"
                state.all_retries_failed = True
                state.last_error_info = {
                    "error_class": _err_class,
                    "stage": "llm_router_call",
                    "model": _hotfix_extract_model_name(agent),
                    "detail": error_msg,
                }
                yield (f"I apologize, but an error occurred while processing your request: {error_msg}", None), None
                yield None, state  # WO-HOTFIX §4.C — propagate state to caller before exit
                return
        except Exception as _exc:  # WO-HOTFIX §4.C — capture last error on exhaustion
            retry_count += 1
            if retry_count > max_retries:
                state.all_retries_failed = True
                state.last_error_info = {
                    "error_class": type(_exc).__name__,
                    "stage": "llm_router_call",
                    "model": _hotfix_extract_model_name(agent),
                    "detail": str(_exc),
                }
                break
            await asyncio.sleep(1)

    yield None, state

# 5. Handle Invoke Error
def _handle_invoke_error(error: Exception) -> tuple[str, dict]:
    log.error(f"Error in comfyui_agent_invoke: {error}")
    log.error(f"Traceback: {traceback.format_exc()}")
    return (str(error), {"data": None, "finished": True})

# 6. Final Build
def _build_invoke_response(state: InvokeState, bridged_pseudo_tools: bool) -> tuple[str, dict]:
    # WO-COPILOT-IMAGE-GEN-HOTFIX-MCP-CLIENT §4.C — surface terminal LLM error with
    # error_class + model + stage so users see a clear diagnostic instead of ext=null.
    if getattr(state, "all_retries_failed", False) and getattr(state, "last_error_info", None):
        _info = state.last_error_info
        _err_text = (
            f"⚠️ LLM 호출 실패: {_info.get('model', 'unknown')} — "
            f"{_info.get('error_class', 'unknown_error')} ({_info.get('stage', 'unknown_stage')}). "
            f"재시도하거나 모델을 바꿔주세요."
        )
        _detail = _info.get("detail")
        if _detail:
            _err_text += f"\n\n<details><summary>details</summary>\n{_detail}\n</details>"
        return (_err_text, {"data": None, "finished": True})

    # Error fallback if needed
    if bridged_pseudo_tools and not state.current_text:
        failed_tools = [n for n in sorted(state.tool_results.keys()) if n != "_message_output_ext" and not (state.tool_results[n].get("data") or state.tool_results[n].get("answer"))]
        state.current_text = f"(도구 호출 {', '.join(failed_tools)} 이(가) 결과를 반환하지 못했습니다. 잠시 후 재시도해 주세요.)" if failed_tools else "(도구 호출은 처리했지만 사용자에게 보여줄 텍스트가 없습니다.)"

    workflow_tools_found = [t for t in ["recall_workflow", "gen_workflow"] if t in state.tool_results]
    finished = False
    ext = None

    if workflow_tools_found:
        if "recall_workflow" in state.tool_results and "gen_workflow" in state.tool_results:
            successful_workflows = []
            if state.tool_results["recall_workflow"]["data"]:
                successful_workflows.extend(state.tool_results["recall_workflow"]["data"])
            if state.tool_results["gen_workflow"]["data"]:
                successful_workflows.insert(0, *state.tool_results["gen_workflow"]["data"])

            seen_ids = set()
            unique_workflows = []
            for w in successful_workflows:
                wid = w.get("id")
                if wid and wid not in seen_ids:
                    seen_ids.add(wid)
                    unique_workflows.append(w)
                elif not wid:
                    unique_workflows.append(w)

            if unique_workflows:
                ext = [{"type": "workflow", "data": unique_workflows}]
            finished = True
        elif "recall_workflow" in state.tool_results:
            if DISABLE_WORKFLOW_GEN:
                if state.tool_results["recall_workflow"]["data"]:
                    ext = [{"type": "workflow", "data": state.tool_results["recall_workflow"]["data"]}]
                finished = True
        elif "gen_workflow" in state.tool_results:
            if state.tool_results["gen_workflow"]["data"]:
                    ext = [{"type": "workflow", "data": state.tool_results["gen_workflow"]["data"]}]
            finished = True
    else:
        for tn, res in state.tool_results.items():
            if res["ext"]:
                ext = res["ext"]
                break
        finished = True

    final_ext = ext
    if state.workflow_update_ext:
        if isinstance(state.workflow_update_ext, list):
            final_ext = state.workflow_update_ext + (ext if ext else [])
        else:
            final_ext = [state.workflow_update_ext] + (ext if ext else [])

    return (state.current_text, {"data": final_ext, "finished": finished})

# Main Orchestrator Function
async def _new_comfyui_agent_invoke(messages: List[Dict[str, Any]], images: List[ImageData] = None):
    # Note: `images` is kept in signature for compatibility with conversational_api.py,
    # but it's not directly used here since images are already formatted inside `messages` by the caller.
    try:
        session_id, config = _validate_invoke_input()
        messages, handled, early_result = await _prepare_invoke_context(messages, session_id)
        if handled:
            yield early_result
            return

        exit_stack, server_list, agent = await _setup_tool_loop(messages, session_id, config)
        try:
            state = None
            async for chunk, s in _stream_llm_response(agent, messages):
                if chunk:
                    yield chunk
                if s:
                    state = s

            if state is None:
                state = InvokeState()

            state.current_text, state.workflow_update_ext, bridged = await _bridge_pseudo_tool_calls(
                current_text=state.current_text, server_list=server_list, tool_results=state.tool_results,
                workflow_tools_called=state.workflow_tools_called, workflow_update_ext=state.workflow_update_ext
            )

            yield _build_invoke_response(state, bridged)
        finally:
            await exit_stack.aclose()
    except Exception as e:
        yield _handle_invoke_error(e)


async def _legacy_comfyui_agent_invoke(
    messages: List[Dict[str, Any]], images: List[ImageData] = None
):
    """
    Invoke the ComfyUI agent with MCP tools and image support.

    This function mimics the behavior of the reference facade.py chat function,
    yielding (text, ext) tuples similar to the reference implementation.

    Args:
        messages: List of messages in OpenAI format [{"role": "user", "content": "..."}, ...]
        images: List of image data objects (optional)

    Yields:
        tuple: (text, ext) where text is accumulated text and ext is structured data
    """
    try:
        session_id = get_session_id()
        if not session_id:
            raise ValueError("No session_id found in request context")

        last_user_msg = _latest_user_message_text(messages)
        # Action-oriented requests (generation, workflow edits, diagnostics) must stay on
        # the local Copilot path so workflow ext data can flow back into ComfyUI.
        if last_user_msg:
            # Simple heuristic intent classification
            gen_keywords = [
                "만들",
                "생성",
                "그려",
                "추가",
                "바꿔",
                "수정",
                "create",
                "generate",
                "draw",
                "추천",
                "workflow",
                "워크플로우",
                "변경",
                "해줘",
            ]
            is_generation_intent = any(
                kw in last_user_msg.lower() for kw in gen_keywords
            )

            if not is_generation_intent:
                # Only low-risk conversational chat should bypass the local tool/execution path.
                try:
                    log.info(
                        f"[MCP] Routing simple chat directly to RAG agent for query: {last_user_msg}"
                    )
                    response_text = await pass_through_rag_agent(
                        last_user_msg, session_id=session_id
                    )
                    ext_with_finished = {"data": None, "finished": True}
                    yield (response_text, ext_with_finished)
                    return
                except Exception as e:
                    log.error(f"[MCP] Error passing to RAG agent: {e}")
                    # Fall back to local execution if pass-through fails.
            else:
                log.info(
                    f"[MCP] Keeping local tool path for action-oriented query: {last_user_msg[:160]}"
                )

        def _strip_trailing_whitespace_from_messages(
            msgs: List[Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
            cleaned: List[Dict[str, Any]] = []
            for msg in msgs:
                role = msg.get("role")
                # Only touch assistant messages to minimize impact
                if role != "assistant":
                    cleaned.append(msg)
                    continue

                msg_copy = dict(msg)
                content = msg_copy.get("content")

                # Simple string content
                if isinstance(content, str):
                    msg_copy["content"] = content.rstrip()
                # OpenAI / Agents style list content blocks
                elif isinstance(content, list):
                    new_content = []
                    for part in content:
                        if isinstance(part, dict):
                            part_copy = dict(part)
                            # Common text block key is "text"
                            text_val = part_copy.get("text")
                            if isinstance(text_val, str):
                                part_copy["text"] = text_val.rstrip()
                            new_content.append(part_copy)
                        else:
                            new_content.append(part)
                    msg_copy["content"] = new_content

                cleaned.append(msg_copy)

            return cleaned

        messages = _strip_trailing_whitespace_from_messages(messages)

        # Get session_id and config from request context
        session_id = get_session_id()
        config = get_config()

        if not session_id:
            raise ValueError("No session_id found in request context")
        if not config:
            raise ValueError("No config found in request context")

        # Optimize messages with memory compression
        log.info(f"[MCP] Original messages count: {len(messages)}")
        messages = message_memory_optimize(session_id, messages)
        messages = _normalize_message_roles(messages)
        log.info(
            f"[MCP] Optimized messages count: {len(messages)}, messages: {messages}"
        )

        # Create MCP server instances
        copilot_mcp_url = os.getenv("COPILOT_MCP_URL") or (
            BACKEND_BASE_URL + "/mcp-server/mcp"
        )

        copilot_headers = {"X-Session-Id": session_id}
        internal_api_key = get_comfyui_copilot_api_key()
        if internal_api_key:
            copilot_headers["Authorization"] = f"Bearer {internal_api_key}"
        else:
            log.warning(
                "[MCP] No internal Authorization token found for copilot_mcp requests."
            )

        mcp_server = MCPServerSse(
            params={
                "url": copilot_mcp_url,
                "timeout": 300.0,
                "headers": copilot_headers,
            },
            cache_tools_list=True,
            client_session_timeout_seconds=300.0,
        )

        server_defs = [
            ("copilot_mcp", mcp_server),
        ]

        external_mcp_enabled = _env_flag("BING_MCP_ENABLED", False)
        attach_external_mcp = _should_attach_external_mcp(messages)
        if attach_external_mcp:
            bing_mcp_url = (os.getenv("BING_MCP_URL") or BING_MCP_DEFAULT_URL).strip()
            bing_mcp_api_key = (os.getenv("BING_MCP_API_KEY") or "").strip()
            if not bing_mcp_api_key:
                log.warning(
                    "[MCP] BING_MCP_ENABLED=true but BING_MCP_API_KEY is missing. Skipping external MCP."
                )
            else:
                log.info(f"[MCP] External MCP enabled for this request: {bing_mcp_url}")
                bing_server = MCPServerSse(
                    params={
                        "url": bing_mcp_url,
                        "timeout": 300.0,
                        "headers": {"Authorization": f"Bearer {bing_mcp_api_key}"},
                    },
                    cache_tools_list=True,
                    client_session_timeout_seconds=300.0,
                )
                server_defs.append(("bing_mcp", bing_server))
        elif external_mcp_enabled:
            log.info(
                "[MCP] External MCP is enabled globally but skipped for this request (no explicit web-search intent)."
            )

        exit_stack, server_list = await _open_available_mcp_servers(server_defs)

        try:
            if not server_list:
                log.warning(
                    "[MCP] No remote MCP servers are reachable. Continuing with local chat only."
                )

            # 创建workflow_rewrite_agent实例 (session_id通过context获取)
            workflow_rewrite_agent_instance = create_workflow_rewrite_agent()

            class HandoffRewriteData(BaseModel):
                latest_rewrite_intent: str

            async def on_handoff(
                ctx: RunContextWrapper[None], input_data: HandoffRewriteData
            ):
                get_rewrite_context().rewrite_intent = input_data.latest_rewrite_intent
                log.info(
                    f"Rewrite agent called with intent: {input_data.latest_rewrite_intent}"
                )

            def rewrite_handoff_input_filter(
                data: HandoffInputData,
            ) -> HandoffInputData:
                """Filter to replace message history with just the rewrite intent"""
                intent = get_rewrite_context().rewrite_intent
                log.info(f"Rewrite handoff filter called. Intent: {intent}")

                # Construct a new HandoffInputData with cleared history
                # We keep new_items (which contains the handoff tool call) so the agent sees the immediate trigger
                # But we clear input_history to remove the conversation context

                new_history = ()
                try:
                    # Attempt to find a user message in history to clone/modify
                    # This is a best-effort attempt to make the agent see the intent as a user message
                    for item in data.input_history:
                        # Check if item looks like a user message (has role='user')
                        if hasattr(item, "role") and getattr(item, "role") == "user":
                            # Try to create a copy with new content if it's a Pydantic model
                            if hasattr(item, "model_copy"):
                                # Pydantic V2
                                new_item = item.model_copy(update={"content": intent})
                                new_history = (new_item,)
                                log.info(
                                    "Successfully constructed new user message item for handoff (Pydantic V2)"
                                )
                                break
                            elif hasattr(item, "copy"):
                                # Pydantic V1
                                new_item = item.copy(update={"content": intent})
                                new_history = (new_item,)
                                log.info(
                                    "Successfully constructed new user message item for handoff (Pydantic V1)"
                                )
                                break
                except Exception as e:
                    log.warning(f"Failed to construct user message item: {e}")

                # If we couldn't construct a user message, we return empty history.
                # The agent will still see the handoff tool call in new_items, which contains the intent.

                return HandoffInputData(
                    input_history=new_history,
                    pre_handoff_items=(),  # Clear pre-handoff items
                    new_items=tuple(data.new_items),  # Keep the handoff tool call
                )

            handoff_rewrite = handoff(
                agent=workflow_rewrite_agent_instance,
                input_type=HandoffRewriteData,
                input_filter=rewrite_handoff_input_filter,
                on_handoff=on_handoff,
            )

            # Construct instructions based on DISABLE_WORKFLOW_GEN
            if DISABLE_WORKFLOW_GEN:
                workflow_creation_instruction = """
**CASE 3: SEARCH WORKFLOW**
IF the user wants to find or generate a NEW workflow.
- Keywords: "create", "generate", "search", "find", "recommend", "生成", "查找", "推荐".
- Action: Use `recall_workflow`.
"""
                workflow_constraint = """
- [Critical!] When the user's intent is to get workflows or generate images with specific requirements, you MUST call `recall_workflow` tool to find existing similar workflows.
"""
            else:
                workflow_creation_instruction = """
**CASE 3: CREATE NEW / SEARCH WORKFLOW**
IF the user wants to find or generate a NEW workflow from scratch.
- Keywords: "create", "generate", "search", "find", "recommend", "生成", "查找", "推荐".
- Action: Use `recall_workflow` AND `gen_workflow`.
"""
                workflow_constraint = """
- [Critical!] When the user's intent is to get workflows or generate images with specific requirements, you MUST ALWAYS call BOTH recall_workflow tool AND gen_workflow tool to provide comprehensive workflow options. Never call just one of these tools - both are required for complete workflow assistance. First call recall_workflow to find existing similar workflows, then call gen_workflow to generate new workflow options.
"""

            agent = create_agent(
                name="ComfyUI-Copilot",
                instructions=f"""You are a powerful AI assistant for designing image processing workflows, capable of automating problem-solving using tools and commands.

When handing off to workflow rewrite agent or other agents, this session ID should be used for workflow data management.

### PRIMARY DIRECTIVE: INTENT CLASSIFICATION & HANDOFF
You act as a router. Your FIRST step is to classify the user's intent.

### TOOL-CALL RELIABILITY OVERRIDE (CONTEXT-TRIM SAFE)
The conversation history may be truncated for brevity and may contain ZERO tool calls/tool results.
- You MUST NOT treat "no prior tool message" as a reason to skip tool usage.
- If a CASE below requires a tool call or handoff, you MUST execute it even if you think you already know the answer.
- If a CASE below requires a tool call or handoff, your IMMEDIATE next assistant turn MUST be that tool call/handoff (do not output any natural-language explanation first).

**CASE 1: MODIFY/UPDATE/FIX CURRENT WORKFLOW (HIGHEST PRIORITY)**
IF the user wants to:
- Modify, enhance, update, or fix the CURRENT workflow/canvas.
- Add nodes/features to the CURRENT workflow (e.g., "add LoRA", "add controlnet", "fix the error").
- Change parameters in the CURRENT workflow.
- Keywords: "modify", "update", "add", "change", "fix", "current", "canvas", "修改", "更新", "添加", "画布", "加一个", "换一个", "调一下".

**ACTION:**
- You MUST IMMEDIATELY handoff to the `Workflow Rewrite Agent`.
- DO NOT call any other tools (like search_node, gen_workflow).
- DO NOT ask for more details. Just handoff.

**CASE 2: ANALYZE CURRENT WORKFLOW**
IF the user wants to:
- Analyze, explain, or understand the current workflow structure/logic.
- Ask questions about the current workflow (e.g., "how does this work?", "explain the workflow").
- Keywords: "analyze", "explain", "understand", "how it works", "workflow structure", "分析", "解释", "怎么工作的", "解读".

**ACTION:
- You MUST call `get_current_workflow` to retrieve the workflow details.
- Then, based on the returned workflow data, provide a detailed analysis or explanation to the user.

{workflow_creation_instruction}

### CONSTRAINT CHECKLIST
You must adhere to the following constraints to complete the task:

- **Tool compliance is mandatory**: If the selected CASE requires a tool/handoff, you MUST perform it. Do not answer directly without performing the required tool/handoff.
- [Important!] Respond must in the language used by the user in their question. Regardless of the language returned by the tools being called, please return the results based on the language used in the user's query. For example, if user ask by English, you must return
- Ensure that the commands or tools you invoke are within the provided tool list.
- If the execution of a command or tool fails, try changing the parameters or their format before attempting again.
- Your generated responses must follow the factual information given above. Do not make up information.
- If the result obtained is incorrect, try rephrasing your approach.
- Do not query for already obtained information repeatedly. If you successfully invoked a tool and obtained relevant information, carefully confirm whether you need to invoke it again.
- Ensure that the actions you generate can be executed accurately. Actions may include specific methods and target outputs.
- When you encounter a concept, try to obtain its precise definition and analyze what inputs can yield specific values for it.
- When generating a natural language query, include all known information in the query.
- Before performing any analysis or calculation, ensure that all sub-concepts involved have been defined.
- Printing the entire content of a file is strictly prohibited, as such actions have high costs and can lead to unforeseen consequences.
- Ensure that when you call a tool, you have obtained all the input variables for that tool, and do not fabricate any input values for it.
- Respond with markdown, using a minimum of 3 heading levels (H3, H4, H5...), and when including images use the format ![alt text](url),
{workflow_constraint}
- When the user's intent is to query, return the query result directly without attempting to assist the user in performing operations.
- When the user's intent is to get prompts for image generation (like Stable Diffusion). Use specific descriptive language with proper weight modifiers (e.g., (word:1.2)), prefer English terms, and separate elements with commas. Include quality terms (high quality, detailed), style specifications (realistic, anime), lighting (cinematic, golden hour), and composition (wide shot, close up) as needed. When appropriate, include negative prompts to exclude unwanted elements. Return words divided by commas directly without any additional text.
- If you cannot find the information needed to answer a query, consider using bing_search to obtain relevant information. For example, if search_node tool cannot find the node, you can use bing_search to obtain relevant information about those nodes or components.
- If search_node tool cannot find the node, you MUST use bing_search to obtain relevant information about those nodes or components.

- **ERROR MESSAGE ANALYSIS** - When a user pastes specific error text/logs (containing terms like "Failed", "Error", "Traceback", or stack traces), prioritize providing troubleshooting help rather than invoking search tools. Follow these steps:
  1. Analyze the error to identify the root cause (error type, affected component, missing dependencies, etc.)
  2. Explain the issue in simple terms
  3. Provide concrete, executable solutions including:
     - Specific shell commands to fix the issue (e.g., `git pull`, `pip install`, file path corrections)
     - Code snippets if applicable
     - Configuration file changes with exact paths and values
  4. If the error relates to a specific ComfyUI extension or node, include instructions for:
     - Updating the extension (`cd path/to/extension && git pull`)
     - Reinstalling dependencies
     - Alternative approaches if the extension is problematic
                """,
                mcp_servers=server_list,
                handoffs=[handoff_rewrite],
                tools=[get_current_workflow],
                config=config,
            )

            # Use messages directly as agent input since they're already in OpenAI format
            # The caller has already handled image formatting within messages
            agent_input = messages
            log.info(f"-- Processing {len(messages)} messages")

            from agents import (
                Runner,
                set_tracing_disabled,
                set_default_openai_api,
            )

            # from langsmith.wrappers import OpenAIAgentsTracingProcessor
            set_tracing_disabled(False)
            set_default_openai_api("chat_completions")
            # set_trace_processors([OpenAIAgentsTracingProcessor()])

            result = Runner.run_streamed(
                agent,
                input=agent_input,
                max_turns=30,
            )
            log.info("=== MCP Agent Run starting ===")

            # Variables to track response state similar to reference facade.py
            current_text = ""
            ext = None
            tool_results = {}  # Store results from different tools
            workflow_tools_called = set()  # Track called workflow tools
            last_yield_length = 0
            tool_call_queue = []  # Queue to track tool calls in order
                        # Collect workflow update ext data from tools and message outputs
            workflow_update_ext = None
            # Track if we've seen any handoffs to avoid showing initial handoff
            handoff_occurred = False

            # Enhanced retry mechanism for OpenAI streaming errors
            max_retries = 3
            retry_count = 0

            async def process_stream_events(stream_result):
                """Process stream events with enhanced error handling"""
                nonlocal \
                    current_text, \
                    last_yield_length, \
                    tool_call_queue, \
                    workflow_update_ext, \
                    tool_results, \
                    workflow_tools_called, \
                    handoff_occurred

                try:
                    async for event in stream_result.stream_events():
                        # Handle different event types similar to reference implementation
                        if event.type == "raw_response_event" and isinstance(
                            event.data, ResponseTextDeltaEvent
                        ):
                            # Stream text deltas for real-time response
                            delta_text = event.data.delta
                            if delta_text:
                                current_text += delta_text
                                # Yield tuple (accumulated_text, None) for streaming - similar to facade.py
                                # Only yield if we have new content to avoid duplicate yields
                                if len(current_text) > last_yield_length:
                                    last_yield_length = len(current_text)
                                    yield (current_text, None)
                            continue

                        elif event.type == "agent_updated_stream_event":
                            new_agent_name = event.new_agent.name
                            log.info(f"Handoff to: {new_agent_name}")

                            # Only show handoff message if we've already seen handoffs
                            # This prevents showing the initial handoff to ComfyUI-Copilot
                            if handoff_occurred:
                                # Add handoff information to the stream
                                handoff_text = (
                                    f"\n▸ **Switching to {new_agent_name}**\n\n"
                                )
                                current_text += handoff_text
                                last_yield_length = len(current_text)

                                # Yield text update only
                                yield (current_text, None)

                            # Mark that we've seen a handoff
                            handoff_occurred = True
                            continue

                        elif event.type == "run_item_stream_event":
                            if event.item.type == "tool_call_item":
                                # Get tool name correctly using raw_item.name
                                tool_name = getattr(
                                    event.item.raw_item, "name", "unknown_tool"
                                )
                                # Add to queue instead of overwriting current_tool_call
                                tool_call_queue.append(tool_name)
                                log.info(f"-- Tool '{tool_name}' was called")

                                # Track workflow tools being called
                                if tool_name in ["recall_workflow", "gen_workflow"]:
                                    workflow_tools_called.add(tool_name)
                            elif event.item.type == "tool_call_output_item":
                                log.info(f"-- Tool output: {event.item.output}")
                                # Store tool output for potential ext data processing
                                tool_output_data_str = str(event.item.output)

                                # Get the next tool from the queue (FIFO)
                                if tool_call_queue:
                                    tool_name = tool_call_queue.pop(0)
                                    log.info(
                                        f"-- Associating output with tool '{tool_name}'"
                                    )
                                else:
                                    tool_name = "unknown_tool"
                                    log.info(
                                        "-- Warning: No tool call in queue for output"
                                    )

                                try:
                                    import json

                                    # event.item.output might already be a dict or a string
                                    if isinstance(event.item.output, dict):
                                        tool_output_data = event.item.output
                                    elif isinstance(event.item.output, str):
                                        tool_output_data = json.loads(event.item.output)
                                    else:
                                        tool_output_data = json.loads(
                                            str(event.item.output)
                                        )

                                    if "result" in tool_output_data and isinstance(
                                        tool_output_data["result"], str
                                    ):
                                        try:
                                            # Unwrap FastMCP "result" wrapper
                                            tool_output_data = json.loads(
                                                tool_output_data["result"]
                                            )
                                        except json.JSONDecodeError:
                                            tool_output_data = {
                                                "text": tool_output_data["result"]
                                            }

                                    if (
                                        "ext" in tool_output_data
                                        and tool_output_data["ext"]
                                    ):
                                        # Store all ext items from tool output, not just workflow_update
                                        tool_ext_items = tool_output_data["ext"]
                                        for ext_item in tool_ext_items:
                                            if ext_item.get("type") in {
                                                "workflow_update",
                                                "param_update",
                                                "workflow",
                                            }:
                                                workflow_update_ext = tool_ext_items  # Store all ext items, not just one
                                                log.info(
                                                    f"-- Captured workflow tool ext from tool output: {len(tool_ext_items)} items"
                                                )
                                                break

                                    if (
                                        "text" in tool_output_data
                                        and tool_output_data.get("text")
                                    ):
                                        parsed_output = json.loads(
                                            tool_output_data["text"]
                                        )

                                        # Handle case where parsed_output might be a list instead of dict
                                        if isinstance(parsed_output, dict):
                                            answer = parsed_output.get("answer")
                                            data = parsed_output.get("data")
                                            tool_ext = parsed_output.get("ext")
                                        else:
                                            # If it's a list or other type, handle gracefully
                                            answer = None
                                            data = (
                                                parsed_output
                                                if isinstance(parsed_output, list)
                                                else None
                                            )
                                            tool_ext = None

                                        # Store tool results similar to reference facade.py
                                        tool_results[tool_name] = {
                                            "answer": answer,
                                            "data": data,
                                            "ext": tool_ext,
                                            "content_dict": parsed_output,
                                        }
                                        log.info(
                                            f"-- Stored result for tool '{tool_name}': data={len(data) if data else 0}, ext={tool_ext}"
                                        )

                                        # Track workflow tools that produced results
                                        if tool_name in [
                                            "recall_workflow",
                                            "gen_workflow",
                                        ]:
                                            log.info(
                                                f"-- Workflow tool '{tool_name}' produced result with data: {len(data) if data else 0}"
                                            )

                                except (json.JSONDecodeError, TypeError) as e:
                                    # If not JSON or parsing fails, treat as regular text
                                    log.error(
                                        f"-- Failed to parse tool output as JSON: {e}"
                                    )
                                    log.error(f"-- Traceback: {traceback.format_exc()}")
                                    tool_results[tool_name] = {
                                        "answer": tool_output_data_str,
                                        "data": None,
                                        "ext": None,
                                        "content_dict": None,
                                    }

                            elif event.item.type == "message_output_item":
                                pass
                            else:
                                pass  # Ignore other event types

                except Exception as e:
                    log.error(f"Unexpected streaming error: {e}")
                    log.error(f"Traceback: {traceback.format_exc()}")
                    raise e

            # Implement retry mechanism with exponential backoff
            while retry_count <= max_retries:
                try:
                    async for stream_data in process_stream_events(result):
                        if stream_data:
                            yield stream_data
                    # If we get here, streaming completed successfully
                    break

                except (
                    AttributeError,
                    TypeError,
                    ConnectionError,
                    OSError,
                    APIError,
                ) as stream_error:
                    retry_count += 1
                    error_msg = str(stream_error)

                    # Check for specific streaming errors that are worth retrying
                    should_retry = (
                        "'NoneType' object has no attribute 'strip'" in error_msg
                        or "Connection broken" in error_msg
                        or "InvalidChunkLength" in error_msg
                        or "socket hang up" in error_msg
                        or "Connection reset" in error_msg
                    )

                    if should_retry and retry_count <= max_retries:
                        wait_time = min(
                            2 ** (retry_count - 1), 10
                        )  # Exponential backoff, max 10 seconds
                        log.error(
                            f"Stream error (attempt {retry_count}/{max_retries}): {error_msg}"
                        )
                        log.info(f"Retrying in {wait_time} seconds...")

                        # Yield current progress before retry
                        if current_text:
                            yield (current_text, None)

                        await asyncio.sleep(wait_time)

                        try:
                            # Create a new result object for retry
                            result = Runner.run_streamed(
                                agent,
                                input=agent_input,
                            )
                            log.info(f"=== Retry attempt {retry_count} starting ===")
                        except Exception as retry_setup_error:
                            log.error(f"Failed to setup retry: {retry_setup_error}")
                            if retry_count >= max_retries:
                                raise stream_error  # Re-raise original error if max retries reached
                            continue
                    else:
                        log.error(
                            f"Non-retryable streaming error or max retries reached: {error_msg}"
                        )
                        log.error(f"Traceback: {traceback.format_exc()}")
                        if isinstance(stream_error, RateLimitError):
                            default_error_msg = "Rate limit exceeded, please try again later."
                            error_body = getattr(stream_error, "body", {})
                            error_msg = error_body.get("message") if isinstance(error_body, dict) else None
                            final_error_msg = error_msg or default_error_msg
                            yield (final_error_msg, None)
                            return
                        else:
                            yield (
                                f"I apologize, but an error occurred while processing your request: {error_msg}",
                                None,
                            )
                            return

                except Exception as unexpected_error:
                    retry_count += 1
                    log.error(
                        f"Unexpected error during streaming (attempt {retry_count}/{max_retries}): {unexpected_error}"
                    )
                    log.error(f"Traceback: {traceback.format_exc()}")

                    if retry_count > max_retries:
                        log.error("Max retries exceeded for unexpected error")
                        break
                    else:
                        # Brief wait before retry for unexpected errors
                        await asyncio.sleep(1)
                        continue

            (
                current_text,
                workflow_update_ext,
                bridged_pseudo_tools,
            ) = await _bridge_pseudo_tool_calls(
                current_text=current_text,
                server_list=server_list,
                tool_results=tool_results,
                workflow_tools_called=workflow_tools_called,
                workflow_update_ext=workflow_update_ext,
            )
            if bridged_pseudo_tools and not current_text:
                # Pseudo-call syntax was stripped out but no bridged answer text
                # remained. Load-bearing edge case: MCP server tools like
                # recall_workflow / gen_workflow return structured workflow data
                # via `result["data"]` and carry NO "answer" field at all, so
                # bridged_answer_texts never gets populated for them. When the
                # LLM emits only bare pseudo-calls with no surrounding prose,
                # the stripped text is empty AND the join above has nothing to
                # substitute. Without this fallback, the raw `tool_name[ARGS]{…}`
                # syntax would leak to the user's chat bubble.
                #
                # NOTE (2026-04-13): This is NOT a band-aid for the old
                # `on_invoke_tool(None, …)` ToolContext bug — that bug caused
                # local tools to return a *truthy* error string, which would
                # have made it past this branch. Don't delete this on the
                # assumption the bug-fix made it redundant.
                attempted_tools = sorted(tool_results.keys())
                failed_tools = [
                    name
                    for name in attempted_tools
                    if name != "_message_output_ext"
                    and not (
                        tool_results[name].get("data")
                        or tool_results[name].get("answer")
                    )
                ]
                if failed_tools:
                    tool_list = ", ".join(failed_tools)
                    current_text = (
                        f"(도구 호출 {tool_list} 이(가) 결과를 반환하지 못했습니다. "
                        f"다른 표현으로 다시 물어보시거나, 잠시 후 재시도해 주세요.)"
                    )
                else:
                    current_text = (
                        "(도구 호출은 처리했지만 사용자에게 보여줄 텍스트가 없습니다.)"
                    )
                log.info(
                    f"[MCP] Pseudo tool calls bridged; substituted fallback text "
                    f"(failed_tools={failed_tools})"
                )

            # Add detailed debugging info about tool results
            log.info(f"Total tool results: {len(tool_results)}")
            for tool_name, result in tool_results.items():
                result_type = (
                    "Message Output" if tool_name == "_message_output_ext" else "Tool"
                )
                log.info(f"{result_type}: {tool_name}")
                log.info(f"  - Has data: {result['data'] is not None}")
                log.info(
                    f"  - Data length: {len(result['data']) if result['data'] else 0}"
                )
                log.info(f"  - Has ext: {result['ext'] is not None}")
                if result["ext"]:
                    log.info(
                        f"  - Ext types: {[item.get('type') for item in (result['ext'] if isinstance(result['ext'], list) else [result['ext']])]}"
                    )
                log.info(
                    f"  - Answer preview: {result['answer'][:100] if result['answer'] else 'None'}..."
                )
            log.info("=== End Tool Results Summary ===\n")

            # Process workflow tools results integration similar to reference facade.py
            workflow_tools_found = [
                tool
                for tool in ["recall_workflow", "gen_workflow"]
                if tool in tool_results
            ]
            finished = False  # Default finished state

            if workflow_tools_found:
                log.info(f"Workflow tools called: {workflow_tools_found}")

                # Check if both workflow tools were called
                if "recall_workflow" in tool_results and "gen_workflow" in tool_results:
                    log.info(
                        "Both recall_workflow and gen_workflow were called, merging results"
                    )

                    # Check each tool's success and merge results
                    successful_workflows = []

                    recall_result = tool_results["recall_workflow"]
                    if recall_result["data"] and len(recall_result["data"]) > 0:
                        log.info(
                            f"recall_workflow succeeded with {len(recall_result['data'])} workflows"
                        )
                        log.info(
                            f"  - Workflow IDs: {[w.get('id') for w in recall_result['data']]}"
                        )
                        successful_workflows.extend(recall_result["data"])
                    else:
                        log.error("recall_workflow failed or returned no data")

                    gen_result = tool_results["gen_workflow"]
                    if gen_result["data"] and len(gen_result["data"]) > 0:
                        log.info(
                            f"gen_workflow succeeded with {len(gen_result['data'])} workflows"
                        )
                        log.info(
                            f"  - Workflow IDs: {[w.get('id') for w in gen_result['data']]}"
                        )
                        successful_workflows.insert(0, *gen_result["data"])
                    else:
                        log.error("gen_workflow failed or returned no data")

                    # Remove duplicates based on workflow ID
                    seen_ids = set()
                    unique_workflows = []
                    for workflow in successful_workflows:
                        workflow_id = workflow.get("id")
                        if workflow_id and workflow_id not in seen_ids:
                            seen_ids.add(workflow_id)
                            unique_workflows.append(workflow)
                            log.info(
                                f"  - Added unique workflow: {workflow_id} - {workflow.get('name', 'Unknown')}"
                            )
                        elif workflow_id:
                            log.info(
                                f"  - Skipped duplicate workflow: {workflow_id} - {workflow.get('name', 'Unknown')}"
                            )
                        else:
                            # If no ID, add anyway (shouldn't happen but just in case)
                            unique_workflows.append(workflow)
                            log.info(
                                f"  - Added workflow without ID: {workflow.get('name', 'Unknown')}"
                            )

                    log.info(
                        f"Total workflows before deduplication: {len(successful_workflows)}"
                    )
                    log.info(
                        f"Total workflows after deduplication: {len(unique_workflows)}"
                    )

                    # Create final ext structure
                    if unique_workflows:
                        ext = [{"type": "workflow", "data": unique_workflows}]
                        log.info(
                            f"Returning {len(unique_workflows)} workflows from successful tools"
                        )
                    else:
                        ext = None
                        log.error("No successful workflow data to return")

                    # Both tools called, finished = True
                    finished = True

                elif (
                    "recall_workflow" in tool_results
                    and "gen_workflow" not in tool_results
                ):
                    if DISABLE_WORKFLOW_GEN:
                        # If generation is disabled, we don't wait for gen_workflow
                        log.info(
                            "Only recall_workflow was called and generation is disabled, returning its result"
                        )
                        recall_result = tool_results["recall_workflow"]
                        if recall_result["data"] and len(recall_result["data"]) > 0:
                            ext = [{"type": "workflow", "data": recall_result["data"]}]
                            log.info(
                                f"Returning {len(recall_result['data'])} workflows from recall_workflow"
                            )
                        else:
                            ext = None
                            log.error("recall_workflow failed or returned no data")
                        finished = True
                    else:
                        # Only recall_workflow was called, don't return ext, keep finished=false
                        log.info(
                            "Only recall_workflow was called, waiting for gen_workflow, not returning ext"
                        )
                        ext = None
                        finished = False  # This is the key: keep finished=false to wait for gen_workflow

                elif (
                    "gen_workflow" in tool_results
                    and "recall_workflow" not in tool_results
                ):
                    # Only gen_workflow was called, return its result normally
                    log.info("Only gen_workflow was called, returning its result")
                    gen_result = tool_results["gen_workflow"]
                    if gen_result["data"] and len(gen_result["data"]) > 0:
                        ext = [{"type": "workflow", "data": gen_result["data"]}]
                        log.info(
                            f"Returning {len(gen_result['data'])} workflows from gen_workflow"
                        )
                    else:
                        ext = None
                        log.error("gen_workflow failed or returned no data")

                    # Only gen_workflow called, finished = True
                    finished = True
            else:
                # No workflow tools called, check if other tools or message output returned ext
                for tool_name, result in tool_results.items():
                    if result["ext"]:
                        ext = result["ext"]
                        log.info(f"Using ext from {tool_name}")
                        break

                # When no workflow tools are called (e.g., handoff to workflow_rewrite_agent)
                # The agent stream has completed at this point, so finished should be True
                # The workflow_update_ext will be included in final_ext regardless
                finished = True

            # Prepare final ext (debug_ext would be empty here since no debug events)
            final_ext = ext
            if workflow_update_ext:
                # workflow_update_ext is now a list of ext items, so extend rather than wrap
                if isinstance(workflow_update_ext, list):
                    final_ext = workflow_update_ext + (ext if ext else [])
                else:
                    # Backward compatibility: if it's a single item, wrap it
                    final_ext = [workflow_update_ext] + (ext if ext else [])
                log.info(
                    f"-- Including workflow_update ext in final response: {len(workflow_update_ext) if isinstance(workflow_update_ext, list) else 1} items"
                )

            # Final yield with complete text, ext data, and finished status
            # Return as tuple (text, ext_with_finished) where ext_with_finished includes finished info
            if final_ext:
                # Add finished status to ext structure
                ext_with_finished = {"data": final_ext, "finished": finished}
            else:
                ext_with_finished = {"data": None, "finished": finished}

            yield (current_text, ext_with_finished)

        finally:
            await exit_stack.aclose()

    except Exception as e:
        log.error(f"Error in comfyui_agent_invoke: {str(e)}")
        log.error(f"Traceback: {traceback.format_exc()}")
        error_message = f"I apologize, but an error occurred while processing your request: {str(e)}"

        # Check if this is a retryable streaming error that should not finish the conversation
        error_msg = str(e)
        is_retryable_streaming_error = (
            "'NoneType' object has no attribute 'strip'" in error_msg
            or "Connection broken" in error_msg
            or "InvalidChunkLength" in error_msg
            or "socket hang up" in error_msg
            or "Connection reset" in error_msg
            or isinstance(e, APIError)
        )

        if is_retryable_streaming_error:
            # For retryable streaming errors, don't finish - allow user to retry
            log.info(
                "Detected retryable streaming error, setting finished=False to allow retry"
            )
            error_ext = {"data": None, "finished": False}
            error_message = f"A temporary streaming error occurred: {str(e)}. Please try your request again."
        else:
            # For other errors, finish the conversation
            error_ext = {"data": None, "finished": True}

        yield (error_message, error_ext)
