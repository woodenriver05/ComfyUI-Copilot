import copy
import json
import os
import pytest
import sys
from pathlib import Path

comfy_root = os.path.abspath("../../../../..")
if comfy_root not in sys.path:
    sys.path.insert(0, comfy_root)
sys.path = [p for p in sys.path if not p.endswith("backend")]
import utils.install_util


from backend.service.mcp_client import comfyui_agent_invoke
from backend.utils.request_context import set_session_id, set_config

FIXTURE_DIR = Path(__file__).parent / "golden" / "mcp_client"

def _load_fixture(n: int) -> dict:
    return json.loads((FIXTURE_DIR / f"comfyui_agent_invoke_{n:02d}.json").read_text())

def _normalize(payload):
    d = copy.deepcopy(payload) if isinstance(payload, (dict, list, tuple)) else payload
    if isinstance(d, tuple):
        d = list(d)
    if isinstance(d, dict):
        for k in ("request_id", "created_at", "session_id", "id", "created", "timestamp", "trace_id", "run_id", "message_id"):
            if k in d:
                d[k] = "<dynamic>"
        for k, v in d.items():
            if isinstance(v, (dict, list, tuple)):
                d[k] = _normalize(v)
            elif isinstance(v, str) and len(v) > 20:
                d[k] = "<text>"
    elif isinstance(d, list):
        for i, item in enumerate(d):
            if isinstance(item, str) and len(item) > 20:
                d[i] = "<text>"
            else:
                d[i] = _normalize(item)
    return d

@pytest.mark.parametrize("n", range(1, 6))
@pytest.mark.parametrize("legacy", ["1", "0"])
@pytest.mark.asyncio
async def test_comfyui_agent_invoke_golden(n: int, legacy: str, monkeypatch):
    monkeypatch.setenv("LEGACY_AGENT_INVOKE", legacy)
    fx = _load_fixture(n)

    set_session_id("test-session-123")
    set_config({"mock": "config"})

    import backend.service.mcp_client as mcp_client
    monkeypatch.setattr(mcp_client, "_LEGACY_AGENT_INVOKE", legacy == "1")

    result = []
    generator = comfyui_agent_invoke(**fx["input"]["kwargs"])
    async for chunk in generator:
        result.append(chunk)

    assert _normalize(result) == _normalize(fx["expected_output"]), \
        f"Fixture {n} with LEGACY_AGENT_INVOKE={legacy} mismatch"
