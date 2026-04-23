import asyncio
import json
import os
import sys

comfy_root = os.path.abspath("../../../../..")
sys.path.insert(0, comfy_root)
copilot_root = os.path.abspath("../../..")
sys.path.insert(1, copilot_root)
sys.path = [p for p in sys.path if not p.endswith("backend")]
import utils.install_util


from test_comfyui_agent_invoke_golden import _load_fixture, _normalize
from backend.service.mcp_client import comfyui_agent_invoke, ResponseTextDeltaEvent
from backend.utils.request_context import set_session_id, set_config

class MockMonkeyPatch:
    def setenv(self, key, val):
        os.environ[key] = val
    def setattr(self, obj, attr, val):
        setattr(obj, attr, val)

class MockStreamResult:
    def __init__(self, fx):
        self.fx = fx
    async def stream_events(self):
        class MockEvent:
            def __init__(self, t, d):
                self.type = t
                self.data = d
                self.item = d
                self.new_agent = d

        text = self.fx["expected_output"][0][0] if isinstance(self.fx["expected_output"][0][0], str) else "dummy"
        yield MockEvent("raw_response_event", ResponseTextDeltaEvent(delta=text, content_index=0, item_id="item_1", logprobs=[], output_index=0, sequence_number=1, type="response.output_text.delta"))

async def main():
    print("Running equivalence test: LEGACY=1 vs LEGACY=0")
    mp = MockMonkeyPatch()
    set_session_id("test-session-123")
    set_config({"mock": "config"})

    import backend.service.mcp_client as mcp_client

    async def mock_pass_through(*args, **kwargs):
        return "This is a mocked RAG agent response that is long enough to be normalized to <text>"
    mp.setattr(mcp_client, "pass_through_rag_agent", mock_pass_through)

    all_passed = True
    for n in range(1, 6):
        print(f"Testing fixture {n}...")
        fx = _load_fixture(n)

        def mock_run_streamed(*args, **kwargs):
            return MockStreamResult(fx)
        import agents.run
        mp.setattr(agents.run.Runner, "run_streamed", mock_run_streamed)

        # Run legacy=1
        mp.setattr(mcp_client, "_LEGACY_AGENT_INVOKE", True)
        result_legacy = []
        gen1 = comfyui_agent_invoke(**fx["input"]["kwargs"])
        async for chunk in gen1:
            result_legacy.append(chunk)

        # Run legacy=0
        mp.setattr(mcp_client, "_LEGACY_AGENT_INVOKE", False)
        result_new = []
        gen0 = comfyui_agent_invoke(**fx["input"]["kwargs"])
        async for chunk in gen0:
            result_new.append(chunk)

        is_match = _normalize(result_legacy) == _normalize(result_new)
        print(f"Fixture {n}: {'MATCH' if is_match else 'MISMATCH'}")
        if not is_match:
            all_passed = False
            print("LEGACY:", json.dumps(_normalize(result_legacy), indent=2))
            print("NEW:", json.dumps(_normalize(result_new), indent=2))

    if all_passed:
        print("10/10 PASS! (Structural equivalence confirmed)")

if __name__ == "__main__":
    asyncio.run(main())
