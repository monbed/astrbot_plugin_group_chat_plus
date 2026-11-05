"""Regression tests for restoring subagent orchestration on plugin requests."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = Path(__file__).parents[1] / "utils" / "subagent_bridge.py"
SPEC = importlib.util.spec_from_file_location("subagent_bridge", MODULE_PATH)
bridge = importlib.util.module_from_spec(SPEC)
sys.modules["subagent_bridge"] = bridge
SPEC.loader.exec_module(bridge)

ROUTER_PROMPT = "You are a task router."
HANDOFF = "transfer_to_life_assistant"


class FakeToolSet:
    """Implements the ToolSet behaviour the bridge relies on."""

    def __init__(self, names):
        self.tools = [SimpleNamespace(name=name) for name in names]

    def add_tool(self, tool):
        self.remove_tool(tool.name)
        self.tools.append(tool)

    def remove_tool(self, name):
        self.tools = [tool for tool in self.tools if tool.name != name]

    def names(self):
        return [tool.name for tool in self.tools]


def make_context(
    *,
    main_enable=True,
    remove_duplicates=True,
    agent_enabled=True,
    persona_tools=("maps_weather", "get-tickets"),
    handoffs=(HANDOFF,),
    router_prompt=ROUTER_PROMPT,
):
    tools = list(persona_tools) if persona_tools is not None else None
    config = {
        "subagent_orchestrator": {
            "main_enable": main_enable,
            "remove_main_duplicate_tools": remove_duplicates,
            "router_system_prompt": router_prompt,
            "agents": [
                {
                    "name": "life_assistant",
                    "persona_id": "life",
                    "enabled": agent_enabled,
                }
            ],
        }
    }
    return SimpleNamespace(
        get_config=lambda: config,
        subagent_orchestrator=SimpleNamespace(
            handoffs=[SimpleNamespace(name=name) for name in handoffs]
        ),
        persona_manager=SimpleNamespace(
            get_persona_v3_by_id=lambda _id: {"tools": tools}
        ),
        get_llm_tool_manager=lambda: SimpleNamespace(
            func_list=[
                SimpleNamespace(name=name)
                for name in ("maps_weather", "get-tickets", "search_memes")
            ]
        ),
    )


def make_request():
    return SimpleNamespace(
        func_tool=FakeToolSet(
            ["maps_weather", "get-tickets", "search_memes", "future_task"]
        ),
        system_prompt="persona",
    )


def test_injects_handoff_and_removes_assigned_tools():
    req = make_request()
    result = bridge.apply_subagent_orchestration(req, make_context())

    assert sorted(req.func_tool.names()) == sorted(
        ["search_memes", "future_task", HANDOFF]
    )
    assert result.handoff_names == [HANDOFF]
    assert result.removed_tool_names == ["get-tickets", "maps_weather"]


def test_appends_router_prompt_once():
    req = make_request()
    context = make_context()

    assert bridge.apply_subagent_orchestration(req, context).router_prompt_added
    assert ROUTER_PROMPT in req.system_prompt
    prompt = req.system_prompt

    second = bridge.apply_subagent_orchestration(req, context)
    assert not second.router_prompt_added
    assert req.system_prompt == prompt
    assert req.func_tool.names().count(HANDOFF) == 1


def test_missing_router_prompt_leaves_system_prompt_unchanged():
    req = make_request()
    bridge.apply_subagent_orchestration(req, make_context(router_prompt=None))
    assert req.system_prompt == "persona"


def test_keeps_tools_when_deduplication_is_off():
    req = make_request()
    bridge.apply_subagent_orchestration(req, make_context(remove_duplicates=False))
    assert {"maps_weather", "get-tickets", HANDOFF} <= set(req.func_tool.names())


def test_does_nothing_when_orchestration_is_disabled():
    req = make_request()
    before = req.func_tool.names()
    result = bridge.apply_subagent_orchestration(req, make_context(main_enable=False))

    assert result.handoff_names == []
    assert req.func_tool.names() == before
    assert req.system_prompt == "persona"


def test_keeps_tools_when_no_handoff_is_available():
    req = make_request()
    before = req.func_tool.names()
    bridge.apply_subagent_orchestration(req, make_context(handoffs=()))
    assert req.func_tool.names() == before


def test_disabled_agent_does_not_remove_tools():
    req = make_request()
    bridge.apply_subagent_orchestration(req, make_context(agent_enabled=False))
    assert "maps_weather" in req.func_tool.names()


def test_all_tools_persona_matches_astrbot_and_removes_nothing():
    req = make_request()
    result = bridge.apply_subagent_orchestration(req, make_context(persona_tools=None))

    assert result.removed_tool_names == []
    assert HANDOFF in req.func_tool.names()
    assert "maps_weather" in req.func_tool.names()


def test_agent_with_explicit_null_tools_claims_registered_tools():
    assigned = bridge.collect_assigned_tool_names(
        {"agents": [{"name": "a", "tools": None}]},
        SimpleNamespace(get_persona_v3_by_id=lambda _id: None),
        [SimpleNamespace(name="maps_weather")],
    )
    assert assigned == {"maps_weather"}


def registered(*names):
    return [SimpleNamespace(name=name) for name in names]


def test_persona_filter_removes_unselected_tools_and_keeps_builtins():
    tool_set = FakeToolSet(["maps_weather", "search_memes", "future_task"])

    removed = bridge.remove_tools_outside_persona(
        tool_set,
        registered("maps_weather", "search_memes"),
        ["search_memes"],
    )

    assert removed == ["maps_weather"]
    assert sorted(tool_set.names()) == ["future_task", "search_memes"]


def test_persona_with_all_tools_removes_nothing():
    tool_set = FakeToolSet(["maps_weather", "future_task"])

    assert (
        bridge.remove_tools_outside_persona(tool_set, registered("maps_weather"), None)
        == []
    )
    assert sorted(tool_set.names()) == ["future_task", "maps_weather"]


def test_persona_with_no_tools_removes_registered_tools_only():
    tool_set = FakeToolSet(["maps_weather", "future_task"])

    bridge.remove_tools_outside_persona(tool_set, registered("maps_weather"), [])

    assert tool_set.names() == ["future_task"]


def test_handoff_survives_persona_filter_then_orchestration():
    req = make_request()
    context = make_context(remove_duplicates=False)

    bridge.remove_tools_outside_persona(
        req.func_tool,
        context.get_llm_tool_manager().func_list,
        ["search_memes"],
    )
    bridge.apply_subagent_orchestration(req, context)

    assert sorted(req.func_tool.names()) == sorted(
        ["search_memes", "future_task", HANDOFF]
    )
