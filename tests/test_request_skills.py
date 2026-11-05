"""Regression tests for selecting skills on plugin requests."""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = Path(__file__).parents[1] / "utils" / "subagent_bridge.py"
SPEC = importlib.util.spec_from_file_location("subagent_bridge_skills", MODULE_PATH)
bridge = importlib.util.module_from_spec(SPEC)
sys.modules["subagent_bridge_skills"] = bridge
SPEC.loader.exec_module(bridge)


def skills(*names, source="installed"):
    return [SimpleNamespace(name=name, source=source) for name in names]


def names(items):
    return [item.name for item in items]


class FakeSkillManager:
    def __init__(self):
        self.runtime = None
        self.workspace_path = None

    def list_skills(self, *, active_only, runtime):
        assert active_only is True
        self.runtime = runtime
        return skills("pdf", "general-web-search")

    def list_workspace_skills(self, workspace_path):
        self.workspace_path = workspace_path
        return skills("local-skill", source="workspace")


def collect(manager, runtime, persona, config_filter=lambda items, _cfg: items):
    async def get_workspace_path():
        return "/workspace"

    return asyncio.run(
        bridge.collect_request_skills(
            manager,
            {"computer_use_runtime": runtime},
            persona,
            config_filter,
            get_workspace_path,
        )
    )


def test_no_persona_keeps_all_skills():
    assert names(bridge.select_persona_skills(skills("a", "b"), [], None)) == ["a", "b"]


def test_all_skills_persona_keeps_all_skills():
    persona = {"skills": None}
    assert names(bridge.select_persona_skills(skills("a", "b"), [], persona)) == [
        "a",
        "b",
    ]


def test_partial_selection_keeps_selected_skills():
    persona = {"skills": ["b"]}
    assert names(bridge.select_persona_skills(skills("a", "b"), [], persona)) == ["b"]


def test_empty_selection_removes_all_skills_including_workspace():
    persona = {"skills": []}
    assert (
        bridge.select_persona_skills(skills("a"), skills("local-skill"), persona) == []
    )


def test_workspace_skills_are_not_limited_by_selection():
    persona = {"skills": ["b"]}
    selected = bridge.select_persona_skills(
        skills("a", "b"), skills("local-skill"), persona
    )
    assert names(selected) == ["b", "local-skill"]


def test_workspace_skill_replaces_same_named_skill():
    selected = bridge.select_persona_skills(
        skills("a"), skills("a", source="workspace"), None
    )
    assert [skill.source for skill in selected] == ["workspace"]


def test_local_runtime_includes_workspace_skills():
    manager = FakeSkillManager()
    selected = collect(manager, "local", {"skills": ["pdf"]})

    assert manager.runtime == "local"
    assert manager.workspace_path == "/workspace"
    assert names(selected) == ["local-skill", "pdf"]


def test_other_runtimes_skip_workspace_skills():
    manager = FakeSkillManager()
    selected = collect(manager, "none", None)

    assert manager.runtime == "none"
    assert manager.workspace_path is None
    assert names(selected) == ["pdf", "general-web-search"]


def test_config_filter_runs_before_persona_selection():
    manager = FakeSkillManager()
    selected = collect(
        manager,
        "none",
        {"skills": ["pdf", "general-web-search"]},
        config_filter=lambda items, _cfg: items[:1],
    )
    assert names(selected) == ["pdf"]
