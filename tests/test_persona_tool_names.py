"""Regression tests for resolving the main persona's tool selection."""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

MODULE_PATH = Path(__file__).parents[1] / "utils" / "tools_reminder.py"


def load_tools_reminder():
    """Load the module with the minimal AstrBot doubles it imports."""

    def module(name, **attributes):
        stub = types.ModuleType(name)
        stub.__dict__.update(attributes)
        return stub

    stubs = {
        "astrbot": module("astrbot"),
        "astrbot.api": module("astrbot.api", logger=mock.Mock()),
        "astrbot.api.all": module("astrbot.api.all"),
        "astrbot.api.event": module("astrbot.api.event", AstrMessageEvent=object),
        "astrbot.core": module("astrbot.core"),
        "astrbot.core.star": module("astrbot.core.star"),
        "astrbot.core.star.context": module("astrbot.core.star.context", Context=object),
        "astrbot.core.star.star": module("astrbot.core.star.star", star_map={}),
    }
    with mock.patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location("tools_reminder", MODULE_PATH)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
    return loaded


def make_context(*, current_conversation_id, conversation, persona):
    calls = []

    async def resolve_selected_persona(**kwargs):
        calls.append(kwargs)
        return ("persona", persona, None, False)

    async def get_curr_conversation_id(_umo):
        return current_conversation_id

    async def get_conversation(_umo, _cid):
        return conversation

    context = SimpleNamespace(
        persona_manager=SimpleNamespace(
            resolve_selected_persona=resolve_selected_persona
        ),
        conversation_manager=SimpleNamespace(
            get_curr_conversation_id=get_curr_conversation_id,
            get_conversation=get_conversation,
        ),
    )
    return context, calls


def get_names(context):
    reminder = load_tools_reminder().ToolsReminder
    return asyncio.run(reminder.get_persona_tool_names(context, "umo", "aiocqhttp"))


def test_uses_conversation_persona():
    context, calls = make_context(
        current_conversation_id="cid",
        conversation=SimpleNamespace(persona_id="chosen"),
        persona={"tools": ["search_memes"]},
    )

    assert get_names(context) == ["search_memes"]
    assert calls[0]["conversation_persona_id"] == "chosen"


def test_without_conversation_resolves_default_persona():
    context, calls = make_context(
        current_conversation_id=None,
        conversation=None,
        persona={"tools": ["search_memes"]},
    )

    assert get_names(context) == ["search_memes"]
    assert calls[0]["conversation_persona_id"] is None


def test_all_tools_persona_returns_none():
    context, _calls = make_context(
        current_conversation_id="cid",
        conversation=SimpleNamespace(persona_id=None),
        persona={"tools": None},
    )

    assert get_names(context) is None


def test_empty_selection_stays_empty():
    context, _calls = make_context(
        current_conversation_id="cid",
        conversation=SimpleNamespace(persona_id=None),
        persona={"tools": []},
    )

    assert get_names(context) == []
