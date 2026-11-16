"""Regression tests for passing keyword-triggered messages to self-learning."""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

MODULE_PATH = Path(__file__).parents[1] / "utils" / "self_learning_bridge.py"
SPEC = importlib.util.spec_from_file_location("self_learning_bridge", MODULE_PATH)
bridge = importlib.util.module_from_spec(SPEC)
sys.modules["self_learning_bridge"] = bridge
SPEC.loader.exec_module(bridge)

# Stands in for self-learning's entry module, which exposes the sample filter.
ENTRY_MODULE = "fake_self_learning_main"


class FakePipeline:
    """Records affection updates like self-learning's MessagePipeline."""

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    async def process_affection(self, group_id, sender_id, message_text):
        self.calls.append((group_id, sender_id, message_text))
        if self.error:
            raise self.error


def make_plugin(
    *,
    affection_enabled=True,
    pipeline="default",
    db_ready=True,
    collects=True,
    is_command=False,
    shutting_down=False,
    ignored_texts=(),
):
    entry = ModuleType(ENTRY_MODULE)
    entry.should_ignore_learning_sample = (
        lambda text, *, sender_id=None, **_metadata: text in ignored_texts
    )
    entry.extract_learning_event_metadata = lambda _event: {"message_type": "group"}
    sys.modules[ENTRY_MODULE] = entry

    plugin_cls = type("SelfLearningPlugin", (), {"__module__": ENTRY_MODULE})
    plugin = plugin_cls()
    plugin._shutting_down = shutting_down
    plugin.plugin_config = SimpleNamespace(enable_affection_system=affection_enabled)
    plugin._pipeline = FakePipeline() if pipeline == "default" else pipeline
    plugin.db_manager = SimpleNamespace(is_ready=db_ready)
    plugin.qq_filter = SimpleNamespace(
        should_collect_message=lambda _sender_id, _group_id: collects
    )
    plugin._command_filter = SimpleNamespace(
        is_astrbot_command=lambda _event: is_command
    )
    plugin.tracked = []
    plugin._track_task = plugin.tracked.append
    return plugin


def make_event(*, sender_id="10001", group_id="20002", self_id="30003"):
    return SimpleNamespace(
        get_sender_id=lambda: sender_id,
        get_group_id=lambda: group_id,
        get_self_id=lambda: self_id,
    )


def submit(plugin, event=None, text="小助手你好"):
    """Submit one message and wait for the background update to finish."""

    async def run():
        pending = set()
        outcome = bridge.submit_affection_interaction(
            plugin, event or make_event(), text, pending
        )
        started = len(pending)
        await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.sleep(0)
        return outcome, started, len(pending)

    return asyncio.run(run())


def plan(**overrides):
    arguments = {
        "is_at_message": False,
        "has_trigger_keyword": False,
        "framework_woken": False,
        "keyword_smart_mode": False,
    }
    arguments.update(overrides)
    return bridge.plan_affection_submission(**arguments)


def test_keyword_trigger_is_submitted_at_once():
    assert plan(has_trigger_keyword=True) == bridge.SUBMIT_NOW


def test_smart_mode_keyword_waits_for_the_decision():
    assert (
        plan(has_trigger_keyword=True, keyword_smart_mode=True)
        == bridge.SUBMIT_AFTER_DECISION
    )


def test_text_at_is_submitted_at_once_even_in_smart_mode():
    assert (
        plan(is_at_message=True, has_trigger_keyword=True, keyword_smart_mode=True)
        == bridge.SUBMIT_NOW
    )


def test_framework_wake_is_left_to_self_learning():
    assert plan(is_at_message=True, framework_woken=True) == bridge.SKIP
    assert plan(has_trigger_keyword=True, framework_woken=True) == bridge.SKIP


def test_untriggered_message_is_skipped():
    assert plan() == bridge.SKIP


def test_find_plugin_returns_the_enabled_instance():
    instance = object()
    names = []

    def get_registered_star(name):
        names.append(name)
        return SimpleNamespace(activated=True, star_cls=instance)

    context = SimpleNamespace(get_registered_star=get_registered_star)
    assert bridge.find_self_learning_plugin(context) is instance
    assert names == ["astrbot_plugin_self_learning"]


def test_find_plugin_ignores_missing_or_disabled_plugin():
    missing = SimpleNamespace(get_registered_star=lambda _name: None)
    disabled = SimpleNamespace(
        get_registered_star=lambda _name: SimpleNamespace(
            activated=False, star_cls=object()
        )
    )
    assert bridge.find_self_learning_plugin(missing) is None
    assert bridge.find_self_learning_plugin(disabled) is None
    assert bridge.find_self_learning_plugin(SimpleNamespace()) is None


def test_submission_runs_the_affection_update():
    plugin = make_plugin()
    outcome, started, left = submit(plugin)

    assert outcome == bridge.SUBMITTED
    assert plugin._pipeline.calls == [("20002", "10001", "小助手你好")]
    assert started == 1 and left == 0
    assert len(plugin.tracked) == 1


def test_private_chat_uses_the_sender_as_scope():
    plugin = make_plugin()
    outcome, _, _ = submit(plugin, make_event(group_id=""))

    assert outcome == bridge.SUBMITTED
    assert plugin._pipeline.calls == [("10001", "10001", "小助手你好")]


def test_skip_reasons_follow_self_learning_checks():
    # Plugins are built one at a time: each registers its own entry module.
    cases = [
        ({"shutting_down": True}, {}, bridge.SHUTTING_DOWN),
        ({"affection_enabled": False}, {}, bridge.AFFECTION_DISABLED),
        ({"pipeline": None}, {}, bridge.INCOMPATIBLE),
        ({"pipeline": SimpleNamespace()}, {}, bridge.INCOMPATIBLE),
        ({"db_ready": False}, {}, bridge.DATABASE_NOT_READY),
        ({}, {"text": "   "}, bridge.EMPTY_MESSAGE),
        ({}, {"event": make_event(sender_id="")}, bridge.NO_SENDER),
        ({}, {"event": make_event(sender_id="30003")}, bridge.SELF_MESSAGE),
        ({"ignored_texts": ("[图片]",)}, {"text": "[图片]"}, bridge.IGNORED_SAMPLE),
        ({"collects": False}, {}, bridge.NOT_TARGET),
        ({"is_command": True}, {}, bridge.COMMAND),
    ]
    for plugin_options, arguments, expected in cases:
        plugin = make_plugin(**plugin_options)
        outcome, started, _ = submit(plugin, **arguments)
        assert outcome == expected
        assert started == 0
        assert getattr(plugin._pipeline, "calls", []) == []


def test_database_without_ready_flag_falls_back_to_engine():
    ready = make_plugin()
    ready.db_manager = SimpleNamespace(engine=object())
    not_ready = make_plugin()
    not_ready.db_manager = SimpleNamespace(engine=None)

    assert submit(ready)[0] == bridge.SUBMITTED
    assert submit(not_ready)[0] == bridge.DATABASE_NOT_READY


def test_optional_filters_may_be_absent():
    plugin = make_plugin()
    plugin.qq_filter = None
    plugin._command_filter = None
    del plugin._track_task
    del sys.modules[ENTRY_MODULE]

    outcome, started, left = submit(plugin)
    assert outcome == bridge.SUBMITTED
    assert started == 1 and left == 0


def test_failed_update_does_not_leave_a_pending_task():
    plugin = make_plugin(pipeline=FakePipeline(error=RuntimeError("database gone")))
    outcome, started, left = submit(plugin)

    assert outcome == bridge.SUBMITTED
    assert started == 1 and left == 0
