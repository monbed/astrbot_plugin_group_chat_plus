"""Tests for ``utils/core_compat.py``.

``get_chat_provider`` replaces direct calls to the core's deprecated, sync
``context.get_using_provider()``. Called without a session it only ever returns
the global default model, so a model chosen for one session (``/provider``, or
a per-session config profile) was ignored by the decision AI and by proactive
chat while normal replies used it.

``db_datetime_to_epoch`` and ``order_history_messages`` deal with the core's
``platform_message_history`` rows. Their ``created_at`` is a naive UTC value
(calling ``.timestamp()`` on it is off by the local UTC offset), and it is the
time the row was inserted, not the time the message was sent: cached messages
are only inserted when the next reply is saved. Sorting official rows and
not-yet-inserted cached messages purely by time therefore puts a message that
arrived while the bot was writing in front of the turn it followed.

The module under test does not import AstrBot, so no stubs are needed.
"""

import asyncio
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).parents[1]
UMO = "aiocqhttp:GroupMessage:10001"


def _load():
    spec = importlib.util.spec_from_file_location(
        "group_chat_plus_core_compat", REPO_ROOT / "utils" / "core_compat.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _NewContext:
    """Core that has the async getter (the sync one is deprecated)."""

    def __init__(self, fail_async=False):
        self.calls = []
        self._fail_async = fail_async

    async def get_using_provider_async(self, umo=None):
        self.calls.append(("async", umo))
        if self._fail_async:
            raise RuntimeError("session lookup failed")
        return f"provider-for-{umo}"

    def get_using_provider(self, umo=None):
        self.calls.append(("sync", umo))
        return "global-default"


class _OldContext:
    """Core from before the async getter existed."""

    def __init__(self):
        self.calls = []

    def get_using_provider(self, umo=None):
        self.calls.append(("sync", umo))
        return f"provider-for-{umo}"


def test_uses_async_getter_with_the_session():
    context = _NewContext()

    provider = asyncio.run(_load().get_chat_provider(context, UMO))

    assert provider == f"provider-for-{UMO}"
    assert context.calls == [("async", UMO)]


def test_old_core_falls_back_to_sync_getter_with_the_session():
    context = _OldContext()

    provider = asyncio.run(_load().get_chat_provider(context, UMO))

    assert provider == f"provider-for-{UMO}"
    assert context.calls == [("sync", UMO)]


def test_missing_session_means_global_default():
    context = _NewContext()

    asyncio.run(_load().get_chat_provider(context))
    asyncio.run(_load().get_chat_provider(context, ""))

    assert context.calls == [("async", None), ("async", None)]


def test_session_lookup_failure_falls_back_to_previous_behaviour():
    """A failing per-session lookup must not break the decision AI."""
    context = _NewContext(fail_async=True)

    provider = asyncio.run(_load().get_chat_provider(context, UMO))

    assert provider == "global-default"
    assert context.calls == [("async", UMO), ("sync", None)]


# ---------------------------------------------------------------------------
# created_at of the core's history rows
# ---------------------------------------------------------------------------


def test_naive_database_time_is_read_as_utc():
    """Independent of the machine's time zone: 06:30 naive means 06:30 UTC."""
    naive = datetime(2026, 5, 29, 6, 30, 0)
    expected = int(datetime(2026, 5, 29, 6, 30, 0, tzinfo=timezone.utc).timestamp())

    assert _load().db_datetime_to_epoch(naive) == expected


def test_aware_database_time_keeps_its_own_zone():
    east8 = timezone(timedelta(hours=8))
    aware = datetime(2026, 5, 29, 14, 30, 0, tzinfo=east8)
    expected = int(datetime(2026, 5, 29, 6, 30, 0, tzinfo=timezone.utc).timestamp())

    assert _load().db_datetime_to_epoch(aware) == expected


def test_missing_or_unknown_database_time_is_zero():
    compat = _load()

    assert compat.db_datetime_to_epoch(None) == 0
    assert compat.db_datetime_to_epoch("2026-05-29 06:30:00") == 0


# ---------------------------------------------------------------------------
# Ordering of merged history
# ---------------------------------------------------------------------------


def _msg(message_id, timestamp):
    return SimpleNamespace(message_id=message_id, timestamp=timestamp)


def _ids(messages):
    return [m.message_id for m in messages]


def test_official_rows_stay_ahead_of_a_message_sent_while_the_bot_was_replying():
    """U1 at t=100, U2 at t=105 (cached), reply saved at t=110 together with U1."""
    user_turn = _msg("official_1", 110)
    bot_reply = _msg("official_2", 110)
    during_reply = _msg("cached_105", 105)

    ordered = _load().order_history_messages([user_turn, bot_reply, during_reply])

    assert _ids(ordered) == ["official_1", "official_2", "cached_105"]


def test_official_rows_keep_insertion_order_within_the_same_second():
    rows = [_msg(f"official_{i}", 110) for i in (7, 8, 9)]

    assert _ids(_load().order_history_messages(rows)) == [
        "official_7",
        "official_8",
        "official_9",
    ]


def test_cached_messages_are_ordered_by_time_after_the_official_rows():
    messages = [
        _msg("cached_130", 130),
        _msg("official_1", 110),
        _msg("cached_120", 120),
    ]

    assert _ids(_load().order_history_messages(messages)) == [
        "official_1",
        "cached_120",
        "cached_130",
    ]


def test_without_official_rows_it_is_a_plain_time_sort():
    """Custom-storage history carries real send times; its behaviour is unchanged."""
    messages = [_msg("bot_110", 110), _msg("cached_105", 105), _msg("proc_100", 100)]

    assert _ids(_load().order_history_messages(messages)) == [
        "proc_100",
        "cached_105",
        "bot_110",
    ]


def test_conversation_history_fallback_ids_are_not_official_rows():
    """``official_<idx>_<ts>`` / ``official_refresh_...`` come from the LLM
    conversation table, not from platform_message_history."""
    compat = _load()

    assert compat.is_official_history_message(_msg("official_42", 1))
    assert compat.is_official_history_message({"message_id": "official_42"})
    assert not compat.is_official_history_message(_msg("official_3_1700000000", 1))
    assert not compat.is_official_history_message(_msg("official_refresh_3_17", 1))
    assert not compat.is_official_history_message(_msg("", 1))
    assert not compat.is_official_history_message(SimpleNamespace(timestamp=1))


def test_ordering_does_not_modify_its_input_and_tolerates_missing_timestamps():
    compat = _load()
    no_timestamp = SimpleNamespace(message_id="cached_x")
    messages = [_msg("cached_5", 5), no_timestamp, _msg("official_1", 9)]
    snapshot = list(messages)

    ordered = compat.order_history_messages(messages)

    assert messages == snapshot
    assert ordered == [messages[2], no_timestamp, messages[0]]
    assert compat.order_history_messages(None) == []
    assert compat.order_history_messages([]) == []
