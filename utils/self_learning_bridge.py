"""Pass messages this plugin treats as addressed to the bot to self-learning.

``astrbot_plugin_self_learning`` updates a user's affection in its message
listener, but only when ``event.is_at_or_wake_command`` is true. AstrBot sets
that flag for a real @, a wake prefix, a reply to the bot, or a private chat.
This plugin's trigger keywords and its text-form @ detection do not set it, so
the bot replies to such a message while self-learning never scores it.

Setting the flag here instead would also make AstrBot's default LLM stage and
every other plugin treat the message as a wake message, so this module calls
self-learning's affection step directly and repeats the checks its listener
applies first.

The attribute names follow self-learning 4.9.5. They are internal to that
plugin, so each one is read defensively: without the affection step the message
is skipped, and the optional filters are skipped one by one when absent.

This module deliberately imports nothing from AstrBot so it can be unit tested
without the runtime.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

SELF_LEARNING_PLUGIN_NAME = "astrbot_plugin_self_learning"

# When to pass a message on, as returned by plan_affection_submission().
SUBMIT_NOW = "now"
SUBMIT_AFTER_DECISION = "after_decision"
SKIP = "skip"

# Outcomes of submit_affection_interaction().
SUBMITTED = "submitted"
SHUTTING_DOWN = "shutting_down"
AFFECTION_DISABLED = "affection_disabled"
INCOMPATIBLE = "incompatible"
DATABASE_NOT_READY = "database_not_ready"
EMPTY_MESSAGE = "empty_message"
NO_SENDER = "no_sender"
SELF_MESSAGE = "self_message"
IGNORED_SAMPLE = "ignored_sample"
NOT_TARGET = "not_target"
COMMAND = "command"


def plan_affection_submission(
    *,
    is_at_message: bool,
    has_trigger_keyword: bool,
    framework_woken: bool,
    keyword_smart_mode: bool,
) -> str:
    """Decide whether and when a message is passed to self-learning.

    Args:
        is_at_message: This plugin detected an @ of the bot, including the
            text form that AstrBot does not treat as a wake.
        has_trigger_keyword: The message contains a trigger keyword.
        framework_woken: ``event.is_at_or_wake_command`` as AstrBot set it.
            Self-learning has already scored such a message.
        keyword_smart_mode: A keyword only skips the probability filter and the
            decision AI still chooses whether to reply.

    Returns:
        ``SUBMIT_NOW`` when this plugin handles the message like an @ of the
        bot, ``SUBMIT_AFTER_DECISION`` when the decision AI has to agree to a
        reply first, otherwise ``SKIP``.
    """
    if framework_woken or not (is_at_message or has_trigger_keyword):
        return SKIP
    if not is_at_message and keyword_smart_mode:
        return SUBMIT_AFTER_DECISION
    return SUBMIT_NOW


def find_self_learning_plugin(context: Any) -> Any | None:
    """Return the loaded and enabled self-learning plugin instance.

    Args:
        context: AstrBot context providing ``get_registered_star``.

    Returns:
        The plugin instance, or ``None`` when it is absent or disabled.
    """
    get_star = getattr(context, "get_registered_star", None)
    if not callable(get_star):
        return None
    star = get_star(SELF_LEARNING_PLUGIN_NAME)
    if star is None or getattr(star, "activated", True) is False:
        return None
    return getattr(star, "star_cls", None)


def _database_ready(db_manager: Any) -> bool:
    """Repeat the readiness check of self-learning's message listener."""
    if not db_manager:
        return False
    ready = getattr(db_manager, "is_ready", None)
    if ready is not None:
        return bool(ready)
    return bool(getattr(db_manager, "engine", None))


def _ignored_as_sample(plugin: Any, event: Any, text: str, sender_id: str) -> bool:
    """Apply self-learning's sample filter when its entry module exposes it.

    The filter covers command-like text, media-only messages and the user's
    own ignore patterns. It is skipped when the functions cannot be found,
    because the command check that follows still applies.
    """
    module = sys.modules.get(type(plugin).__module__)
    should_ignore = getattr(module, "should_ignore_learning_sample", None)
    extract_metadata = getattr(module, "extract_learning_event_metadata", None)
    if not callable(should_ignore) or not callable(extract_metadata):
        return False
    return bool(should_ignore(text, sender_id=sender_id, **extract_metadata(event)))


def submit_affection_interaction(
    plugin: Any,
    event: Any,
    message_text: str,
    pending_tasks: set,
) -> str:
    """Start self-learning's affection update for one message.

    The checks repeat those of self-learning's listener, apart from the wake
    flag this bridge stands in for. The update runs in the background, as it
    does there. Call this from a running event loop.

    Args:
        plugin: The self-learning plugin instance.
        event: The message event.
        message_text: The text the user sent, before this plugin rewrote the
            event for forwarded or welcome messages.
        pending_tasks: Receives the task until it finishes, so that it is not
            garbage collected and the caller can cancel it on shutdown.

    Returns:
        ``SUBMITTED``, or the reason the message was skipped.

    Raises:
        Exception: Whatever one of self-learning's own filters raises.
    """
    if getattr(plugin, "_shutting_down", False):
        return SHUTTING_DOWN

    config = getattr(plugin, "plugin_config", None)
    if not config or not getattr(config, "enable_affection_system", False):
        return AFFECTION_DISABLED

    pipeline = getattr(plugin, "_pipeline", None)
    process_affection = getattr(pipeline, "process_affection", None)
    if not callable(process_affection):
        return INCOMPATIBLE

    if not _database_ready(getattr(plugin, "db_manager", None)):
        return DATABASE_NOT_READY

    text = str(message_text or "")
    if not text.strip():
        return EMPTY_MESSAGE

    sender_id = str(event.get_sender_id() or "").strip()
    if not sender_id:
        return NO_SENDER
    # Self-learning keys affection by group, or by sender in a private chat.
    group_id = str(event.get_group_id() or sender_id)

    if sender_id == str(event.get_self_id() or "").strip():
        return SELF_MESSAGE

    if _ignored_as_sample(plugin, event, text, sender_id):
        return IGNORED_SAMPLE

    qq_filter = getattr(plugin, "qq_filter", None)
    if qq_filter and not qq_filter.should_collect_message(sender_id, group_id):
        return NOT_TARGET

    command_filter = getattr(plugin, "_command_filter", None)
    if command_filter and command_filter.is_astrbot_command(event):
        return COMMAND

    task = asyncio.create_task(process_affection(group_id, sender_id, text))
    pending_tasks.add(task)

    def _finished(done: asyncio.Task) -> None:
        pending_tasks.discard(done)
        # Self-learning logs its own failures. Reading the exception here only
        # stops asyncio from reporting it a second time.
        if not done.cancelled():
            done.exception()

    task.add_done_callback(_finished)

    # Let self-learning cancel the task with its own on shutdown.
    track_task = getattr(plugin, "_track_task", None)
    if callable(track_task):
        track_task(task)
    return SUBMITTED
