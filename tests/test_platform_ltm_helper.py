"""Regression tests for the platform group-chat-context adapter.

``utils/platform_ltm_helper.py`` reads image captions out of AstrBot's builtin
"group chat context" feature. The core renamed that feature's private
structures:

==========================  ===========================================
old core                    new core (checked against v4.29.4 / master)
==========================  ===========================================
``star_inst.ltm``           ``star_inst.group_chat_context``
``.session_chats``          ``.raw_records``
``dict[str, list[str]]``    ``dict[str, deque[str]]``
(none)                      ``._record_ids`` + event extra
                            ``_group_context_record_id``
==========================  ===========================================

These tests pin both generations so the helper keeps working on either core.
The doubles below copy the attribute names and container types of the real
classes; they do not import AstrBot.
"""

import asyncio
import importlib.util
import sys
import types
from collections import defaultdict, deque
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
PACKAGE_NAME = "group_chat_plus_ltm_test_package"
UMO = "aiocqhttp:GroupMessage:10001"


class _Logger:
    def info(self, *_args, **_kwargs):
        pass

    warning = error = debug = info


class _Component:
    pass


class _Plain(_Component):
    def __init__(self, text):
        self.text = text


class _Image(_Component):
    pass


class _Reply(_Component):
    def __init__(self, chain=None):
        self.chain = chain


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@pytest.fixture
def helper(monkeypatch):
    """Load the real helper module with lightweight AstrBot stubs."""
    stubs = (
        _module("astrbot"),
        _module("astrbot.api", logger=_Logger()),
        _module("astrbot.api.event", AstrMessageEvent=object),
        _module(
            "astrbot.api.message_components",
            Plain=_Plain,
            At=type("At", (_Component,), {}),
            AtAll=type("AtAll", (_Component,), {}),
            Image=_Image,
            Face=type("Face", (_Component,), {}),
            Reply=_Reply,
            Video=type("Video", (_Component,), {}),
            Record=type("Record", (_Component,), {}),
            File=type("File", (_Component,), {}),
        ),
        _module("astrbot.core"),
        _module("astrbot.core.message"),
        _module(
            "astrbot.core.message.components",
            Video=type("Video", (_Component,), {}),
            Record=type("Record", (_Component,), {}),
            File=type("File", (_Component,), {}),
        ),
    )
    for stub in stubs:
        monkeypatch.setitem(sys.modules, stub.__name__, stub)

    for name, path in (
        (PACKAGE_NAME, REPO_ROOT),
        (f"{PACKAGE_NAME}.utils", REPO_ROOT / "utils"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)

    loaded = {}
    for stem in ("reply_chain_utils", "platform_ltm_helper"):
        full_name = f"{PACKAGE_NAME}.utils.{stem}"
        spec = importlib.util.spec_from_file_location(
            full_name, REPO_ROOT / "utils" / f"{stem}.py"
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full_name, module)
        spec.loader.exec_module(module)
        loaded[stem] = module

    return loaded["platform_ltm_helper"].PlatformLTMHelper


# ---------------------------------------------------------------------------
# Doubles for the two generations of the core
# ---------------------------------------------------------------------------


class _NewGroupChatContext:
    """Same attribute names / container types as the core's GroupChatContext."""

    def __init__(self, image_caption=True):
        self.raw_records = defaultdict(deque)
        self._record_ids = defaultdict(deque)
        self._image_caption = image_caption

    def cfg(self, _event):
        return {"image_caption": self._image_caption}

    def record(self, event, text, record_id):
        """What ``handle_message`` does once a message has been formatted."""
        self.raw_records[UMO].append(text)
        self._record_ids[UMO].append(record_id)
        if event is not None:
            event.set_extra("_group_context_record_id", record_id)


class _OldLongTermMemory:
    """Same attribute names / container types as the old LongTermMemory."""

    def __init__(self, image_caption=True):
        self.session_chats = defaultdict(list)
        self._image_caption = image_caption

    def cfg(self, _event):
        return {"image_caption": self._image_caption}


class _Event:
    def __init__(self, sender_name="Alice", timestamp=None):
        self.unified_msg_origin = UMO
        self._sender_name = sender_name
        self._extras = {}
        self.message_obj = types.SimpleNamespace(timestamp=timestamp, message=[])

    def get_sender_name(self):
        return self._sender_name

    def get_extra(self, key=None, default=None):
        return self._extras.get(key, default)

    def set_extra(self, key, value):
        self._extras[key] = value


def _context(*star_instances):
    """A Context whose public ``get_all_stars()`` lists the given instances."""
    metadatas = [
        types.SimpleNamespace(name=f"star{i}", star_cls=inst)
        for i, inst in enumerate(star_instances)
    ]
    return types.SimpleNamespace(get_all_stars=lambda: metadatas)


def _extract(helper, context, event, original_text="[图片]", **kwargs):
    return asyncio.run(
        helper.extract_image_caption_from_platform(
            context, event, original_text, **kwargs
        )
    )


# ---------------------------------------------------------------------------
# Locating the platform object
# ---------------------------------------------------------------------------


def test_finds_new_core_group_chat_context(helper):
    platform = _NewGroupChatContext()
    builtin_star = types.SimpleNamespace(group_chat_context=platform)

    assert helper._get_platform_ltm(_context(builtin_star)) is platform


def test_finds_old_core_ltm(helper):
    platform = _OldLongTermMemory()
    builtin_star = types.SimpleNamespace(ltm=platform)

    assert helper._get_platform_ltm(_context(builtin_star)) is platform


def test_ignores_unrelated_attribute_with_the_same_name(helper):
    """Another plugin may own an unrelated ``ltm`` / ``group_chat_context``."""
    platform = _NewGroupChatContext()
    other_plugin = types.SimpleNamespace(ltm="not the platform", group_chat_context={})
    builtin_star = types.SimpleNamespace(group_chat_context=platform)

    assert helper._get_platform_ltm(_context(other_plugin, builtin_star)) is platform


def test_returns_none_when_platform_object_failed_to_initialise(helper):
    """The builtin star sets ``group_chat_context = None`` if init fails."""
    builtin_star = types.SimpleNamespace(group_chat_context=None)

    assert helper._get_platform_ltm(_context(builtin_star)) is None


# ---------------------------------------------------------------------------
# New core: raw_records + record id
# ---------------------------------------------------------------------------


def test_new_core_extracts_caption_of_the_current_message(helper):
    platform = _NewGroupChatContext()
    event = _Event()
    platform.record(event, "[Alice/12:00:00]:  看这个 [Image: 一只橘猫]", "id-1")
    context = _context(types.SimpleNamespace(group_chat_context=platform))

    assert _extract(helper, context, event, "看这个[图片]") == (
        True,
        "看这个 [图片内容: 一只橘猫]",
    )


def test_new_core_matches_by_record_id_not_by_position(helper):
    """Same sender, same second, pure images: only the record id tells them apart."""
    platform = _NewGroupChatContext()
    first, second = _Event(), _Event()
    platform.record(first, "[Alice/12:00:00]:  [Image: 第一张]", "id-1")
    platform.record(second, "[Alice/12:00:00]:  [Image: 第二张]", "id-2")
    context = _context(types.SimpleNamespace(group_chat_context=platform))

    assert _extract(helper, context, first) == (True, "[图片内容: 第一张]")
    assert _extract(helper, context, second) == (True, "[图片内容: 第二张]")


def test_new_core_does_not_borrow_another_message_when_unrecorded(helper):
    """An unrecorded message must not pick up a neighbour's caption, nor wait."""
    platform = _NewGroupChatContext()
    platform.record(_Event(), "[Alice/12:00:00]:  [Image: 别人的图]", "id-1")
    unrecorded = _Event()  # e.g. a command: the core skips recording it
    context = _context(types.SimpleNamespace(group_chat_context=platform))

    async def _timed():
        loop = asyncio.get_running_loop()
        started = loop.time()
        result = await helper.extract_image_caption_from_platform(
            context, unrecorded, "[图片]", max_wait=2.0
        )
        return result, loop.time() - started

    result, elapsed = asyncio.run(_timed())

    assert result == (False, None)
    assert elapsed < 0.5


def test_new_core_does_not_create_sessions_while_reading(helper):
    """Reading must not leave empty deques behind in the core's defaultdicts."""
    platform = _NewGroupChatContext()
    context = _context(types.SimpleNamespace(group_chat_context=platform))

    assert _extract(helper, context, _Event(), max_wait=0) == (False, None)
    assert UMO not in platform.raw_records
    assert UMO not in platform._record_ids


def test_new_core_strips_directed_at_you_marker(helper):
    platform = _NewGroupChatContext()
    event = _Event()
    platform.record(
        event,
        "[Alice/12:00:00]: ⚠️[DIRECTED AT YOU]  [At: bot] 看 [Image: 一只橘猫]",
        "id-1",
    )
    context = _context(types.SimpleNamespace(group_chat_context=platform))

    assert _extract(helper, context, event) == (
        True,
        "[At: bot] 看 [图片内容: 一只橘猫]",
    )


def test_new_core_quoted_image_placeholder_is_not_a_pending_image(helper):
    """``[Quote(..: [Image])]`` is a placeholder, not an image awaiting a caption."""
    platform = _NewGroupChatContext()
    event = _Event()
    platform.record(
        event,
        "[Alice/12:00:00]:  [Quote(Bob: [Image])] 这张呢 [Image: 一只橘猫]",
        "id-1",
    )
    context = _context(types.SimpleNamespace(group_chat_context=platform))

    assert _extract(helper, context, event) == (
        True,
        "[Quote(Bob: [Image])] 这张呢 [图片内容: 一只橘猫]",
    )


def test_new_core_record_without_caption_fails_fast(helper):
    """Caption disabled for this image / caption call failed: nothing to extract."""
    platform = _NewGroupChatContext()
    event = _Event()
    platform.record(event, "[Alice/12:00:00]:  看这个", "id-1")
    context = _context(types.SimpleNamespace(group_chat_context=platform))

    assert _extract(helper, context, event, "看这个[图片]") == (False, None)


def test_image_caption_switch_off_returns_immediately(helper):
    platform = _NewGroupChatContext(image_caption=False)
    event = _Event()
    platform.record(event, "[Alice/12:00:00]:  [Image: 一只橘猫]", "id-1")
    context = _context(types.SimpleNamespace(group_chat_context=platform))

    assert _extract(helper, context, event) == (False, None)


# ---------------------------------------------------------------------------
# Old core: session_chats, matched by nickname / timestamp
# ---------------------------------------------------------------------------


def test_old_core_extracts_caption_from_session_chats(helper):
    platform = _OldLongTermMemory()
    platform.session_chats[UMO].append("[Alice/12:00:00]:  看这个 [Image: 一只橘猫]")
    context = _context(types.SimpleNamespace(ltm=platform))

    assert _extract(helper, context, _Event(), "看这个[图片]", max_wait=0) == (
        True,
        "看这个 [图片内容: 一只橘猫]",
    )


def test_old_core_sync_variant_still_works(helper):
    platform = _OldLongTermMemory()
    platform.session_chats[UMO].append("[Alice/12:00:00]:  [Image: 一只橘猫]")
    context = _context(types.SimpleNamespace(ltm=platform))

    assert helper.extract_image_caption_from_platform_sync(
        context, _Event(), "[图片]"
    ) == (True, "[图片内容: 一只橘猫]")


def test_old_core_missing_session_does_not_create_one(helper):
    platform = _OldLongTermMemory()
    context = _context(types.SimpleNamespace(ltm=platform))

    assert _extract(helper, context, _Event(), max_wait=0) == (False, None)
    assert UMO not in platform.session_chats


# ---------------------------------------------------------------------------
# New core: group history injected into req.extra_user_content_parts
# ---------------------------------------------------------------------------

# Copied from the core's GROUP_HISTORY_HEADER / GROUP_HISTORY_FOOTER.
_PLATFORM_HISTORY = (
    "<system_reminder>"
    "You are in a group chat. "
    "Belows are group chat context after your last reply:\n"
    "--- BEGIN CONTEXT---\n"
    "[Alice/12:00:00]:  hi\n"
    "--- END CONTEXT ---\n</system_reminder>"
)


class _TextPart:
    def __init__(self, text):
        self.text = text


def test_strip_removes_only_the_platform_history_block(helper):
    reminder = _TextPart(
        "<system_reminder>User ID: 1, Nickname: Alice</system_reminder>"
    )
    third_party = _TextPart("<memory>Alice likes cats</memory>")
    parts = [reminder, _TextPart(_PLATFORM_HISTORY), third_party]
    req = types.SimpleNamespace(extra_user_content_parts=parts)

    assert helper.strip_platform_group_history(req) == 1
    # modified in place: the core may hold a reference to the same list
    assert req.extra_user_content_parts is parts
    assert parts == [reminder, third_party]


def test_strip_accepts_dict_shaped_parts(helper):
    req = types.SimpleNamespace(
        extra_user_content_parts=[{"type": "text", "text": _PLATFORM_HISTORY}]
    )

    assert helper.strip_platform_group_history(req) == 1
    assert req.extra_user_content_parts == []


def test_strip_leaves_text_that_merely_mentions_the_header(helper):
    quoting = _TextPart("The core says: You are in a group chat. --- BEGIN CONTEXT---")
    req = types.SimpleNamespace(extra_user_content_parts=[quoting])

    assert helper.strip_platform_group_history(req) == 0
    assert req.extra_user_content_parts == [quoting]


def test_strip_tolerates_requests_without_extra_parts(helper):
    assert helper.strip_platform_group_history(types.SimpleNamespace()) == 0
    assert (
        helper.strip_platform_group_history(
            types.SimpleNamespace(extra_user_content_parts=None)
        )
        == 0
    )


def test_strip_drops_the_platform_from_per_plugin_tracking(helper):
    """The builtin star must not be reported as a third-party injection."""
    platform_part = _TextPart(_PLATFORM_HISTORY)
    other_part = _TextPart("<memory>Alice likes cats</memory>")
    req = types.SimpleNamespace(extra_user_content_parts=[platform_part, other_part])
    tracking = {
        "astrbot": {"extra_parts_added": [platform_part]},
        "memory_plugin": {
            "extra_parts_added": [other_part],
            "prompt_added": "x",
        },
    }

    assert helper.strip_platform_group_history(req, tracking) == 1
    assert tracking == {
        "memory_plugin": {"extra_parts_added": [other_part], "prompt_added": "x"}
    }
