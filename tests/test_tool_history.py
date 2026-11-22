"""Tests for rendering tool calls as text records in the plugin's context."""

import copy
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "utils" / "tool_history.py"
SPEC = importlib.util.spec_from_file_location("tool_history", MODULE_PATH)
tool_history = importlib.util.module_from_spec(SPEC)
sys.modules["tool_history"] = tool_history
SPEC.loader.exec_module(tool_history)

START = tool_history.TOOL_RECORD_START
END = tool_history.TOOL_RECORD_END


def call(call_id, name, arguments="{}"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def record(name, arguments, result):
    return f"{START}\n- {name}({arguments}) → {result}\n{END}"


@dataclass
class FakeRequest:
    """Mirrors the two ``ProviderRequest`` members the lookup depends on."""

    tool_calls_result: list | None = None

    def append_tool_calls_result(self, result):
        if not self.tool_calls_result:
            self.tool_calls_result = []
        self.tool_calls_result.append(result)


def test_request_lookup_finds_calls_appended_to_the_copy():
    stored = FakeRequest()
    running = copy.copy(stored)
    running.append_tool_calls_result("weather call")

    assert stored.tool_calls_result is None
    assert tool_history.request_with_tool_calls(running, stored) is running
    assert tool_history.request_with_tool_calls(None, stored) is None


def test_request_lookup_falls_back_to_the_stored_request():
    stored = FakeRequest(tool_calls_result=["weather call"])

    assert tool_history.request_with_tool_calls(None, stored) is stored


def test_official_tool_turn_becomes_one_assistant_message():
    history = [
        {"role": "user", "content": "北京天气怎么样"},
        {
            "role": "assistant",
            "content": [
                {"type": "think", "think": "need weather", "encrypted": None},
                {"type": "text", "text": "我查一下"},
            ],
            "tool_calls": [call("c1", "get_weather", '{"city": "北京"}')],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "晴 25℃"},
        {"role": "assistant", "content": [{"type": "text", "text": "北京今天晴"}]},
        {"role": "user", "content": "谢谢"},
    ]
    original = copy.deepcopy(history)

    folded = tool_history.fold_tool_messages(history)

    assert folded == [
        {"role": "user", "content": "北京天气怎么样"},
        {
            "role": "assistant",
            "content": "我查一下\n"
            + record("get_weather", '{"city": "北京"}', "晴 25℃")
            + "\n北京今天晴",
        },
        {"role": "user", "content": "谢谢"},
    ]
    assert history == original


def test_tool_results_are_never_left_as_non_assistant_messages():
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [call("c1", "search")]},
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "assistant", "content": None, "tool_calls": [call("c2", "fetch")]},
        {"role": "tool", "tool_call_id": "c2", "content": "r2"},
        {"role": "assistant", "content": "done"},
    ]

    folded = tool_history.fold_tool_messages(history)

    assert [message["role"] for message in folded] == ["user", "assistant"]
    assert folded[1]["content"] == "\n".join(
        [record("search", "{}", "r1"), record("fetch", "{}", "r2"), "done"]
    )
    assert "tool_calls" not in folded[1]


def test_parallel_calls_are_matched_by_id_not_by_order():
    history = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [call("a", "first"), call("b", "second")],
        },
        {"role": "tool", "tool_call_id": "b", "content": "result b"},
        {"role": "tool", "tool_call_id": "a", "content": "result a"},
    ]

    folded = tool_history.fold_tool_messages(history)

    assert folded[0]["content"] == "\n".join(
        [record("first", "{}", "result a"), record("second", "{}", "result b")]
    )


def test_call_without_result_and_result_without_call_are_both_kept():
    history = [
        {"role": "tool", "tool_call_id": "gone", "content": "orphan"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [call("c1", "search")]},
    ]

    folded = tool_history.fold_tool_messages(history)

    assert folded == [
        {
            "role": "assistant",
            "content": record(tool_history.UNKNOWN_TOOL_NAME, "", "orphan"),
        },
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": record("search", "{}", tool_history.NO_RESULT_TEXT),
        },
    ]


def test_a_later_reply_is_not_merged_into_a_finished_tool_turn():
    history = [
        {"role": "assistant", "content": None, "tool_calls": [call("c1", "search")]},
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "assistant", "content": "first reply"},
        {"role": "assistant", "content": "second reply"},
    ]

    folded = tool_history.fold_tool_messages(history)

    assert [message["content"] for message in folded] == [
        record("search", "{}", "r1") + "\nfirst reply",
        "second reply",
    ]
    assert folded[1] is history[3]


def test_history_without_tool_calls_is_passed_through():
    history = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": "hello"},
        "unexpected",
    ]

    folded = tool_history.fold_tool_messages(history)

    assert folded == history
    assert all(new is old for new, old in zip(folded, history))
    assert tool_history.fold_tool_messages(None) is None


def test_long_arguments_are_truncated_and_results_are_kept_in_full():
    long_result = "b" * 30000

    text = tool_history.format_tool_record("search", "a" * 300, long_result + "  ")

    assert text == record("search", "a" * 200 + "...", long_result)
    assert tool_history.format_tool_record("t", {"k": "值"}, "") == record(
        "t", '{"k": "值"}', tool_history.NO_RESULT_TEXT
    )


def test_official_tool_results_are_kept_in_full_too():
    long_result = "r" * 30000
    history = [
        {"role": "assistant", "content": None, "tool_calls": [call("c1", "search")]},
        {"role": "tool", "tool_call_id": "c1", "content": long_result},
    ]

    folded = tool_history.fold_tool_messages(history)

    assert folded[0]["content"] == record("search", "{}", long_result)
