"""Render tool calls as the text records this plugin keeps in its history.

Replies are requested with the whole chat history flattened into the prompt
text, so tool calls can only reach the model as text. Two sources feed that
text:

* Tool calls made during a reply this plugin requested. AstrBot hands the agent
  a shallow copy of the request and appends ``tool_calls_result`` to the copy,
  so the request stored in the event extras never receives them.
* Tool calls AstrBot's own pipeline stored in the conversation, as an assistant
  message carrying ``tool_calls`` followed by ``tool`` messages.

This module deliberately imports nothing from AstrBot so it can be unit tested
without the runtime.
"""

from __future__ import annotations

import json
from typing import Any

TOOL_RECORD_START = "[工具调用记录开始]"
TOOL_RECORD_END = "[工具调用记录结束]"
UNKNOWN_TOOL_NAME = "未知工具"
NO_RESULT_TEXT = "(无返回)"
MAX_ARGUMENTS_LENGTH = 200
MAX_RESULT_LENGTH = 500

_TOOL_ONLY_KEYS = ("tool_calls", "tool_call_id")


def request_with_tool_calls(*requests: Any) -> Any | None:
    """Pick the request the agent actually appended tool calls to.

    Args:
        *requests: Candidate ``ProviderRequest`` objects, most reliable first.

    Returns:
        The first candidate with a non-empty ``tool_calls_result``, or ``None``.
    """
    for request in requests:
        if request is not None and getattr(request, "tool_calls_result", None):
            return request
    return None


def content_text(content: Any) -> str:
    """Extract the plain text of a stored message content.

    Reasoning, image and audio parts are skipped.

    Args:
        content: A string, a list of content-part dicts, or ``None``.

    Returns:
        The text parts joined by spaces, or an empty string.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    texts = []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and part.get("type", "text") == "text":
            text = part.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
    return " ".join(texts)


def format_tool_record(name: Any, arguments: Any, result: Any) -> str:
    """Format one tool call and its result as a text record.

    Args:
        name: Tool name.
        arguments: Call arguments, normally a JSON string.
        result: Text the tool returned.

    Returns:
        A block delimited by the record markers, with long values truncated.
    """
    if isinstance(arguments, str):
        arguments_text = arguments
    elif arguments is None:
        arguments_text = ""
    else:
        try:
            arguments_text = json.dumps(arguments, ensure_ascii=False)
        except (TypeError, ValueError):
            arguments_text = str(arguments)
    if len(arguments_text) > MAX_ARGUMENTS_LENGTH:
        arguments_text = arguments_text[:MAX_ARGUMENTS_LENGTH] + "..."

    result_text = result if isinstance(result, str) else ""
    if len(result_text) > MAX_RESULT_LENGTH:
        result_text = result_text[:MAX_RESULT_LENGTH] + "..."
    result_text = result_text.strip() or NO_RESULT_TEXT

    tool_name = name if isinstance(name, str) and name else UNKNOWN_TOOL_NAME
    return (
        f"{TOOL_RECORD_START}\n"
        f"- {tool_name}({arguments_text}) → {result_text}\n"
        f"{TOOL_RECORD_END}"
    )


def _role(message: Any) -> str | None:
    return message.get("role") if isinstance(message, dict) else None


def _records(tool_calls: Any, results: list[dict]) -> list[str]:
    """Pair each tool call with its result message and format them in order."""
    if not isinstance(tool_calls, list):
        tool_calls = []
    by_id: dict[str, dict] = {}
    for result in results:
        call_id = result.get("tool_call_id")
        if isinstance(call_id, str) and call_id not in by_id:
            by_id[call_id] = result

    used: set[int] = set()
    records = []
    for position, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            function = {}
        call_id = call.get("id")
        result = by_id.get(call_id) if isinstance(call_id, str) else None
        if (
            result is None
            and position < len(results)
            and not results[position].get("tool_call_id")
        ):
            result = results[position]
        if result is not None:
            used.add(id(result))
        records.append(
            format_tool_record(
                function.get("name"),
                function.get("arguments"),
                content_text(result.get("content")) if result else "",
            )
        )

    for result in results:
        if id(result) not in used:
            records.append(
                format_tool_record(None, None, content_text(result.get("content")))
            )
    return records


def fold_tool_messages(history: Any) -> Any:
    """Fold stored tool calls into the assistant message that made them.

    An assistant message with ``tool_calls``, the ``tool`` messages answering
    it, further tool rounds and the closing assistant reply become a single
    assistant message whose content interleaves the text and one record per
    call. That is the layout this plugin saves for its own replies. Without
    this, a ``tool`` message has no assistant role and is shown to the model as
    something a chat member said, and the call's name and arguments are lost.

    Args:
        history: Messages loaded from the conversation. Not modified.

    Returns:
        A new list with all other messages passed through unchanged. A value
        that is not a list is returned as is.
    """
    if not isinstance(history, list):
        return history

    folded: list = []
    turn: dict | None = None

    def add_to_turn(source: dict, pieces: list[str]) -> None:
        nonlocal turn
        pieces = [piece for piece in pieces if piece.strip()]
        if not pieces:
            return
        text = "\n".join(pieces)
        if turn is None:
            turn = {
                key: value
                for key, value in source.items()
                if key not in _TOOL_ONLY_KEYS
            }
            turn["role"] = "assistant"
            turn["content"] = text
            folded.append(turn)
        else:
            turn["content"] += "\n" + text

    index = 0
    while index < len(history):
        message = history[index]
        index += 1
        role = _role(message)
        if role == "assistant" and message.get("tool_calls"):
            results = []
            while index < len(history) and _role(history[index]) == "tool":
                results.append(history[index])
                index += 1
            add_to_turn(
                message,
                [
                    content_text(message.get("content")),
                    *_records(message.get("tool_calls"), results),
                ],
            )
        elif role == "tool":
            # 没有对应调用的结果：仍归到 AI 名下，不能显示成用户发言
            add_to_turn(
                message,
                [format_tool_record(None, None, content_text(message.get("content")))],
            )
        elif role == "assistant" and turn is not None:
            add_to_turn(message, [content_text(message.get("content"))])
            turn = None
        else:
            folded.append(message)
            turn = None
    return folded
