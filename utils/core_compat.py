"""
内核兼容小工具

把「不同版本 AstrBot 内核行为不一致、或容易用错」的调用集中到这里，
其余模块只调用这里的函数，内核再变时只改一处。

本模块不导入 astrbot，便于脱离 AstrBot 单独测试。

作者: Him666233
版本: V1.2.3.hotfix.2
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, List, Optional

# AstrBot 的日志器就叫 "astrbot"，用标准库按名字取即可，不必导入 astrbot
logger = logging.getLogger("astrbot")


async def get_chat_provider(context: Any, umo: Optional[str] = None) -> Any:
    """
    获取会话当前使用的对话模型（提供商）

    内核的同步方法 context.get_using_provider() 已标记弃用。更重要的是：
    不传 umo 时它只会返回全局默认模型，用户用 /provider 给某个会话单独切换的
    模型、以及按会话生效的配置文件里选的模型都会被忽略，于是读空气 / 主动对话
    用的模型和正常回复用的不是同一个。

    这里优先调用异步版本并带上 umo；旧内核没有异步版本时退回同步版本。

    Args:
        context: AstrBot 的 Context 对象
        umo: 会话的 unified_msg_origin；为空时取全局默认模型

    Returns:
        提供商实例，没有可用提供商时返回 None
    """
    umo = umo or None

    async_getter = getattr(context, "get_using_provider_async", None)
    if callable(async_getter):
        try:
            return await async_getter(umo)
        except Exception as e:
            # 按会话解析失败时退回旧行为（全局默认模型），不让读空气/主动对话中断
            logger.warning(
                f"[内核兼容] 按会话获取对话模型失败，改用全局默认模型: {e}"
            )
            return context.get_using_provider()

    return context.get_using_provider(umo)


def db_datetime_to_epoch(value: Any) -> int:
    """
    把内核数据库里的时间字段转换成 Unix 时间戳（秒）

    内核写入 created_at 时用的是 datetime.now(timezone.utc)，但列类型不带时区，
    读回来是「不带时区信息的 UTC 时间」。直接对它调用 .timestamp() 会被 Python
    按本机时区解释，在 UTC+8 的机器上得到的时间戳比真实时间早 8 小时。

    Args:
        value: 数据库返回的 datetime；也接受已带时区信息的 datetime

    Returns:
        Unix 时间戳（秒）；不是 datetime 或无法转换时返回 0
    """
    if not isinstance(value, datetime):
        return 0
    try:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return int(value.timestamp())
    except (OverflowError, OSError, ValueError):
        return 0


# 官方 platform_message_history 的记录转换成消息后，message_id 形如 official_<行号>。
# 注意：从对话历史(conversations)补进来的消息在没有原始 ID 时也以 official_ 开头
# （official_<序号>_<时间戳> / official_refresh_...），它们不是这里说的「官方历史」。
_OFFICIAL_ROW_ID_RE = re.compile(r"^official_\d+$")


def is_official_history_message(message: Any) -> bool:
    """这条消息是否来自官方 platform_message_history 的记录"""
    if isinstance(message, dict):
        message_id = message.get("message_id")
    else:
        message_id = getattr(message, "message_id", None)
    return isinstance(message_id, str) and bool(_OFFICIAL_ROW_ID_RE.match(message_id))


def _timestamp_of(message: Any) -> Any:
    """排序用的时间戳；没有或为空时按 0 处理（与原先各处的排序键一致）"""
    return (
        message.timestamp
        if hasattr(message, "timestamp") and message.timestamp
        else 0
    )


def order_history_messages(messages: Optional[List[Any]]) -> List[Any]:
    """
    给「历史 + 缓存」合并后的消息列表排序：官方历史在前，其余按时间戳在后

    官方记录的时间是「入库时间」而不是消息的发送时间：缓存消息要等到下一次回复
    保存时才入库，届时它们和那条回复的入库时间几乎相同。如果把官方记录和尚未入库
    的缓存消息混在一起纯按时间排，机器人生成回复期间别人发的消息（发送时间早于
    回复的入库时间）会被排到上一轮对话的前面。

    规则：
    - 官方记录保持入库顺序，整体排在最前
    - 其余消息（未入库的缓存消息、自定义存储的记录等）按时间戳排在后面
    - 列表里没有官方记录时，等同于原来的纯按时间戳排序

    Args:
        messages: 待排序的消息列表（不会被修改）

    Returns:
        排好序的新列表
    """
    if not messages:
        return []
    official = [m for m in messages if is_official_history_message(m)]
    others = [m for m in messages if not is_official_history_message(m)]
    # sorted 是稳定排序：同一秒入库的官方记录保持原有的入库顺序
    return sorted(official, key=_timestamp_of) + sorted(others, key=_timestamp_of)
