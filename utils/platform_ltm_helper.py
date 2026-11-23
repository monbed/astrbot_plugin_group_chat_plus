"""
平台 LTM (Long Term Memory) 辅助模块
用于从平台的聊天记忆增强功能中提取图片描述信息

内核接口兼容说明（两套命名都支持，按「新 → 旧」顺序探测）：

    旧版内核  内置插件实例.ltm                 -> LongTermMemory
              .session_chats: dict[umo, list[str]]
    新版内核  内置插件实例.group_chat_context   -> GroupChatContext
              .raw_records:   dict[umo, deque[str]]
              ._record_ids:   dict[umo, deque[str]]（与 raw_records 一一对应）
              并在事件上写入 extra "_group_context_record_id"

（新版命名已在 AstrBot v4.29.4 与 master 源码
 astrbot/builtin_stars/astrbot/{main,group_chat_context}.py 中核对。）

两个版本的 cfg(event) 返回值都含 "image_caption"，单条记录的格式也都是
"[昵称/HH:MM:SS]:  内容 [Image: 描述]"。新版记录里额外可能出现
"⚠️[DIRECTED AT YOU] "、" [Quote(昵称: 内容)]"、" [Shared Card...]" 标记。

作者: Him666233
版本: V1.2.3.hotfix.2
"""

import re
import asyncio
from typing import List, Optional, Tuple
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from .reply_chain_utils import analyze_chain, has_image_in_chain

# 调试模式开关
DEBUG_MODE: bool = False

# === 平台内核接口名（新 → 旧，逐个探测） ===
# 内置插件实例上「群聊上下文感知」对象的属性名
_PLATFORM_CTX_ATTRS: Tuple[str, ...] = ("group_chat_context", "ltm")
# 该对象上「按会话保存的聊天记录」容器的属性名
_PLATFORM_RECORDS_ATTRS: Tuple[str, ...] = ("raw_records", "session_chats")
# 新版内核：与记录一一对应的记录 ID 容器，以及写在事件上的记录 ID
_PLATFORM_RECORD_IDS_ATTR: str = "_record_ids"
_PLATFORM_RECORD_ID_EXTRA: str = "_group_context_record_id"
# 按昵称/时间戳匹配时，只需要会话尾部的这么多条记录
_RECORD_TAIL_SIZE: int = 30

# === 平台记录内的标记 ===
# 尚未带描述的图片占位：[Image]（不是 [Image: xxx]）
_UNPROCESSED_IMAGE_RE = re.compile(r"\[Image\](?!\s*:)")
# 新版内核写入的引用块：" [Quote(昵称: 内容)]" 或 " [Quote]"
_QUOTE_BLOCK_RE = re.compile(r"\[Quote\((.*?)\)\]|\[Quote\]", re.DOTALL)
# 新版内核写入的「@了机器人」标记，只对平台自己的提示词有意义
_DIRECTED_AT_YOU_MARK: str = "⚠️[DIRECTED AT YOU]"

# === 平台注入到 LLM 请求里的群聊历史块 ===
# 新版内核不再改 system_prompt / contexts，而是往 req.extra_user_content_parts
# 里追加一个文本块，开头固定为（GROUP_HISTORY_HEADER）：
#   "<system_reminder>You are in a group chat. Belows are group chat context
#    after your last reply:\n--- BEGIN CONTEXT---\n"
_GROUP_HISTORY_BLOCK_PREFIX: str = "<system_reminder>"
_GROUP_HISTORY_BLOCK_MARKERS: Tuple[str, ...] = (
    "You are in a group chat",
    "--- BEGIN CONTEXT---",
)
# 只在文本块开头这么多字符内找特征，避免误伤正文里恰好提到这些词的内容
_GROUP_HISTORY_HEAD_SIZE: int = 200

# === 默认配置（当未传入参数时使用） ===
_DEFAULT_MAX_WAIT: float = 2.0  # 默认最大等待时间(秒)
_DEFAULT_RETRY_INTERVAL: int = 50  # 默认重试间隔(毫秒)
_DEFAULT_FAST_CHECK_COUNT: int = 5  # 默认快速检查次数
_FAST_CHECK_INTERVAL: float = 0.02  # 快速检查间隔(秒)，固定20ms


class PlatformLTMHelper:
    """
    平台 LTM 辅助类

    用于从平台的 LongTermMemory 模块中提取当前消息的图片描述信息
    平台会将图片转换为 [Image: 描述] 格式存储

    性能优化策略：
    1. 快速失败：未开启功能时立即返回，零等待
    2. 智能等待：只在检测到平台可能正在处理时才等待
    3. 超时保护：最大等待2秒，避免卡死
    """

    @staticmethod
    async def extract_image_caption_from_platform(
        context,
        event: AstrMessageEvent,
        original_text: str,
        max_wait: float = None,
        retry_interval: int = None,
        fast_check_count: int = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        从平台的 LTM 中提取当前消息的图片描述（异步版本，支持智能等待）

        性能优化：
        - 未开启图片理解功能时：立即返回 (< 1ms)
        - 平台已处理完成时：立即返回 (< 5ms)
        - 平台正在处理时：智能等待，最多等待 max_wait 秒
        - 平台处理失败时：检测到 [Image] 标记后快速返回

        Args:
            context: AstrBot 的 Context 对象
            event: 当前消息事件
            original_text: 原始消息文本（用于匹配验证）
            max_wait: 最大等待时间(秒)，默认2秒。设置0则不等待
            retry_interval: 重试间隔(毫秒)，默认50ms
            fast_check_count: 快速检查次数，默认5次

        Returns:
            (是否成功提取, 处理后的消息文本)
            - 成功: (True, "包含图片描述的消息文本")
            - 失败: (False, None)
        """
        # 使用默认值
        if max_wait is None:
            max_wait = _DEFAULT_MAX_WAIT
        if retry_interval is None:
            retry_interval = _DEFAULT_RETRY_INTERVAL
        if fast_check_count is None:
            fast_check_count = _DEFAULT_FAST_CHECK_COUNT

        # 计算重试参数
        retry_interval_sec = retry_interval / 1000.0  # 转换为秒
        max_retry_count = (
            int(max_wait / retry_interval_sec) if retry_interval_sec > 0 else 0
        )

        try:
            # === 第一阶段：快速失败检查（零等待） ===

            # 获取平台的 LTM 实例（使用缓存）
            ltm = PlatformLTMHelper._get_platform_ltm(context)
            if not ltm:
                if DEBUG_MODE:
                    logger.info("[PlatformLTM] 未找到平台 LTM 实例")
                return False, None

            # 检查 LTM 是否启用了图片理解功能（快速失败点）
            cfg = ltm.cfg(event)
            if not cfg.get("image_caption", False):
                # 用户未开启图片理解，立即返回，零开销
                if DEBUG_MODE:
                    logger.info("[PlatformLTM] 平台未启用图片理解功能，快速跳过")
                return False, None

            # === 第二阶段：智能等待获取图片描述 ===

            umo = event.unified_msg_origin
            sender_name = event.get_sender_name() or ""

            # 🔧 获取当前消息的时间戳，用于精确匹配
            msg_timestamp = PlatformLTMHelper._get_message_timestamp(event)

            # 首次尝试（可能平台已经处理完成）
            result = PlatformLTMHelper._try_extract_caption(
                ltm, umo, sender_name, original_text, msg_timestamp, event
            )
            if result[0]:
                # 平台已处理完成，直接返回
                return result

            # 如果 max_wait <= 0，不等待直接返回
            if max_wait <= 0 or max_retry_count <= 0:
                if DEBUG_MODE:
                    logger.info("[PlatformLTM] max_wait=0，不等待直接返回")
                return False, None

            # 检查是否需要等待（平台可能正在处理中）
            # 条件：会话存在 且 最后一条消息匹配当前发送者 但 还没有图片描述
            should_wait = PlatformLTMHelper._should_wait_for_platform(
                ltm, umo, sender_name, original_text, msg_timestamp, event
            )

            if not should_wait:
                # 不需要等待（可能是会话不存在、消息不匹配等）
                if DEBUG_MODE:
                    logger.info("[PlatformLTM] 无需等待平台处理")
                return False, None

            # === 第三阶段：等待平台处理完成 ===
            if DEBUG_MODE:
                logger.info(
                    f"[PlatformLTM] 检测到平台可能正在处理图片，开始等待(最多{max_wait}秒)..."
                )

            # 🔧 优化：记录会话是否曾经存在，用于判断平台是否会处理这条消息
            session_ever_existed = bool(
                PlatformLTMHelper._get_session_records(ltm, umo)
            )

            for retry in range(max_retry_count):
                # 动态调整等待间隔（前几次更快）
                if retry < fast_check_count:
                    await asyncio.sleep(_FAST_CHECK_INTERVAL)
                else:
                    await asyncio.sleep(retry_interval_sec)

                # 重新尝试提取
                result = PlatformLTMHelper._try_extract_caption(
                    ltm, umo, sender_name, original_text, msg_timestamp, event
                )

                if result[0]:
                    # 成功获取图片描述
                    if DEBUG_MODE:
                        logger.info(f"[PlatformLTM] 第 {retry + 1} 次重试成功")
                    return result

                # 检查是否平台处理失败（出现 [Image] 而非 [Image: xxx]）
                if PlatformLTMHelper._check_platform_failed(
                    ltm, umo, sender_name, msg_timestamp
                ):
                    if DEBUG_MODE:
                        logger.info("[PlatformLTM] 检测到平台图片处理失败，停止等待")
                    return False, None

                # 🔧 优化：如果会话从未存在，且已经等待了足够长时间（超过快速检查阶段），
                # 说明平台 LTM 可能不会处理这条消息，提前退出
                if not session_ever_existed and retry >= fast_check_count:
                    current_session_exists = bool(
                        PlatformLTMHelper._get_session_records(ltm, umo)
                    )
                    if not current_session_exists:
                        if DEBUG_MODE:
                            logger.info(
                                "[PlatformLTM] 会话一直不存在，平台可能不会处理这条消息，停止等待"
                            )
                        return False, None
                    else:
                        # 会话现在存在了，更新标记
                        session_ever_existed = True

            # 超时，返回失败
            if DEBUG_MODE:
                logger.info("[PlatformLTM] 等待超时，平台可能处理失败")
            return False, None

        except Exception as e:
            logger.warning(f"[PlatformLTM] 提取图片描述时发生错误: {e}")
            return False, None

    @staticmethod
    def extract_image_caption_from_platform_sync(
        context,
        event: AstrMessageEvent,
        original_text: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        从平台的 LTM 中提取当前消息的图片描述（同步版本，无等待）

        用于不需要等待的场景，立即返回当前状态

        Args:
            context: AstrBot 的 Context 对象
            event: 当前消息事件
            original_text: 原始消息文本（用于匹配验证）

        Returns:
            (是否成功提取, 处理后的消息文本)
        """
        try:
            ltm = PlatformLTMHelper._get_platform_ltm(context)
            if not ltm:
                return False, None

            cfg = ltm.cfg(event)
            if not cfg.get("image_caption", False):
                return False, None

            umo = event.unified_msg_origin
            sender_name = event.get_sender_name() or ""
            msg_timestamp = PlatformLTMHelper._get_message_timestamp(event)

            return PlatformLTMHelper._try_extract_caption(
                ltm, umo, sender_name, original_text, msg_timestamp, event
            )

        except Exception as e:
            logger.warning(f"[PlatformLTM] 同步提取图片描述时发生错误: {e}")
            return False, None

    # ------------------------------------------------------------------
    # 平台内核接口适配层：所有对内核私有结构的访问都集中在下面几个方法里，
    # 内核再改名时只需要改模块顶部的 _PLATFORM_* 常量。
    # ------------------------------------------------------------------

    @staticmethod
    def _get_records_map(ltm):
        """
        取出平台对象上「会话 -> 聊天记录」的容器

        新版为 raw_records（dict[str, deque[str]]），旧版为 session_chats
        （dict[str, list[str]]）。

        Returns:
            容器对象；该对象不是平台的群聊上下文对象时返回 None
        """
        for attr in _PLATFORM_RECORDS_ATTRS:
            try:
                records_map = getattr(ltm, attr, None)
            except Exception:
                records_map = None
            if records_map is not None and hasattr(records_map, "get"):
                return records_map
        return None

    @staticmethod
    def _get_session_records(ltm, umo: str) -> List[str]:
        """
        获取某个会话最近的聊天记录（尾部快照，旧 → 新）

        用 .get() 读取，避免在 defaultdict 上凭空创建会话；统一转成 list，
        屏蔽新版 deque 与旧版 list 的差异。

        Returns:
            记录列表；会话不存在或为空时返回空列表
        """
        try:
            records_map = PlatformLTMHelper._get_records_map(ltm)
            if records_map is None:
                return []
            records = records_map.get(umo)
            if not records:
                return []
            snapshot = [r for r in records if isinstance(r, str)]
            return snapshot[-_RECORD_TAIL_SIZE:]
        except Exception:
            return []

    @staticmethod
    def _lookup_record_by_id(ltm, event, umo: str) -> Tuple[bool, Optional[str]]:
        """
        按记录 ID 精确定位「当前事件」在平台里的那条记录（仅新版内核）

        新版内核在记录消息时会把记录 ID 写进事件的 extra，并保存一份与
        raw_records 一一对应的 _record_ids，因此可以精确定位，不必再靠
        昵称 + 时间戳 + 内容去猜（同一人连发图片时不会错位）。

        Returns:
            (是否支持精确定位, 记录)
            - (False, None): 旧版内核，或结构对不上，调用方应回退到昵称/时间戳匹配
            - (True, None):  新版内核，但这条消息没有被平台记录（未开启群聊上下文、
                             是指令消息、或记录已被平台消费）
            - (True, str):   找到了当前消息对应的记录
        """
        try:
            ids_map = getattr(ltm, _PLATFORM_RECORD_IDS_ATTR, None)
            records_map = PlatformLTMHelper._get_records_map(ltm)
            if ids_map is None or records_map is None or not hasattr(ids_map, "get"):
                return False, None
            if event is None or not hasattr(event, "get_extra"):
                return False, None

            record_id = event.get_extra(_PLATFORM_RECORD_ID_EXTRA, None)
            if not isinstance(record_id, str) or not record_id:
                return True, None

            ids = list(ids_map.get(umo) or ())
            records = list(records_map.get(umo) or ())
            if len(ids) != len(records):
                # 两个容器没对齐，下标不可信，回退到启发式匹配
                return False, None
            if record_id not in ids:
                return True, None

            record = records[ids.index(record_id)]
            return True, record if isinstance(record, str) else None
        except Exception:
            return False, None

    @staticmethod
    def _strip_quote_blocks(record: str) -> str:
        """去掉记录里的引用块，引用块里的 [Image] 只是占位，不代表当前消息的图片"""
        return _QUOTE_BLOCK_RE.sub("", record)

    @staticmethod
    def _has_unprocessed_image(record: str) -> bool:
        """记录中是否还有未带描述的 [Image]（忽略引用块内的占位）"""
        return bool(
            _UNPROCESSED_IMAGE_RE.search(PlatformLTMHelper._strip_quote_blocks(record))
        )

    # ------------------------------------------------------------------
    # 平台注入到 LLM 请求里的群聊历史
    # ------------------------------------------------------------------

    @staticmethod
    def is_platform_group_history_text(text) -> bool:
        """
        判断一段文本是否为新版内核注入的「群聊历史」文本块

        Args:
            text: 待判断的文本

        Returns:
            是否为平台的群聊历史块
        """
        if not isinstance(text, str) or not text:
            return False
        head = text.lstrip()[:_GROUP_HISTORY_HEAD_SIZE]
        if not head.startswith(_GROUP_HISTORY_BLOCK_PREFIX):
            return False
        return all(marker in head for marker in _GROUP_HISTORY_BLOCK_MARKERS)

    @staticmethod
    def _is_platform_group_history_part(part) -> bool:
        """判断 extra_user_content_parts 里的一项是否为平台的群聊历史块"""
        try:
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
        except Exception:
            return False
        return PlatformLTMHelper.is_platform_group_history_text(text)

    @staticmethod
    def strip_platform_group_history(req, per_plugin_injections=None) -> int:
        """
        从 LLM 请求里移除平台注入的群聊历史块（仅新版内核会注入）

        本插件用自己格式化的历史构造请求，平台的群聊历史留在请求里会让 AI
        看到两份内容相同、格式不同的历史。旧版内核把历史写进 system_prompt /
        contexts，由调用方原有逻辑处理；新版内核改为追加到
        req.extra_user_content_parts，由本方法处理。

        只移除能确认是平台群聊历史的文本块，其他插件追加的内容原样保留。

        Args:
            req: ProviderRequest
            per_plugin_injections: 逐插件追踪数据（可选）。其中记录到的同一批
                文本块会一并剔除，避免把平台当成「第三方插件注入」上报

        Returns:
            从请求里移除的文本块数量
        """
        removed = 0
        is_platform_part = PlatformLTMHelper._is_platform_group_history_part

        try:
            parts = getattr(req, "extra_user_content_parts", None)
            if isinstance(parts, list) and parts:
                kept = [part for part in parts if not is_platform_part(part)]
                removed = len(parts) - len(kept)
                if removed:
                    # 原地修改：内核与其他钩子可能持有同一个列表对象
                    parts[:] = kept
        except Exception as e:
            logger.warning(f"[PlatformLTM] 移除平台群聊历史块时发生错误: {e}")

        try:
            if isinstance(per_plugin_injections, dict):
                for plugin_name in list(per_plugin_injections.keys()):
                    entry = per_plugin_injections.get(plugin_name)
                    if not isinstance(entry, dict):
                        continue
                    tracked = entry.get("extra_parts_added")
                    if not isinstance(tracked, list):
                        continue
                    kept = [part for part in tracked if not is_platform_part(part)]
                    if len(kept) == len(tracked):
                        continue
                    if kept:
                        entry["extra_parts_added"] = kept
                    else:
                        entry.pop("extra_parts_added", None)
                    if not entry:
                        # 该插件只注入了平台群聊历史，整条追踪记录都不再有意义
                        per_plugin_injections.pop(plugin_name, None)
        except Exception as e:
            logger.warning(f"[PlatformLTM] 清理逐插件追踪数据时发生错误: {e}")

        return removed

    @staticmethod
    def _get_message_timestamp(event: AstrMessageEvent) -> Optional[str]:
        """
        获取消息的时间戳（HH:MM:SS 格式）

        用于与平台存储的时间戳进行精确匹配，避免同一人连续发图片时错位

        Args:
            event: 消息事件

        Returns:
            时间戳字符串（HH:MM:SS），获取失败返回 None
        """
        try:
            import datetime

            # 尝试从 message_obj 获取时间戳
            if hasattr(event, "message_obj") and hasattr(
                event.message_obj, "timestamp"
            ):
                ts = event.message_obj.timestamp
                if ts:
                    # 如果是数字时间戳，转换为 HH:MM:SS
                    if isinstance(ts, (int, float)):
                        dt = datetime.datetime.fromtimestamp(ts)
                        return dt.strftime("%H:%M:%S")
                    # 如果已经是字符串，尝试提取时间部分
                    elif isinstance(ts, str):
                        # 可能是 "HH:MM:SS" 或 "YYYY-MM-DD HH:MM:SS" 格式
                        if len(ts) == 8 and ts.count(":") == 2:
                            return ts
                        elif " " in ts:
                            return ts.split(" ")[-1][:8]

            # 尝试从 raw_message 获取
            if hasattr(event, "raw_message") and hasattr(event.raw_message, "time"):
                ts = event.raw_message.time
                if isinstance(ts, (int, float)):
                    dt = datetime.datetime.fromtimestamp(ts)
                    return dt.strftime("%H:%M:%S")

            return None

        except Exception as e:
            if DEBUG_MODE:
                logger.info(f"[PlatformLTM] 获取消息时间戳失败: {e}")
            return None

    @staticmethod
    def _try_extract_caption(
        ltm,
        umo: str,
        sender_name: str,
        original_text: str,
        msg_timestamp: Optional[str] = None,
        event=None,
    ) -> Tuple[bool, Optional[str]]:
        """
        尝试从 LTM 提取图片描述（内部方法）

        Args:
            ltm: LTM 实例
            umo: 统一消息来源
            sender_name: 发送者昵称
            original_text: 原始消息文本
            msg_timestamp: 消息时间戳（HH:MM:SS），用于精确匹配
            event: 当前消息事件；新版内核下用它按记录 ID 精确定位

        Returns:
            (是否成功, 处理后的文本)
        """
        try:
            # 新版内核：按记录 ID 精确定位，不走启发式匹配
            exact_supported, exact_record = PlatformLTMHelper._lookup_record_by_id(
                ltm, event, umo
            )

            if exact_supported:
                if not exact_record:
                    # 平台没有记录这条消息，不能拿别的记录顶替（会张冠李戴）
                    return False, None
                matched_chat = exact_record
            else:
                session_chats = PlatformLTMHelper._get_session_records(ltm, umo)
                if not session_chats:
                    return False, None

                # 🔧 改进：使用 sender_name + timestamp + content 三重验证
                # 确保即使同一秒内多条消息也能正确匹配
                if msg_timestamp:
                    matched_chat = PlatformLTMHelper._find_message_by_timestamp(
                        session_chats, sender_name, msg_timestamp, original_text
                    )
                    if not matched_chat:
                        return False, None
                else:
                    # 没有时间戳，回退到只检查最后一条
                    matched_chat = session_chats[-1]
                    if not PlatformLTMHelper._verify_message_match(
                        matched_chat, sender_name, original_text, None
                    ):
                        return False, None

            # 🔧 修复多图片场景：检查是否所有图片都已处理完成
            # 如果存在未处理的 [Image]（没有描述），说明还有图片在处理中
            # （引用块里的 [Image] 只是占位，不算）
            if PlatformLTMHelper._has_unprocessed_image(matched_chat):
                return False, None

            # 检查是否包含完整的图片描述 [Image: xxx]
            if "[Image:" not in matched_chat:
                return False, None

            # 提取消息内容
            processed_text = PlatformLTMHelper._extract_message_content(matched_chat)

            if processed_text:
                logger.info(
                    f"🖼️ [PlatformLTM] 成功提取平台图片描述: {processed_text[:100]}..."
                )
                return True, processed_text

            return False, None

        except Exception:
            return False, None

    @staticmethod
    def _find_message_by_timestamp(
        session_chats: list,
        sender_name: str,
        msg_timestamp: str,
        original_text: str = "",
    ) -> Optional[str]:
        """
        根据时间戳从聊天记录中查找匹配的消息

        从后往前查找，最多检查最近15条，避免性能问题

        匹配优先级：
        1. sender_name + msg_timestamp 精确匹配 + 内容验证
        2. sender_name + msg_timestamp（3秒容差）+ 内容验证
        3. sender_name + 内容验证（无时间戳匹配）

        Args:
            session_chats: 聊天记录列表
            sender_name: 发送者昵称
            msg_timestamp: 消息时间戳（HH:MM:SS）
            original_text: 原始消息文本（用于辅助验证）

        Returns:
            匹配的聊天记录，未找到返回 None
        """
        try:
            # 最多检查最近15条消息（增加一点以应对高并发场景）
            check_count = min(15, len(session_chats))

            # 第一轮：精确匹配 sender_name + timestamp
            for i in range(1, check_count + 1):
                chat = session_chats[-i]

                # 精确匹配格式: [昵称/HH:MM:SS]: 内容
                expected_prefix = f"[{sender_name}/{msg_timestamp}]"
                if chat.startswith(expected_prefix):
                    # 如果有原始文本，进一步验证内容
                    if original_text:
                        if PlatformLTMHelper._content_matches(chat, original_text):
                            return chat
                        # 内容不匹配，可能是同一秒的另一条消息，继续查找
                        continue
                    return chat

            # 第二轮：宽松匹配（3秒容差，因为平台使用处理时的时间，可能有延迟）
            for i in range(1, check_count + 1):
                chat = session_chats[-i]

                # 提取聊天记录中的时间戳
                match = re.match(
                    rf"^\[{re.escape(sender_name)}/(\d{{2}}:\d{{2}}:\d{{2}})\]", chat
                )
                if match:
                    record_time = match.group(1)
                    if PlatformLTMHelper._timestamps_close(
                        msg_timestamp, record_time, tolerance=3
                    ):
                        # 如果有原始文本，验证内容
                        if original_text:
                            if PlatformLTMHelper._content_matches(chat, original_text):
                                return chat
                            continue
                        return chat

            # 第三轮：仅通过发送者和内容匹配（时间戳可能完全不同）
            if original_text:
                for i in range(1, check_count + 1):
                    chat = session_chats[-i]
                    # 检查是否是同一发送者
                    if f"[{sender_name}/" in chat[:50]:
                        if PlatformLTMHelper._content_matches(chat, original_text):
                            return chat

            return None

        except Exception:
            return None

    @staticmethod
    def _content_matches(chat_record: str, original_text: str) -> bool:
        """
        验证聊天记录的内容是否与原始消息匹配

        用于区分同一秒内同一人发的多条消息

        Args:
            chat_record: 平台存储的聊天记录
            original_text: 原始消息文本

        Returns:
            是否匹配
        """
        try:
            # 清理原始文本中的图片标记
            clean_original = (
                original_text.replace("[图片]", "").replace("[Image]", "").strip()
            )

            # 统计原始消息中的图片数量（通过 [图片] 或 [Image] 标记）
            original_image_count = original_text.count("[图片]") + original_text.count(
                "[Image]"
            )

            # 统计聊天记录中的图片数量
            record_image_count = chat_record.count("[Image:") + chat_record.count(
                "[Image]"
            )

            # 如果原始文本为空或只有图片
            if not clean_original or len(clean_original) < 2:
                # 纯图片消息：通过图片数量来辅助验证
                if original_image_count > 0:
                    # 检查图片数量是否匹配（允许一定误差，因为有些图片可能处理失败）
                    if record_image_count >= original_image_count:
                        return True
                    # 图片数量不匹配，可能是不同的消息
                    return False
                # 没有图片标记，无法验证，放行
                return True

            # 检查聊天记录中是否包含原始文本的关键部分
            # 取前20个字符进行匹配（避免图片描述干扰）
            check_text = clean_original[: min(20, len(clean_original))]

            # 从聊天记录中提取内容部分（去除前缀）
            if "]: " in chat_record:
                content_part = chat_record.split("]: ", 1)[1]
                # 去除图片描述部分再比较
                content_without_image = re.sub(
                    r"\[Image:[^\]]*\]", "", content_part
                ).strip()
                content_without_image = content_without_image.replace(
                    "[Image]", ""
                ).strip()

                if check_text in content_without_image:
                    return True
                # 也检查完整内容（可能图片描述在中间）
                if check_text in content_part:
                    return True

            return False

        except Exception:
            return True  # 出错时放行

    @staticmethod
    def _timestamps_close(ts1: str, ts2: str, tolerance: int = 1) -> bool:
        """
        检查两个时间戳是否接近（在容差范围内）

        Args:
            ts1: 时间戳1（HH:MM:SS）
            ts2: 时间戳2（HH:MM:SS）
            tolerance: 容差秒数

        Returns:
            是否接近
        """
        try:

            def to_seconds(ts: str) -> int:
                parts = ts.split(":")
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

            diff = abs(to_seconds(ts1) - to_seconds(ts2))
            return diff <= tolerance

        except Exception:
            return False

    @staticmethod
    def _should_wait_for_platform(
        ltm,
        umo: str,
        sender_name: str,
        original_text: str,
        msg_timestamp: Optional[str] = None,
        event=None,
    ) -> bool:
        """
        判断是否应该等待平台处理

        条件：
        1. 会话存在 且 最后一条消息是当前发送者的 且 消息中没有图片描述（可能正在处理）
        2. 或者会话不存在/消息不存在（平台可能还没处理到，需要等待）

        新版内核（能按记录 ID 精确定位）不需要等待：平台的群聊上下文 handler
        优先级远高于本插件，且图片转述是在写入记录之前 await 完成的，所以轮到
        本插件时记录要么已是最终状态，要么平台根本没记录这条消息。

        Returns:
            是否应该等待
        """
        try:
            exact_supported, _ = PlatformLTMHelper._lookup_record_by_id(
                ltm, event, umo
            )
            if exact_supported:
                if DEBUG_MODE:
                    logger.info("[PlatformLTM] 新版内核记录已是最终状态，无需等待")
                return False

            session_chats = PlatformLTMHelper._get_session_records(ltm, umo)
            if not session_chats:
                # 🔧 修复：会话不存在/为空时，可能是平台 LTM 还没处理到，应该等待
                if DEBUG_MODE:
                    logger.info(
                        "[PlatformLTM] 会话不存在或为空，平台可能还没处理到，需要等待"
                    )
                return True

            # 检查是否是当前发送者的消息
            if not sender_name:
                # 🔧 修复：即使没有发送者名称，也应该等待（无法精确匹配，但可以尝试）
                if DEBUG_MODE:
                    logger.info("[PlatformLTM] 发送者名称为空，但仍尝试等待")
                return True

            # 🔧 如果有时间戳，精确查找
            if msg_timestamp:
                # 检查最近几条消息中是否有匹配的
                check_count = min(5, len(session_chats))
                for i in range(1, check_count + 1):
                    chat = session_chats[-i]
                    # 检查是否是当前消息（通过时间戳匹配）
                    if f"[{sender_name}/{msg_timestamp}]" in chat[:50]:
                        # 找到了，检查是否有 [Image] 标记（引用块内的占位不算）
                        if (
                            PlatformLTMHelper._has_unprocessed_image(chat)
                            and "[Image:" not in chat
                        ):
                            return True
                        # 已经有描述或没有图片，不需要等待
                        return False
                    # 宽松匹配时间戳
                    match = re.match(
                        rf"^\[{re.escape(sender_name)}/(\d{{2}}:\d{{2}}:\d{{2}})\]",
                        chat,
                    )
                    if match:
                        record_time = match.group(1)
                        if PlatformLTMHelper._timestamps_close(
                            msg_timestamp, record_time, tolerance=1
                        ):
                            if (
                                PlatformLTMHelper._has_unprocessed_image(chat)
                                and "[Image:" not in chat
                            ):
                                return True
                            return False
                # 没找到匹配的消息，可能平台还没处理到
                return True

            # 没有时间戳，回退到检查最后一条
            last_chat = session_chats[-1]

            # 宽松匹配发送者
            if f"[{sender_name}" not in last_chat[:50]:
                return False

            # 如果已经有图片描述，不需要等待
            if "[Image:" in last_chat:
                return False

            # 如果有 [Image] 标记（无描述），说明平台可能正在处理或已失败
            if PlatformLTMHelper._has_unprocessed_image(last_chat):
                return True

            return False

        except Exception:
            return False

    @staticmethod
    def _check_platform_failed(
        ltm, umo: str, sender_name: str, msg_timestamp: Optional[str] = None
    ) -> bool:
        """
        检查平台是否处理失败

        如果最后一条消息包含 [Image] 但不包含 [Image: xxx]，
        说明平台处理失败了

        Returns:
            是否处理失败
        """
        try:
            session_chats = PlatformLTMHelper._get_session_records(ltm, umo)
            if not session_chats:
                return False

            # 🔧 如果有时间戳，精确查找
            if msg_timestamp:
                check_count = min(5, len(session_chats))
                for i in range(1, check_count + 1):
                    chat = session_chats[-i]
                    # 检查是否是当前消息
                    is_match = f"[{sender_name}/{msg_timestamp}]" in chat[:50]
                    if not is_match:
                        match = re.match(
                            rf"^\[{re.escape(sender_name)}/(\d{{2}}:\d{{2}}:\d{{2}})\]",
                            chat,
                        )
                        if match:
                            record_time = match.group(1)
                            is_match = PlatformLTMHelper._timestamps_close(
                                msg_timestamp, record_time, tolerance=1
                            )

                    if is_match:
                        # 🔧 修复多图片场景：检查是否有未处理的图片
                        # 匹配独立的 [Image]（不是 [Image: xxx] 的一部分，也不在引用块内）
                        if PlatformLTMHelper._has_unprocessed_image(chat):
                            # 还有未处理的图片，但不一定是失败，可能还在处理中
                            # 只有当没有任何 [Image: xxx] 时才认为是失败
                            if "[Image:" not in chat:
                                return True
                            # 有部分处理完成，继续等待
                            return False
                        return False
                return False

            # 没有时间戳，检查最后一条
            last_chat = session_chats[-1]

            # 检查是否是当前发送者
            if f"[{sender_name}" not in last_chat[:50]:
                return False

            # 🔧 修复多图片场景：检查是否有未处理的图片
            if PlatformLTMHelper._has_unprocessed_image(last_chat):
                # 还有未处理的图片，但不一定是失败
                # 只有当没有任何 [Image: xxx] 时才认为是失败
                if "[Image:" not in last_chat:
                    return True

            return False

        except Exception:
            return False

    @staticmethod
    def _pick_platform_ctx(star_inst):
        """
        从一个插件实例上取出平台的群聊上下文对象

        依次探测新版属性名 group_chat_context 和旧版属性名 ltm，并校验它确实是
        平台的对象（带 cfg() 和聊天记录容器），避免把其他插件上同名的属性当成它。

        Returns:
            平台的群聊上下文对象，不是则返回 None
        """
        for attr in _PLATFORM_CTX_ATTRS:
            try:
                candidate = getattr(star_inst, attr, None)
            except Exception:
                candidate = None
            if candidate is None:
                continue
            if not callable(getattr(candidate, "cfg", None)):
                continue
            if PlatformLTMHelper._get_records_map(candidate) is None:
                continue
            return candidate
        return None

    @staticmethod
    def _iter_star_instances(context):
        """
        依次产出已加载的插件实例

        来源按可靠程度排列：公开 API context.get_all_stars() → 内核的
        star_registry → 更早版本内核上的若干属性（仅作兜底）。
        """
        # 方法1: 通过 context.get_all_stars() 获取所有插件的 Metadata，
        # 其 star_cls 属性是插件的实际实例
        try:
            if hasattr(context, "get_all_stars"):
                for star_md in context.get_all_stars() or []:
                    star_inst = getattr(star_md, "star_cls", None)
                    if star_inst is not None:
                        yield star_inst
        except Exception as e:
            if DEBUG_MODE:
                logger.info(f"[PlatformLTM] get_all_stars 遍历失败: {e}")

        # 方法2: 直接读取 star_registry（备用方案）。
        # 这是内核内部模块，没有公开替代，因此放在 try 里兜底。
        try:
            from astrbot.core.star.star import star_registry

            for star_md in list(star_registry):
                star_inst = getattr(star_md, "star_cls", None)
                if star_inst is not None:
                    yield star_inst
        except Exception as e:
            if DEBUG_MODE:
                logger.info(f"[PlatformLTM] star_registry 遍历失败: {e}")

        # 方法3~6: 更早版本内核上的属性（兼容旧版本）
        for owner_attr, list_attr in (
            (None, "stars"),
            ("star_manager", "stars"),
            (None, "_stars"),
            ("_star_manager", "star_insts"),
        ):
            try:
                owner = getattr(context, owner_attr, None) if owner_attr else context
                if owner is None:
                    continue
                for star_inst in getattr(owner, list_attr, None) or []:
                    if star_inst is not None:
                        yield star_inst
            except Exception:
                continue

    @staticmethod
    def _get_platform_ltm(context):
        """
        获取平台的群聊上下文对象

        新版内核为内置插件上的 group_chat_context（GroupChatContext），
        旧版内核为 ltm（LongTermMemory）。通过遍历已加载的插件实例查找。
        """
        try:
            for star_inst in PlatformLTMHelper._iter_star_instances(context):
                platform_ctx = PlatformLTMHelper._pick_platform_ctx(star_inst)
                if platform_ctx is not None:
                    if DEBUG_MODE:
                        logger.info(
                            f"[PlatformLTM] 找到平台群聊上下文对象: "
                            f"{type(platform_ctx).__name__}"
                        )
                    return platform_ctx

            return None

        except Exception as e:
            if DEBUG_MODE:
                logger.info(f"[PlatformLTM] 获取 LTM 实例失败: {e}")
            return None

    @staticmethod
    def _verify_message_match(
        chat_record: str,
        sender_name: str,
        original_text: str,
        msg_timestamp: Optional[str] = None,
    ) -> bool:
        """
        验证聊天记录是否匹配当前消息

        平台存储格式: [发送者昵称/时间]: 消息内容

        Args:
            chat_record: 平台存储的聊天记录
            sender_name: 当前消息的发送者昵称
            original_text: 原始消息文本
            msg_timestamp: 消息时间戳（HH:MM:SS），用于精确匹配

        Returns:
            是否匹配
        """
        try:
            # 检查发送者昵称是否在记录开头
            # 格式: [昵称/HH:MM:SS]:
            if not sender_name:
                return False

            # 🔧 如果有时间戳，优先使用精确匹配
            if msg_timestamp:
                expected_prefix = f"[{sender_name}/{msg_timestamp}]"
                if chat_record.startswith(expected_prefix):
                    return True
                # 宽松匹配：允许1秒误差
                match = re.match(
                    rf"^\[{re.escape(sender_name)}/(\d{{2}}:\d{{2}}:\d{{2}})\]",
                    chat_record,
                )
                if match:
                    record_time = match.group(1)
                    if PlatformLTMHelper._timestamps_close(
                        msg_timestamp, record_time, tolerance=1
                    ):
                        return True
                return False

            # 没有时间戳，使用原有的宽松匹配逻辑
            # 使用正则匹配格式 [昵称/时间]:
            pattern = rf"^\[{re.escape(sender_name)}/\d{{2}}:\d{{2}}:\d{{2}}\]:\s*"
            if not re.match(pattern, chat_record):
                # 尝试更宽松的匹配（昵称可能被截断或有特殊字符）
                if f"[{sender_name}" not in chat_record[:50]:
                    return False

            # 如果原始文本不为空，进一步验证内容
            if original_text and len(original_text) > 3:
                # 提取原始文本的前几个字符进行匹配（排除图片标记）
                clean_original = (
                    original_text.replace("[图片]", "").replace("[Image]", "").strip()
                )
                if clean_original and len(clean_original) > 3:
                    # 检查聊天记录中是否包含原始文本的一部分
                    if (
                        clean_original[: min(10, len(clean_original))]
                        not in chat_record
                    ):
                        # 可能是纯图片消息，放宽验证
                        if "[Image:" not in chat_record:
                            return False

            return True

        except Exception as e:
            if DEBUG_MODE:
                logger.info(f"[PlatformLTM] 验证消息匹配时出错: {e}")
            return False

    @staticmethod
    def _extract_message_content(chat_record: str) -> Optional[str]:
        """
        从聊天记录中提取消息内容（去除前缀）

        输入格式: [发送者/时间]: 消息内容 [Image: 描述]
        输出格式: 消息内容 [图片内容: 描述]

        Args:
            chat_record: 平台存储的聊天记录

        Returns:
            提取的消息内容，失败返回 None
        """
        try:
            # 使用正则提取内容部分
            # 格式: [xxx/HH:MM:SS]: 内容
            match = re.match(r"^\[[^\]]+/\d{2}:\d{2}:\d{2}\]:\s*(.*)$", chat_record)
            if match:
                content = match.group(1)
            elif "]: " in chat_record:
                # 备用方案：直接查找 ]: 后的内容
                content = chat_record.split("]: ", 1)[1]
            else:
                return None

            # 新版内核在「@了机器人」的记录开头插入的标记，只服务于平台自己的
            # 提示词；本插件有独立的 @ 判定，带进缓存只会误导 AI，这里去掉
            content = content.replace(_DIRECTED_AT_YOU_MARK, "", 1).strip()
            # 将 [Image: xxx] 转换为 [图片内容: xxx] 以保持与插件格式一致
            content = re.sub(r"\[Image:\s*([^\]]+)\]", r"[图片内容: \1]", content)
            return content if content else None

        except Exception as e:
            if DEBUG_MODE:
                logger.info(f"[PlatformLTM] 提取消息内容时出错: {e}")
            return None

    @staticmethod
    def has_image_in_message(event: AstrMessageEvent) -> bool:
        """
        检查消息中是否包含图片

        会一并下钻 `Reply.chain`：群聊里「引用一张图片」时，
        被引用的图片挂在 Reply 组件内部，只扫顶层组件会漏判。

        Args:
            event: 消息事件

        Returns:
            是否包含图片
        """
        try:
            if not hasattr(event, "message_obj") or not hasattr(
                event.message_obj, "message"
            ):
                return False

            return has_image_in_chain(event.message_obj.message)

        except Exception:
            return False

    @staticmethod
    def is_pure_image_message(event: AstrMessageEvent) -> bool:
        """
        检查是否是纯图片消息（不包含文字）

        引用、视频、语音、文件等非文字段均视为「有内容」，
        避免「引用图片 + @机器人」被误判成纯图片消息。
        会被引消息里的图片同样计入。

        Args:
            event: 消息事件

        Returns:
            是否是纯图片消息
        """
        try:
            if not hasattr(event, "message_obj") or not hasattr(
                event.message_obj, "message"
            ):
                return False

            has_image, has_text, _ = analyze_chain(event.message_obj.message)

            return has_image and not has_text

        except Exception:
            return False
