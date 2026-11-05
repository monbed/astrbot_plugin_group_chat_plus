"""Restore AstrBot's persona and subagent steps for plugin requests.

Replies are requested through ``event.request_llm()`` without a conversation.
AstrBot's ``_ensure_persona_and_skills()`` returns early for such requests, so
it never applies the persona's tool and skill selections, adds the
``transfer_to_*`` handoff tools, removes duplicated tools, or appends the router
prompt. The ``on_llm_request`` hook applies those steps here.

This module deliberately imports nothing from AstrBot so it can be unit tested
without the runtime.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any


def select_persona_skills(
    skills: list[Any],
    workspace_skills: list[Any],
    persona: dict | None,
) -> list[Any]:
    """Apply a persona's skill selection the way AstrBot 4.29.4 does.

    Workspace skills are not limited by the selection. They are only dropped
    when the persona disables skills entirely, and they replace a skill with the
    same name.

    Args:
        skills: Active installed skills.
        workspace_skills: Skills found in the session workspace.
        persona: The selected persona, or ``None`` when no persona applies.

    Returns:
        The skills to list in the system prompt.
    """
    selection = persona.get("skills") if persona else None
    if selection is not None:
        allowed = set(selection)
        skills = [skill for skill in skills if skill.name in allowed]
    if workspace_skills and selection != []:
        by_name = {skill.name: skill for skill in skills}
        for skill in workspace_skills:
            by_name[skill.name] = skill
        skills = [by_name[name] for name in sorted(by_name)]
    return skills


async def collect_request_skills(
    skill_manager: Any,
    provider_settings: dict,
    persona: dict | None,
    filter_for_config: Callable[[list[Any], dict], list[Any]],
    get_workspace_path: Callable[[], Awaitable[Any]],
) -> list[Any]:
    """Collect the skills to list for one request.

    Args:
        skill_manager: AstrBot's skill manager.
        provider_settings: The session's ``provider_settings``.
        persona: The selected persona, or ``None`` when no persona applies.
        filter_for_config: AstrBot's filter for the current configuration.
        get_workspace_path: Returns the session workspace path.

    Returns:
        The skills to list in the system prompt.
    """
    runtime = provider_settings.get("computer_use_runtime", "none")
    skills = skill_manager.list_skills(active_only=True, runtime=runtime)
    skills = filter_for_config(skills, provider_settings)

    workspace_skills: list[Any] = []
    if runtime == "local":
        workspace_skills = skill_manager.list_workspace_skills(
            await get_workspace_path()
        )
    return select_persona_skills(skills, workspace_skills, persona)


def remove_tools_outside_persona(
    tool_set: Any,
    registered_tools: Iterable[Any],
    allowed_tool_names: Iterable[str] | None,
) -> list[str]:
    """Remove registered tools that the main persona did not select.

    Only plugin and MCP tools are affected. AstrBot adds builtin tools according
    to global configuration, and they are not part of ``registered_tools``.

    Args:
        tool_set: The request tool set, changed in place.
        registered_tools: The tool manager's ``func_list``. Do not pass the set
            given to ``request_llm``: AstrBot adds builtin tools to that object.
        allowed_tool_names: The persona's tool names. ``None`` means all tools,
            and an empty value means none.

    Returns:
        Names of the removed tools.
    """
    if allowed_tool_names is None or not hasattr(tool_set, "remove_tool"):
        return []

    allowed = {str(name) for name in allowed_tool_names}
    registered = {tool.name for tool in registered_tools}
    present = {tool.name for tool in tool_set.tools}
    removed = sorted((present & registered) - allowed)
    for name in removed:
        tool_set.remove_tool(name)
    return removed


@dataclass
class SubagentBridgeResult:
    """Outcome of applying subagent orchestration to one request.

    Attributes:
        handoff_names: Handoff tools now available to the main agent.
        removed_tool_names: Duplicate tools removed from the main agent.
        router_prompt_added: Whether the router prompt was appended.
    """

    handoff_names: list[str] = field(default_factory=list)
    removed_tool_names: list[str] = field(default_factory=list)
    router_prompt_added: bool = False


def collect_assigned_tool_names(
    orchestrator_config: dict,
    persona_manager: Any,
    registered_tools: list[Any],
) -> set[str]:
    """Collect tools assigned to enabled subagents.

    This mirrors AstrBot 4.29.4's main-agent logic, including its handling of
    personas set to "all tools": their ``tools`` value is ``None``, the agent's
    own ``tools`` value is used instead, and WebUI-saved agents have none, so
    nothing is treated as a duplicate and the main agent keeps its tools.

    Args:
        orchestrator_config: The ``subagent_orchestrator`` configuration.
        persona_manager: Provides ``get_persona_v3_by_id``.
        registered_tools: Registered tools, used when an agent's tools are None.

    Returns:
        Names of the tools assigned to enabled subagents.
    """
    assigned: set[str] = set()
    agents = orchestrator_config.get("agents", [])
    if not isinstance(agents, list):
        return assigned

    for agent in agents:
        if not isinstance(agent, dict) or agent.get("enabled", True) is False:
            continue

        persona_tools = None
        persona_id = agent.get("persona_id")
        if persona_id:
            persona = persona_manager.get_persona_v3_by_id(persona_id)
            if persona is not None:
                persona_tools = persona.get("tools")

        tools = agent.get("tools", [])
        if persona_tools is not None:
            tools = persona_tools

        if tools is None:
            assigned.update(tool.name for tool in registered_tools)
            continue
        if not isinstance(tools, list):
            continue
        for tool_name in tools:
            name = str(tool_name).strip()
            if name:
                assigned.add(name)
    return assigned


def apply_subagent_orchestration(req: Any, context: Any) -> SubagentBridgeResult:
    """Add handoff tools, remove duplicates, and append the router prompt.

    Every step is idempotent, so the result is unchanged if AstrBot has already
    applied orchestration to the request. Duplicates are kept when there are no
    handoff tools, because the main agent could not delegate that work.

    Args:
        req: Provider request whose ``func_tool`` and ``system_prompt`` change.
        context: AstrBot context holding the configuration and orchestrator.

    Returns:
        What changed. It is empty when orchestration does not apply.
    """
    result = SubagentBridgeResult()

    orchestrator_config = context.get_config().get("subagent_orchestrator", {})
    orchestrator = getattr(context, "subagent_orchestrator", None)
    if not orchestrator_config.get("main_enable", False) or not orchestrator:
        return result

    tool_set = req.func_tool
    handoffs = list(getattr(orchestrator, "handoffs", None) or [])
    if (
        not handoffs
        or not hasattr(tool_set, "add_tool")
        or not hasattr(tool_set, "remove_tool")
    ):
        return result

    for handoff in handoffs:
        tool_set.add_tool(handoff)
    result.handoff_names = [handoff.name for handoff in handoffs]

    if orchestrator_config.get("remove_main_duplicate_tools", False):
        handoff_names = set(result.handoff_names)
        registered_tools = [
            tool
            for tool in context.get_llm_tool_manager().func_list
            if tool.name not in handoff_names
        ]
        assigned = collect_assigned_tool_names(
            orchestrator_config,
            context.persona_manager,
            registered_tools,
        )
        present = {tool.name for tool in tool_set.tools}
        for name in sorted(assigned & present - handoff_names):
            tool_set.remove_tool(name)
            result.removed_tool_names.append(name)

    router_prompt = str(orchestrator_config.get("router_system_prompt") or "").strip()
    system_prompt = req.system_prompt or ""
    if router_prompt and router_prompt not in system_prompt:
        req.system_prompt = f"{system_prompt}\n{router_prompt}\n"
        result.router_prompt_added = True

    return result
