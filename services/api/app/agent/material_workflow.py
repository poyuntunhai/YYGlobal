from typing import Any, Dict, List, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.planner import build_material_writing_plan
from app.core.config import settings
from app.models.entities import AgentRun, AgentStep, Conversation


async def start_material_run(
    session: AsyncSession,
    conversation: Conversation,
    goal: str,
    goal_spec: Dict[str, Any],
) -> Tuple[AgentRun, List[AgentStep]]:
    plan = build_material_writing_plan(goal_spec)
    run = AgentRun(
        owner_id=settings.local_owner_id,
        conversation_id=conversation.id,
        skill_name="material-writing",
        skill_version="0.1.0",
        goal=goal,
        plan=plan,
        context_snapshot={"goal_spec": goal_spec},
        status="running",
    )
    session.add(run)
    await session.flush()
    steps = []
    for position, item in enumerate(plan):
        step = AgentStep(
            owner_id=settings.local_owner_id,
            run_id=run.id,
            position=position,
            name=item["name"],
            expected_output=item["expected_output"],
            checkpoint={
                "plan_id": item["id"],
                "step_type": item["step_type"],
                "acceptance_criteria": item["acceptance_criteria"],
            },
        )
        session.add(step)
        steps.append(step)
    await session.flush()
    return run, steps


def set_material_step(
    steps: List[AgentStep],
    step_type: str,
    status: str,
    result: Dict[str, Any],
) -> None:
    step = next(
        item
        for item in steps
        if (item.checkpoint or {}).get("step_type") == step_type
    )
    step.status = status
    step.result = result
    step.checkpoint = {**(step.checkpoint or {}), "last_status": status}


def sync_material_plan(run: AgentRun, steps: List[AgentStep]) -> None:
    statuses = {item.position: item.status for item in steps}
    run.plan = [
        {**item, "status": statuses.get(index, item.get("status", "pending"))}
        for index, item in enumerate(run.plan or [])
    ]
