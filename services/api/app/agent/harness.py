import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.context import build_context
from app.agent.goals import resolve_skill_and_goal
from app.agent.guardrails import check_user_input, verify_output
from app.agent.planner import build_plan
from app.agent.provider import provider
from app.agent.skills import parse_skill_output
from app.agent.tools import tool_registry
from app.agent.validators import audit_program_research_run, update_synthesis_steps
from app.core.config import settings
from app.models.entities import AgentRun, AgentStep, Conversation, Message

EventCallback = Callable[[str, Dict[str, Any]], Awaitable[None]]


async def no_op_event(event: str, data: Dict[str, Any]) -> None:
    return None


class AgentHarness:
    async def run(
        self,
        session: AsyncSession,
        message: str,
        conversation_id: Optional[str] = None,
        emit: EventCallback = no_op_event,
    ) -> AgentRun:
        started = time.perf_counter()
        findings = check_user_input(message)
        if findings:
            await emit("guardrail.triggered", {"findings": findings})
            raise ValueError("输入触发安全边界，无法执行")

        conversation = None
        if conversation_id:
            conversation = await session.get(Conversation, conversation_id)
        if not conversation:
            conversation = Conversation(owner_id=settings.local_owner_id, title=message[:80])
            session.add(conversation)
            await session.flush()
        else:
            conversation.updated_at = datetime.now(timezone.utc)
        pending_resource_ids = list(conversation.resource_ids or [])
        attachment_sources = []
        for resource_id in pending_resource_ids:
            if resource_id.startswith("document:"):
                attachment_sources.append({"type": "document", "id": resource_id.removeprefix("document:")})
            elif resource_id.startswith("draft:"):
                attachment_sources.append({"type": "reference_draft", "id": resource_id.removeprefix("draft:")})
        session.add(
            Message(
                conversation_id=conversation.id,
                owner_id=settings.local_owner_id,
                role="user",
                content=message,
                sources=attachment_sources,
            )
        )
        skill, goal_spec = await resolve_skill_and_goal(session, message)
        context = await build_context(session, skill.name, message, conversation)
        context["goal_spec"] = goal_spec
        plan = build_plan(skill, message, goal_spec)
        context["plan"] = plan
        run = AgentRun(
            owner_id=settings.local_owner_id,
            conversation_id=conversation.id,
            skill_name=skill.name,
            skill_version=skill.version,
            goal=message,
            plan=plan,
            context_snapshot=context,
            status="running",
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)
        await emit(
            "run.started",
            {"run_id": run.id, "conversation_id": conversation.id, "skill": skill.name},
        )
        await emit("plan.created", {"run_id": run.id, "plan": plan})

        step_records = []
        for position, item in enumerate(plan):
            step = AgentStep(
                owner_id=settings.local_owner_id,
                run_id=run.id,
                position=position,
                name=item["name"],
                expected_output=item["expected_output"],
                checkpoint={
                    "plan_id": item["id"],
                    "step_type": item.get("step_type", "generic"),
                    "allowed_tools": item.get("allowed_tools", []),
                    "acceptance_criteria": item.get("acceptance_criteria", []),
                },
            )
            session.add(step)
            step_records.append(step)
        await session.commit()

        try:
            if skill.name == "program-research" and step_records:
                step_records[0].status = "accepted"
                step_records[0].result = {
                    "status": "accepted",
                    "goal_spec": goal_spec,
                }
                await session.commit()
                await emit("step.completed", {"run_id": run.id, "step": step_records[0].name})
            elif step_records:
                step_records[0].status = "running"
                await session.commit()
                await emit("step.started", {"run_id": run.id, "step": step_records[0].name})
            final_output, usage = await provider.run(
                session=session,
                run_id=run.id,
                message=message,
                context=context,
                skill=skill,
                tools=tool_registry,
                emit=emit,
                goal_spec=goal_spec,
            )
            structured_output = parse_skill_output(skill, final_output)
            audit = None
            if skill.name == "program-research":
                structured_output, audit = await audit_program_research_run(
                    session, run.id, structured_output, goal_spec
                )
                step_records = await update_synthesis_steps(session, run.id, audit)
            display_output = structured_output["summary"]
            # Guard the entire structured result. Unsafe text can otherwise hide in a
            # nested recommendation while the user-facing summary remains harmless.
            output_findings = verify_output(
                json.dumps(structured_output, ensure_ascii=False)
            )
            if output_findings:
                await emit("guardrail.triggered", {"findings": output_findings})
                display_output = "模型输出未通过安全检查，请换一种方式描述任务。"
                run.stop_reason = "output_guardrail"
            else:
                run.stop_reason = (
                    "success"
                    if not audit or audit.get("release")
                    else "verification_incomplete"
                )
            if skill.name != "program-research":
                for step in step_records:
                    step.status = "completed"
                    step.result = {"status": "verified", "skill": skill.name}
                    step.checkpoint = {"position": step.position, "completed": True}
                    await emit("step.completed", {"run_id": run.id, "step": step.name})
            else:
                await session.flush()
                for step in step_records:
                    event = "step.completed" if step.status == "accepted" else "step.blocked"
                    if step.status in {"accepted", "blocked", "rejected", "failed"}:
                        await emit(event, {"run_id": run.id, "step": step.name, "status": step.status})
            statuses = {step.position: step.status for step in step_records}
            run.plan = [
                {**item, "status": statuses.get(index, item.get("status", "pending"))}
                for index, item in enumerate(run.plan or [])
            ]
            run.status = "completed"
            run.final_output = display_output
            run.structured_output = structured_output
            run.token_usage = usage
            run.duration_ms = int((time.perf_counter() - started) * 1000)
            conversation.resource_ids = []
            session.add(
                Message(
                    conversation_id=conversation.id,
                    owner_id=settings.local_owner_id,
                    role="assistant",
                    content=display_output,
                )
            )
            await session.commit()
            await emit(
                "message.completed",
                {"run_id": run.id, "content": display_output, "structured_output": structured_output},
            )
            await emit(
                "run.completed",
                {"run_id": run.id, "duration_ms": run.duration_ms, "usage": usage},
            )
            return run
        except asyncio.CancelledError:
            for step in step_records:
                if step.status == "running":
                    step.status = "cancelled"
                    step.result = {"status": "cancelled"}
            run.status = "cancelled"
            run.stop_reason = "user_cancelled"
            run.final_output = ""
            run.duration_ms = int((time.perf_counter() - started) * 1000)
            await session.commit()
            await emit("run.cancelled", {"run_id": run.id})
            raise
        except Exception as exc:
            for step in step_records:
                if step.status == "running":
                    step.status = "failed"
                    step.result = {"status": "failed", "error": str(exc)[:300]}
            run.status = "failed"
            run.stop_reason = "error"
            run.final_output = str(exc)[:1000]
            run.duration_ms = int((time.perf_counter() - started) * 1000)
            await session.commit()
            await emit("run.failed", {"run_id": run.id, "error": str(exc)})
            raise


harness = AgentHarness()
