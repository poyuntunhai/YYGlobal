import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.agent.provider import (
    CandidateFact,
    DocumentExtraction,
    OpenAIResponsesProvider,
)
from app.agent.skills import skill_registry
from app.agent.tools import ToolRegistry, ToolSpec, object_schema, tool_registry
from app.core.config import settings
from app.core.database import SessionLocal
from app.models.entities import AgentRun, ToolCall


class FakeResponses:
    def __init__(self, final_output=None):
        self.create_calls = []
        self.parse_calls = []
        self.final_output = final_output or {
            "programs": [], "sources": [],
            "unverified": ["正式申请前请重新核验官网"],
            "summary": "已找到有官方来源的测试项目；正式申请前请重新核验官网。",
        }

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if len(self.create_calls) == 1:
            call = SimpleNamespace(
                type="function_call",
                name="discover_official_programs",
                arguments=json.dumps({
                    "countries": ["United States"],
                    "fields": ["Computer Science"],
                    "degree_level": "master",
                    "target_university": "",
                    "max_qs_rank": 100,
                    "excluded_program_ids": [],
                    "max_candidates": 5,
                }),
                call_id="call-test-1",
            )
            return SimpleNamespace(
                output=[call],
                output_text="",
                usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
            )
        return SimpleNamespace(
            output=[SimpleNamespace(type="message")],
            output_text=json.dumps(self.final_output, ensure_ascii=False),
            usage=SimpleNamespace(input_tokens=20, output_tokens=10, total_tokens=30),
        )

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        return SimpleNamespace(
            output_parsed=DocumentExtraction(
                document_type="transcript",
                summary="成绩单候选信息",
                candidate_facts=[
                    CandidateFact(
                        field="gpa", value="3.7/4.0", evidence="GPA 3.7/4.0", confidence=0.98
                    )
                ],
                requires_confirmation=True,
            )
        )


async def test_responses_function_call_loop_records_trace(monkeypatch):
    fake = FakeResponses()
    instance = OpenAIResponsesProvider()
    instance.client = SimpleNamespace(responses=fake)
    skill = skill_registry.get("program-research")
    async def fake_discovery(_session, _arguments):
        return [{"id": "test-program", "university": "Test University"}]

    monkeypatch.setattr(
        tool_registry.tools["discover_official_programs"], "handler", fake_discovery
    )
    async with SessionLocal() as session:
        run = AgentRun(
            owner_id=settings.local_owner_id,
            skill_name=skill.name,
            skill_version=skill.version,
            goal="搜索计算机项目",
        )
        session.add(run)
        await session.commit()
        output, usage = await instance.run(
            session,
            run.id,
            "搜索计算机项目",
            {"profile": {}, "catalog": {"program_count": 20}},
            skill,
            tool_registry,
        )
        traces = list(
            (await session.scalars(select(ToolCall).where(ToolCall.run_id == run.id))).all()
        )
    assert "没有从项目目录" in json.loads(output)["summary"]
    assert usage["total_tokens"] == 45
    assert len(fake.create_calls) == 2
    assert traces[0].tool_name == "discover_official_programs"
    assert traces[0].status == "completed"


async def test_write_tools_require_explicit_human_confirmation():
    skill = skill_registry.get("cv-planner")
    cv_output = {
        "selected_experiences": [], "order": [], "focus": "真实经历",
        "gaps": ["缺少经历"], "summary": "仅使用真实经历规划 CV。",
    }
    without_confirmation = FakeResponses(cv_output)
    instance = OpenAIResponsesProvider()
    instance.client = SimpleNamespace(responses=without_confirmation)
    async with SessionLocal() as session:
        run = AgentRun(
            owner_id=settings.local_owner_id,
            skill_name=skill.name,
            skill_version=skill.version,
            goal="帮我规划材料",
        )
        session.add(run)
        await session.commit()
        await instance.run(session, run.id, "帮我规划材料", {}, skill, tool_registry)
    exposed = {item["name"] for item in without_confirmation.create_calls[0]["tools"]}
    assert "build_material_plan" not in exposed

    with_confirmation = FakeResponses(cv_output)
    instance.client = SimpleNamespace(responses=with_confirmation)
    async with SessionLocal() as session:
        run = AgentRun(
            owner_id=settings.local_owner_id,
            skill_name=skill.name,
            skill_version=skill.version,
            goal="确认生成材料方案",
        )
        session.add(run)
        await session.commit()
        await instance.run(session, run.id, "确认生成材料方案", {}, skill, tool_registry)
    exposed = {item["name"] for item in with_confirmation.create_calls[0]["tools"]}
    assert "build_material_plan" in exposed

    profile_skill = skill_registry.get("applicant-profile")
    profile_output = {"known": {}, "missing": [], "questions": [], "summary": "画像已读取。"}
    profile_fake = FakeResponses(profile_output)
    instance.client = SimpleNamespace(responses=profile_fake)
    async with SessionLocal() as session:
        run = AgentRun(
            owner_id=settings.local_owner_id,
            skill_name=profile_skill.name,
            skill_version=profile_skill.version,
            goal="更新我的画像",
        )
        session.add(run)
        await session.commit()
        await instance.run(session, run.id, "更新我的画像", {}, profile_skill, tool_registry)
    exposed = {item["name"] for item in profile_fake.create_calls[0]["tools"]}
    assert "save_applicant_profile" not in exposed

    profile_fake = FakeResponses(profile_output)
    instance.client = SimpleNamespace(responses=profile_fake)
    async with SessionLocal() as session:
        run = AgentRun(
            owner_id=settings.local_owner_id,
            skill_name=profile_skill.name,
            skill_version=profile_skill.version,
            goal="确认保存我的画像",
        )
        session.add(run)
        await session.commit()
        await instance.run(
            session, run.id, "确认保存我的画像", {}, profile_skill, tool_registry
        )
    exposed = {item["name"] for item in profile_fake.create_calls[0]["tools"]}
    assert "save_applicant_profile" in exposed


async def test_multimodal_structured_extraction(tmp_path):
    fake = FakeResponses()
    instance = OpenAIResponsesProvider()
    instance.client = SimpleNamespace(responses=fake)
    image = tmp_path / "transcript.png"
    image.write_bytes(b"test-image-bytes")
    result = await instance.extract_document(image, "image/png", "transcript", "")
    assert result["document_type"] == "transcript"
    assert result["candidate_facts"][0]["field"] == "gpa"
    content = fake.parse_calls[0]["input"][0]["content"]
    assert any(item["type"] == "input_image" for item in content)
    assert fake.parse_calls[0]["text_format"] is DocumentExtraction


class FailingThenRecoveringResponses:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            call = SimpleNamespace(
                type="function_call",
                name="empty_search",
                arguments="{}",
                call_id="call-empty",
            )
            return SimpleNamespace(
                output=[call],
                output_text="",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
            )
        return SimpleNamespace(
            output=[SimpleNamespace(type="message")],
            output_text=json.dumps({"summary": "工具无结果，已调整计划并明确标记待确认。"}),
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        )


async def test_tool_failure_emits_replan_and_preserves_run():
    async def empty_handler(session, arguments):
        return []

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="empty_search",
            description="Return no results for bounded replanning test.",
            parameters=object_schema({}, []),
            handler=empty_handler,
        )
    )
    skill = SimpleNamespace(
        name="test-skill",
        version="0.1.0",
        instructions="",
        prompt="",
        tools=["empty_search"],
        approval_required=[],
        output_schema={
            "type": "object", "properties": {"summary": {"type": "string"}},
            "required": ["summary"], "additionalProperties": False,
        },
    )
    fake = FailingThenRecoveringResponses()
    instance = OpenAIResponsesProvider()
    instance.client = SimpleNamespace(responses=fake)
    events = []

    async def emit(name, payload):
        events.append((name, payload))

    async with SessionLocal() as session:
        run = AgentRun(
            owner_id=settings.local_owner_id,
            skill_name="test-skill",
            skill_version="0.1.0",
            goal="test recovery",
            plan=[
                {
                    "id": "step-1",
                    "name": "search",
                    "status": "completed",
                    "dependencies": [],
                    "expected_output": "result",
                }
            ],
        )
        session.add(run)
        await session.commit()
        output, _ = await instance.run(
            session, run.id, "test recovery", {}, skill, registry, emit=emit
        )
        await session.refresh(run)
    assert "调整计划" in json.loads(output)["summary"]
    assert any(name == "tool.error" for name, _ in events)
    assert any(name == "plan.updated" for name, _ in events)
    assert run.plan[-1]["recovery"] is True
    assert run.plan[-1]["status"] == "pending"


async def test_tool_registry_validates_schema_and_permission():
    async with SessionLocal() as session:
        with pytest.raises(PermissionError):
            await tool_registry.execute(session, "run", "search_programs", {}, [])
        with pytest.raises(ValueError, match="缺少必填"):
            await tool_registry.execute(
                session, "run", "search_programs", {"query": "CS"}, ["search_programs"]
            )
        with pytest.raises(PermissionError, match="明确确认"):
            await tool_registry.execute(
                session,
                "run",
                "save_applicant_profile",
                {},
                ["save_applicant_profile"],
            )
