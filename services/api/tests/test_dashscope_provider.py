import json
from types import SimpleNamespace

from sqlalchemy import select

from app.agent.provider import DashScopeChatProvider, ProviderRouter
from app.agent.skills import skill_registry
from app.agent.tools import tool_registry
from app.core.config import settings
from app.core.database import SessionLocal
from app.models.entities import AgentRun, ToolCall


class FakeDashScopeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            message = SimpleNamespace(
                role="assistant",
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        id="call-qwen-1",
                        type="function",
                        function=SimpleNamespace(
                            name="discover_official_programs",
                            arguments=json.dumps(
                                {
                                    "countries": ["United States"],
                                    "fields": ["Computer Science"],
                                    "degree_level": "master",
                                    "target_university": "",
                                    "max_qs_rank": 100,
                                    "excluded_program_ids": [],
                                    "max_candidates": 5,
                                }
                            ),
                        ),
                    )
                ],
            )
        else:
            payload = {
                "programs": [], "sources": [],
                "unverified": ["正式申请前请重新核验官网"],
                "summary": "已找到有官方来源的测试项目；正式申请前请重新核验官网。",
            }
            message = SimpleNamespace(
                role="assistant",
                content=json.dumps(payload, ensure_ascii=False),
                tool_calls=None,
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


async def test_dashscope_function_call_loop_records_trace(monkeypatch):
    fake = FakeDashScopeCompletions()
    instance = DashScopeChatProvider()
    instance.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
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
    assert usage == {
        "input_tokens": 20,
        "output_tokens": 10,
        "total_tokens": 30,
        "mode": "dashscope",
    }
    assert len(fake.calls) == 2
    assert fake.calls[0]["model"] == settings.dashscope_reasoning_model
    assert fake.calls[0]["tools"][0]["type"] == "function"
    assert fake.calls[0]["tools"][0]["function"]["parameters"]["type"] == "object"
    assert any(message.get("role") == "tool" for message in fake.calls[1]["messages"] if isinstance(message, dict))
    assert traces[0].tool_name == "discover_official_programs"
    assert traces[0].status == "completed"


class FakeDashScopeExtraction:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = {
            "document_type": "transcript",
            "summary": "成绩单候选信息",
            "candidate_facts": [
                {
                    "field": "gpa",
                    "value": "3.7/4.0",
                    "evidence": "GPA 3.7/4.0",
                    "confidence": 0.98,
                }
            ],
            "requires_confirmation": True,
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
        )


async def test_dashscope_multimodal_json_extraction(tmp_path):
    fake = FakeDashScopeExtraction()
    instance = DashScopeChatProvider()
    instance.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    image = tmp_path / "transcript.png"
    image.write_bytes(b"test-image-bytes")

    result = await instance.extract_document(image, "image/png", "transcript", "")

    assert result["candidate_facts"][0]["field"] == "gpa"
    assert fake.calls[0]["model"] == settings.dashscope_extraction_model
    assert fake.calls[0]["response_format"] == {"type": "json_object"}
    content = fake.calls[0]["messages"][0]["content"]
    assert any(item["type"] == "image_url" for item in content)


def test_provider_router_auto_prefers_dashscope(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "auto")
    router = ProviderRouter()
    router.openai.client = object()
    router.dashscope.client = object()
    assert router.mode == "dashscope"

    router.dashscope.client = None
    assert router.mode == "openai"

    router.openai.client = None
    assert router.mode == "local-fallback"
