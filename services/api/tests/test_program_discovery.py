from types import SimpleNamespace

from sqlalchemy import select

from app.agent.program_recommendation_workflow import (
    recommendation_goal_spec,
    run_online_program_recommendation,
)
from app.agent.tools import tool_registry
from app.agent.validators import validate_program_search_result
from app.core.database import SessionLocal
from app.models.entities import Program, ToolCall
from app.services import program_discovery
from app.services.business import get_or_create_recommendation_goal
from app.services.program_discovery import (
    _page_supports_identity,
    _school_matches,
    _trusted_qs_rank,
    discover_official_programs,
    enrich_candidates_from_catalogs,
    official_domain_allowed,
    search_official_program_candidates,
)


def test_direct_search_identity_and_official_domains():
    url = "https://www.mscai.hku.hk/"
    page = SimpleNamespace(
        url=url,
        title="The University of Hong Kong - Master of Science in Artificial Intelligence",
        text="The University of Hong Kong Master of Science in Artificial Intelligence",
    )
    assert _page_supports_identity(
        page,
        {
            "university": "The University of Hong Kong",
            "name": "Master of Science in Artificial Intelligence",
            "degree": "master",
        },
    )
    assert official_domain_allowed(url, "Hong Kong")
    assert official_domain_allowed(
        "https://www.comp.nus.edu.sg/programmes/msc-cs/", "Singapore"
    )
    assert not official_domain_allowed("https://example-ranking.com/nus-cs", "Singapore")
    assert _school_matches(
        "Nanyang Technological University, Singapore",
        "Nanyang Technological University",
    )
    assert not _school_matches(
        "The University of Hong Kong", "The Chinese University of Hong Kong"
    )
    assert _trusted_qs_rank("The Chinese University of Hong Kong") == 32

def test_goal_spec_keeps_degree_qs_and_school_priority_over_country():
    goal = SimpleNamespace(
        target_fields=["Artificial Intelligence"],
        target_university="Imperial College London",
        target_degree_level="doctoral",
        target_countries=["United States"],
        max_qs_rank=50,
    )
    spec = recommendation_goal_spec(goal, {"shown-program"}, 5)
    assert spec["entities"]["university"] == "Imperial College London"
    assert spec["entities"]["degree_level"] == "doctoral"
    assert spec["entities"]["countries"] == []
    assert spec["parameters"]["max_qs_rank"] == 50
    assert spec["parameters"]["excluded_program_ids"] == ["shown-program"]


def test_goal_spec_preserves_recognized_and_freeform_fields_individually():
    goal = SimpleNamespace(
        target_fields=["人工智能", "计算机科学", "国际教育", "国际教育"],
        target_university="National University of Singapore (NUS)",
        target_degree_level="master",
        target_countries=["Singapore"],
        max_qs_rank=100,
    )
    spec = recommendation_goal_spec(goal, set(), 5)
    assert spec["entities"]["fields"] == ["人工智能", "计算机科学", "国际教育"]
    assert spec["parameters"]["canonical_fields"] == ["computer science"]


def test_program_validator_accepts_custom_field_and_rejects_unrelated_field():
    goal = SimpleNamespace(
        target_fields=["人工智能", "计算机科学", "国际教育"],
        target_university="National University of Singapore (NUS)",
        target_degree_level="master",
        target_countries=["Singapore"],
        max_qs_rank=100,
    )
    spec = recommendation_goal_spec(goal, set(), 5)
    base = {
        "university": "National University of Singapore",
        "degree": "Master",
        "country": "Singapore",
        "qs_rank": 8,
    }
    accepted = validate_program_search_result(
        spec, [{**base, "id": "education", "field": "国际教育"}]
    )
    rejected = validate_program_search_result(
        spec, [{**base, "id": "chemistry", "field": "Chemistry"}]
    )
    assert accepted["verdict"] == "accepted"
    assert rejected["verdict"] == "rejected"
    assert rejected["rejected"][0]["reasons"] == ["wrong_field"]


async def test_direct_online_search_combines_fields_per_university(monkeypatch):
    prompts = []

    class FakeCompletions:
        async def create(self, **kwargs):
            prompt = kwargs["messages"][-1]["content"]
            prompts.append(prompt)
            if "当前步骤只定位" in prompt:
                payload = {
                    "university": "The University of Hong Kong",
                    "country": "Hong Kong",
                    "city": "Hong Kong",
                    "qs_rank": 11,
                    "field_aliases": {
                        "人工智能": ["artificial intelligence"],
                        "计算机科学": ["computer science"],
                    },
                    "catalogs": {
                        "university": [
                            {"url": "https://www.hku.hk/programmes", "title": "Programmes"}
                        ],
                        "faculty": [],
                    },
                }
            else:
                payload = {
                    "programs": [
                        {
                            "url": "https://www.hku.hk/mscai",
                            "name": "Master of Science in Artificial Intelligence",
                        },
                        {
                            "url": "https://www.hku.hk/msccs",
                            "name": "Master of Science in Computer Science",
                        },
                    ],
                    "next_catalog_urls": [],
                }
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=__import__("json").dumps(payload))
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(program_discovery, "AsyncOpenAI", FakeOpenAI)
    monkeypatch.setattr(program_discovery.settings, "dashscope_api_key", "test-key")

    async def fake_fetch(url):
        return SimpleNamespace(
            url=url,
            title="Programmes",
            links=[
                {
                    "url": "https://www.hku.hk/mscai",
                    "label": "Master of Science in Artificial Intelligence",
                },
                {
                    "url": "https://www.hku.hk/msccs",
                    "label": "Master of Science in Computer Science",
                },
            ],
        )

    monkeypatch.setattr(program_discovery, "fetch_page", fake_fetch)
    result = await search_official_program_candidates(
        countries=["Hong Kong"],
        fields=["人工智能", "计算机科学"],
        degree_level="master",
        target_university="The University of Hong Kong",
        max_qs_rank=100,
        excluded_urls=set(),
        max_candidates=5,
    )
    assert len(prompts) == 2
    assert [item["field"] for item in result] == ["人工智能", "计算机科学"]


def test_program_identity_dedup_requires_complete_normalized_name():
    assert program_discovery._program_names_match(
        "MSc in Artificial Intelligence",
        "Master of Science in Artificial Intelligence",
    )
    assert not program_discovery._program_names_match(
        "MSc in Data Science",
        "MSc in Advanced Studies in Statistics and Data Science",
    )


def test_master_identity_rejects_undergraduate_news_page():
    page = SimpleNamespace(
        url="https://example.edu/news/new-beng-artificial-intelligence-program",
        title="New BEng in Artificial Intelligence Program | News",
        text="The university also offers master programmes elsewhere.",
    )
    assert not program_discovery._page_supports_identity(
        page,
        {
            "university": "Example University",
            "name": "BEng in Artificial Intelligence",
            "degree": "master",
        },
    )


async def test_two_level_catalog_adds_project_and_preserves_directory_links(monkeypatch):
    university_catalog = "https://www.example.edu/graduate-programmes"
    faculty_catalog = "https://www.example.edu/computing/graduate-programmes"
    direct_url = "https://www.example.edu/programs/master-computer-science"
    catalog_project_url = "https://www.example.edu/programs/master-software-engineering"
    pages = {
        university_catalog: SimpleNamespace(
            url=university_catalog,
            title="Graduate Programmes - Example University",
            links=[
                {
                    "url": faculty_catalog,
                    "label": "School of Computing Graduate Programmes",
                }
            ],
        ),
        faculty_catalog: SimpleNamespace(
            url=faculty_catalog,
            title="Computing Graduate Programmes",
            links=[
                {
                    "url": catalog_project_url,
                    "label": "Master of Software Engineering",
                }
            ],
        ),
    }

    async def fake_fetch(url):
        return pages[program_discovery.normalize_program_url(url)]

    monkeypatch.setattr(program_discovery, "fetch_page", fake_fetch)
    candidates, report = await enrich_candidates_from_catalogs(
        [
            {
                "university": "Example University",
                "name": "Master of Computer Science",
                "degree": "master",
                "country": "United States",
                "city": "Example City",
                "field": "计算机科学",
                "official_url": direct_url,
                "qs_rank": 80,
                "_university_catalog_url": university_catalog,
                "_faculty_catalog_url": "",
                "_field_aliases": ["computer science", "computing", "software engineering"],
            }
        ],
        fields=["计算机科学"],
        degree_level="master",
        max_candidates=5,
    )

    assert {item["official_url"] for item in candidates} == {
        direct_url,
        catalog_project_url,
    }
    assert candidates[0]["_university_catalog_url"] == university_catalog
    assert candidates[0]["_faculty_catalog_url"] == faculty_catalog
    assert report["mode"] == "direct_plus_two_level_catalog"
    assert report["catalog_candidate_count"] == 1


async def test_catalog_failure_never_removes_direct_candidate(monkeypatch):
    async def failed_fetch(_url):
        raise TimeoutError("catalog unavailable")

    monkeypatch.setattr(program_discovery, "fetch_page", failed_fetch)
    direct = {
        "university": "Example University",
        "name": "Master of Computer Science",
        "degree": "master",
        "country": "United States",
        "field": "计算机科学",
        "official_url": "https://www.example.edu/programs/master-computer-science",
        "_university_catalog_url": "https://www.example.edu/graduate-programmes",
    }
    candidates, report = await enrich_candidates_from_catalogs(
        [direct], fields=["计算机科学"], degree_level="master", max_candidates=5
    )

    assert len(candidates) == 1
    assert candidates[0]["official_url"] == direct["official_url"]
    assert candidates[0]["_university_catalog_url"] == ""
    assert report["errors"][0]["reason"] == "TimeoutError"


async def test_discovery_uses_live_candidates_and_excludes_displayed_url(client, monkeypatch):
    seen = {}

    async def fake_search(**kwargs):
        seen.update(kwargs)
        return [
            {
                "university": "National University of Singapore",
                "name": "Master of Computing in Computer Science",
                "degree": "Master",
                "country": "Singapore",
                "city": "Singapore",
                "field": "Computer Science",
                "official_url": "https://www.comp.nus.edu.sg/programmes/msc-cs/",
                "qs_rank": 8,
            }
        ]

    async def fake_fetch(url):
        return SimpleNamespace(
            url=url,
            title="Master of Computing in Computer Science - NUS",
            text="National University of Singapore Master of Computing Computer Science",
        )

    monkeypatch.setattr(program_discovery, "search_official_program_candidates", fake_search)
    monkeypatch.setattr(program_discovery, "fetch_page", fake_fetch)

    async def fake_identity_audit(_page, _candidate):
        return {"verdict": "accepted", "reason_code": "program_page"}

    monkeypatch.setattr(
        program_discovery,
        "verify_program_identity_independently",
        fake_identity_audit,
    )

    async with SessionLocal() as session:
        first = await discover_official_programs(
            session,
            countries=["Singapore", "Hong Kong"],
            fields=["Computer Science"],
            max_qs_rank=100,
        )
        assert len(first) == 1
        assert first[0].country == "Singapore"
        assert first[0].qs_rank == 8

        second = await discover_official_programs(
            session,
            countries=["Singapore", "Hong Kong"],
            fields=["Computer Science"],
            excluded_program_ids={first[0].id},
            max_qs_rank=100,
        )
        assert second == []
        assert seen["degree_level"] == "master"
        assert first[0].official_url in seen["excluded_urls"]


async def test_recommendation_workflow_uses_discovery_and_independent_audit(client, monkeypatch):
    async with SessionLocal() as session:
        program = await session.scalar(
            select(Program).where(Program.university == "Rice University")
        )
        assert program is not None
        program.qs_rank = 100
        program.qs_ranking_year = 2026
        await session.commit()
        program_id = program.id
        program_snapshot = {
            "id": program.id,
            "university": program.university,
            "name": program.name,
            "degree": program.degree,
            "country": program.country,
            "field": program.field,
            "qs_rank": program.qs_rank,
            "official_url": program.official_url,
        }

        async def fake_discovery(_session, arguments):
            if arguments["excluded_program_ids"]:
                assert arguments["excluded_program_ids"] == [program_id]
                return []
            return [program_snapshot]

        async def fake_verify(_session, arguments):
            assert arguments["program_id"] == program_id
            return {
                "program_id": program_id,
                "status": "verified",
                "official_url": program_snapshot["official_url"],
                "evidence": [{"field": "materials", "quote": "Submit a statement."}],
                "source_results": [{"url": program_snapshot["official_url"]}],
            }

        monkeypatch.setattr(
            tool_registry.tools["discover_official_programs"], "handler", fake_discovery
        )
        monkeypatch.setattr(
            tool_registry.tools["verify_program_official"], "handler", fake_verify
        )
        goal = await get_or_create_recommendation_goal(session)
        goal.target_countries = ["United States"]
        goal.target_fields = ["Computer Science"]
        goal.target_university = "Rice University"
        goal.target_degree_level = "master"
        goal.max_qs_rank = 100
        await session.commit()

        programs, run = await run_online_program_recommendation(
            session, goal, excluded_program_ids=set(), limit=5
        )
        assert [item.id for item in programs] == [program_id]
        assert run.stop_reason == "success"
        calls = list(
            (
                await session.scalars(
                    select(ToolCall)
                    .where(ToolCall.run_id == run.id)
                    .order_by(ToolCall.created_at)
                )
            ).all()
        )
        assert [item.tool_name for item in calls] == [
            "discover_official_programs",
            "verify_program_official",
            "discover_official_programs",
            "discover_official_programs",
        ]
        assert run.structured_output["verification_report"]["released_count"] == 1
