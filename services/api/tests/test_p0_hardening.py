import json
from types import SimpleNamespace

import pytest

from app.agent.provider import DashScopeChatProvider
from app.agent.skills import local_skill_output, parse_skill_output, skill_registry
from app.core.database import SessionLocal
from app.schemas.api import ProfileUpdate, RecommendationGoalUpdate
from app.services.business import (
    canonical_countries,
    search_programs_for_profile,
    update_profile,
    update_recommendation_goal,
)
from app.services.requirements import (
    extract_requirement_candidates,
    extract_source_requirements,
    merge_ai_extraction,
    merge_source_extractions,
)


def test_country_aliases_are_canonicalized():
    assert canonical_countries(["英国", "UK", "United Kingdom"]) == ["United Kingdom"]
    assert canonical_countries(["美国", "USA"]) == ["United States"]
    assert canonical_countries(
        ["Hong Kong SAR", "Hong Kong SAR, China", "中国香港"]
    ) == ["Hong Kong"]


async def test_catalog_is_driven_by_profile_target_field():
    async with SessionLocal() as session:
        await update_profile(
            session,
            ProfileUpdate(
                current_major="Business Administration",
                target_countries=["United States"],
                target_fields=["商科"],
                confirmed=True,
            ),
        )
        await update_recommendation_goal(
            session,
            RecommendationGoalUpdate(
                target_countries=["United States"],
                target_fields=["商科"],
                max_qs_rank=100,
            ),
        )
        business = await search_programs_for_profile(session)
        assert business
        assert {item.field for item in business} <= {"Business Analytics", "Finance", "Accounting"}
        assert all("master" in item.name.lower() for item in business)

        await update_profile(
            session,
            ProfileUpdate(
                current_major="Computer Science",
                target_countries=["United States"],
                target_fields=["计算机"],
                confirmed=True,
            ),
        )
        await update_recommendation_goal(
            session,
            RecommendationGoalUpdate(
                target_countries=["United States"],
                target_fields=["计算机"],
                max_qs_rank=100,
            ),
        )
        computer = await search_programs_for_profile(session)
        assert computer
        assert {item.field for item in computer} == {"Computer Science"}
        assert all(item.qs_rank is not None and item.qs_rank <= 100 for item in computer)
        assert all("cs" in item.official_url.lower() or "computer" in item.official_url.lower() for item in computer)

        for target, expected in [
            ("商业分析", {"Business Analytics"}),
            ("金融", {"Finance"}),
            ("会计", {"Accounting"}),
            ("公共政策", {"Public Policy"}),
        ]:
            await update_profile(
                session,
                ProfileUpdate(
                    target_countries=["United States"], target_fields=[target], confirmed=True
                ),
            )
            await update_recommendation_goal(
                session,
                RecommendationGoalUpdate(
                    target_countries=["United States"],
                    target_fields=[target],
                    max_qs_rank=100,
                ),
            )
            matched = await search_programs_for_profile(session)
            assert matched, target
            assert {item.field for item in matched} == expected
            assert all(item.official_url.startswith("https://") for item in matched)


async def test_recommendation_goal_supports_multiple_countries_fields_and_rank_limit():
    async with SessionLocal() as session:
        await update_profile(session, ProfileUpdate(confirmed=True))
        await update_recommendation_goal(
            session,
            RecommendationGoalUpdate(
                target_countries=["英国", "美国"],
                target_fields=["Computer Science", "Public Policy"],
                max_qs_rank=50,
            ),
        )
        matched = await search_programs_for_profile(session)
        assert matched
        assert {item.country for item in matched} == {"United States", "United Kingdom"}
        assert {item.field for item in matched} == {"Computer Science", "Public Policy"}
        assert all(item.qs_rank is not None and item.qs_rank <= 50 for item in matched)


async def test_selected_university_overrides_country_but_keeps_other_filters():
    async with SessionLocal() as session:
        await update_profile(session, ProfileUpdate(confirmed=True))
        await update_recommendation_goal(
            session,
            RecommendationGoalUpdate(
                target_countries=["United States"],
                target_fields=["Computer Science"],
                target_university="Imperial College London",
                max_qs_rank=100,
            ),
        )
        matched = await search_programs_for_profile(session)
        assert matched
        assert {item.university for item in matched} == {"Imperial College London"}
        assert {item.country for item in matched} == {"United Kingdom"}
        assert {item.field for item in matched} == {"Computer Science"}
        assert all(item.qs_rank is not None and item.qs_rank <= 100 for item in matched)


async def test_unknown_custom_field_does_not_match_every_program():
    async with SessionLocal() as session:
        await update_profile(session, ProfileUpdate(confirmed=True))
        await update_recommendation_goal(
            session,
            RecommendationGoalUpdate(
                target_countries=["United States"],
                target_fields=["Marine Biology"],
                max_qs_rank=100,
            ),
        )
        assert await search_programs_for_profile(session) == []
        await update_recommendation_goal(
            session,
            RecommendationGoalUpdate(
                target_countries=["United States"],
                target_fields=["Computer Science"],
                max_qs_rank=100,
            ),
        )


def test_requirement_extraction_covers_deadline_tuition_materials_and_verbatim_evidence():
    text = """
    Application Deadline: January 5, 2027
    Tuition and fees: $76,700 for the academic year.
    Application fee: $100.
    Minimum GPA 3.20 is expected.
    TOEFL minimum score 100. IELTS minimum score 7.0.
    Submit a resume, statement of purpose, official transcript, and three recommendation letters.
    Required quantitative preparation includes calculus and statistics.
    """
    extracted = extract_requirement_candidates(text)
    assert extracted["deadline"] == "2027-01-05"
    assert extracted["tuition"] == 76700
    assert extracted["fees"]["application_fee"] == 100
    assert extracted["min_gpa"] == 3.2
    assert extracted["language"] == {"TOEFL": 100.0, "IELTS": 7.0}
    assert {"CV / Resume", "Statement of Purpose / Essays", "Transcripts", "Recommendations"} <= set(extracted["materials"])
    assert extracted["evidence"]
    assert all(item["quote"] in text for item in extracted["evidence"])


def test_requirement_extraction_keeps_all_rounds_and_rejects_graduation_gpa():
    text = """
    Application Round 1 deadline: October 1, 2026
    Application Round 2 deadline: January 5, 2027
    Students must maintain at least a 3.0 GPA to graduate.
    """
    extracted = extract_requirement_candidates(text)
    assert [item["date"] for item in extracted["deadlines"]] == [
        "2026-10-01", "2027-01-05"
    ]
    assert extracted["min_gpa"] is None
    assert not any(item["field"] == "min_gpa" for item in extracted["evidence"])

    table_text = """
    Application Deadline
    The committee reviews applications before the dates listed below.
    Deadline
    Decision date
    January 5, 2027
    March 2027
    """
    table = extract_requirement_candidates(table_text)
    assert table["deadline"] == "2027-01-05"
    assert table["deadlines"][0]["raw"] == "January 5, 2027"


def test_requirement_extraction_rejects_toefl_code_and_preserves_hkd_currency():
    text = """
    Please note that the University's TOEFL institution code is 9671.
    Tuition: HK$350,000 for the whole programme.
    """
    extracted = extract_requirement_candidates(text)
    assert "TOEFL" not in extracted["language"]
    assert extracted["tuition"] == 350000
    assert extracted["currency"] == "HKD"


def test_requirement_extraction_reads_split_label_values_and_admission_section():
    text = """
    Tuition Fee
    HK$ 350,000 for the whole programme
    Admission Requirements
    In addition to the general requirements, applicants should have:
    - obtained a bachelor's degree in Engineering or Science discipline.
    - two years of relevant professional work experience.
    Application Deadline
    """
    extracted = extract_requirement_candidates(text)
    assert extracted["tuition"] == 350000
    assert extracted["currency"] == "HKD"
    assert len(extracted["prerequisites"]) == 3
    assert {item["field"] for item in extracted["evidence"]} == {
        "tuition",
        "prerequisites",
    }


@pytest.mark.asyncio
async def test_master_admission_evidence_keeps_bachelor_background_requirement():
    text = """
    Admission Requirements
    Applicants should have obtained a bachelor's degree in Engineering or Science.
    Applicants from other disciplines should have two years of relevant professional work experience.
    """
    extracted = await extract_source_requirements(
        SimpleNamespace(degree="master"),
        SimpleNamespace(content=text),
        use_ai=False,
    )
    quotes = [
        item["quote"]
        for item in extracted["evidence"]
        if item["field"] == "prerequisites"
    ]
    assert any("bachelor's degree" in quote for quote in quotes)
    assert any("professional work experience" in quote for quote in quotes)


def test_ai_requirement_evidence_must_exist_verbatim():
    text = "Application Deadline: January 5, 2027"
    rules = extract_requirement_candidates(text)
    ai = {
        "tuition": 99999,
        "materials": ["CV"],
        "evidence": [
            {"field": "tuition", "quote": "Tuition is $99,999", "value": 99999, "confidence": 0.9},
            {"field": "materials", "quote": "Application Deadline: January 5, 2027", "value": "CV", "confidence": 0.5},
        ],
    }
    merged = merge_ai_extraction(rules, ai, text)
    assert merged.get("tuition") is None
    assert merged["materials"] == ["CV"]
    assert all(item["quote"] in text for item in merged["evidence"])


def test_merged_requirements_do_not_promote_past_deadlines():
    source = SimpleNamespace(id="source-1", url="https://example.edu/program")
    merged = merge_source_extractions(
        [
            (
                source,
                {
                    "deadline": "2024-02-01",
                    "deadline_raw": "February 1",
                    "deadlines": [
                        {"date": "2024-02-01", "raw": "February 1", "round": ""}
                    ],
                    "evidence": [
                        {
                            "field": "deadline",
                            "quote": "The application deadline is February 1.",
                            "confidence": 0.9,
                        }
                    ],
                },
            )
        ]
    )
    assert merged["deadline"] is None
    assert merged["deadlines"] == []
    assert merged["evidence"][0]["quote"] == "The application deadline is February 1."


def test_all_seven_skill_outputs_enforce_json_schema():
    assert len(skill_registry.list()) == 7
    for skill in skill_registry.list():
        assert skill_registry.validate_output(skill, local_skill_output(skill, {}))
        with pytest.raises(ValueError):
            parse_skill_output(skill, "普通自然语言不是结构化输出")
        with pytest.raises(ValueError):
            skill_registry.validate_input(skill, {"unexpected": True})


class CorrectingCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            content = "不符合 Schema"
        else:
            content = json.dumps(
                {
                    "programs": [], "sources": [], "unverified": ["待确认"],
                    "summary": "已完成结构化纠错。",
                },
                ensure_ascii=False,
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content=content, tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


async def test_dashscope_repairs_invalid_skill_output_once():
    fake = CorrectingCompletions()
    provider = DashScopeChatProvider()
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    skill = skill_registry.get("program-research")
    async with SessionLocal() as session:
        output, _ = await provider.run(
            session, "schema-repair", "搜索项目", {}, skill,
            SimpleNamespace(allowed=lambda names: []),
        )
    assert "没有从项目目录" in json.loads(output)["summary"]
    assert len(fake.calls) == 2
    assert "输出校验失败" in fake.calls[1]["messages"][-1]["content"]
