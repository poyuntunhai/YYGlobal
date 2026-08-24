from typing import Any, Dict, List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entities import Program
from app.services.business import canonical_fields

COUNTRY_ALIASES = {
    "United States": ("united states", "usa", "u.s.", "美国"),
    "United Kingdom": ("united kingdom", "uk", "英国"),
    "Canada": ("canada", "加拿大"),
    "Singapore": ("singapore", "新加坡"),
    "Australia": ("australia", "澳大利亚", "澳洲"),
}

MASTER_KEYWORDS = ("master", "master's", "masters", "ms", "m.s.", "mcs", "meng", "硕士")
DOCTORAL_KEYWORDS = ("phd", "ph.d", "doctoral", "doctorate", "博士")

MATERIAL_EDIT_PATTERNS = (
    "重写", "改写", "润色", "修改", "删掉", "删除", "替换", "扩写", "缩写",
    "缩短", "翻译", "合并", "rewrite", "revise", "edit", "polish", "shorten",
    "expand", "translate", "replace",
)
MATERIAL_GENERATE_PATTERNS = (
    "生成", "起草", "写一份", "写一篇", "完整文稿", "完整文书", "正文",
    "generate", "draft", "write a", "write the", "full statement", "full cv",
)
MATERIAL_ADOPT_PATTERNS = (
    "采用这个版本", "确认当前版本", "用于申请包", "adopt this version",
    "use this version",
)


def _contains_keyword(value: str, keyword: str) -> bool:
    if keyword.isascii() and keyword.replace(".", "").isalnum():
        padded = f" {value.replace('-', ' ')} "
        return f" {keyword} " in padded
    return keyword in value


async def parse_goal_spec(
    session: AsyncSession, skill_name: str, message: str
) -> Dict[str, Any]:
    """Build a conservative, deterministic GoalSpec before the LLM plans execution."""
    lowered = " ".join(message.lower().split())
    items: List[Program] = list(
        (await session.scalars(select(Program).where(Program.active.is_(True)))).all()
    )
    universities = sorted({item.university for item in items}, key=len, reverse=True)
    named_university = next(
        (value for value in universities if value.lower() in lowered), ""
    )
    program_names = sorted(
        {
            item.name
            for item in items
            if not named_university or item.university == named_university
        },
        key=len,
        reverse=True,
    )
    named_program = next(
        (value for value in program_names if value.lower() in lowered), ""
    )
    fields = canonical_fields([message])
    countries = [
        canonical
        for canonical, aliases in COUNTRY_ALIASES.items()
        if any(_contains_keyword(lowered, alias) for alias in aliases)
    ]
    degree_level = ""
    if any(_contains_keyword(lowered, keyword) for keyword in DOCTORAL_KEYWORDS):
        degree_level = "doctoral"
    elif any(_contains_keyword(lowered, keyword) for keyword in MASTER_KEYWORDS):
        degree_level = "master"
    elif named_university:
        matching_degrees = {
            item.degree.strip().lower()
            for item in items
            if item.university == named_university
            and (not fields or item.field.lower() in {field.lower() for field in fields})
        }
        if matching_degrees == {"master"}:
            degree_level = "master"

    entities = {
        "university": named_university,
        "program": named_program,
        "fields": fields,
        "degree_level": degree_level,
        "countries": countries,
    }
    hard_constraints = [
        {"field": key, "operator": "exact", "value": value}
        for key, value in (
            ("university", named_university),
            ("program", named_program),
            ("degree_level", degree_level),
        )
        if value
    ]
    hard_constraints.extend(
        {"field": "field", "operator": "in", "value": value} for value in fields
    )
    hard_constraints.extend(
        {"field": "country", "operator": "in", "value": value} for value in countries
    )
    return {
        "intent": skill_name.replace("-", "_"),
        "recipe": "program_research" if skill_name == "program-research" else skill_name,
        "entities": entities,
        "hard_constraints": hard_constraints,
        "required_outputs": (
            ["program_identity", "official_requirements", "evidence"]
            if skill_name == "program-research"
            else []
        ),
        "evidence_policy": {
            "official_source_required": skill_name == "program-research",
            "verbatim_quote_required": skill_name == "program-research",
        },
        "write_intent": False,
    }


async def resolve_skill_and_goal(session: AsyncSession, message: str):
    """Let explicit catalog entities override the generic profile fallback route."""
    from app.agent.skills import skill_registry

    skill = skill_registry.route(message)
    if skill.name == "applicant-profile":
        program_goal = await parse_goal_spec(session, "program-research", message)
        entities = program_goal.get("entities", {})
        if entities.get("university") or entities.get("program"):
            return skill_registry.get("program-research"), program_goal
    return skill, await parse_goal_spec(session, skill.name, message)


def parse_material_goal(
    message: str,
    material_kind: str,
    program_id: str,
    slot_key: str,
    has_current_draft: bool,
) -> Dict[str, Any]:
    lowered = " ".join(message.lower().split())
    if any(value in lowered for value in MATERIAL_ADOPT_PATTERNS):
        intent = "adopt"
    elif has_current_draft and any(value in lowered for value in MATERIAL_EDIT_PATTERNS):
        intent = "edit"
    elif any(value in lowered for value in MATERIAL_GENERATE_PATTERNS):
        intent = "generate"
    else:
        intent = "chat"
    return {
        "intent": intent,
        "recipe": "material_writing",
        "entities": {
            "program_id": program_id,
            "material_kind": material_kind,
            "slot_key": slot_key,
            "current_draft_required": intent == "edit",
        },
        "hard_constraints": [
            {"field": "program_id", "operator": "exact", "value": program_id},
            {"field": "material_kind", "operator": "exact", "value": material_kind},
            {"field": "slot_key", "operator": "exact", "value": slot_key},
        ],
        "required_outputs": (
            ["complete_draft", "source_lineage", "context_manifest"]
            if intent in {"generate", "edit"}
            else ["grounded_chat_response"]
        ),
        "write_intent": intent in {"generate", "edit", "adopt"},
        "requires_user_confirmation": intent == "adopt",
    }
