import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml
from jsonschema import Draft202012Validator

from app.core.config import settings


@dataclass
class Skill:
    name: str
    version: str
    description: str
    instructions: str
    prompt: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    tools: List[str]
    approval_required: List[str]
    evals: Dict[str, Any]


class SkillRegistry:
    def __init__(self, root: Path = settings.skills_dir) -> None:
        self.root = root
        self._skills: Dict[str, Skill] = {}

    def load(self) -> None:
        skills: Dict[str, Skill] = {}
        for name in settings.enabled_skills:
            directory = self.root / name
            if not directory.exists():
                continue
            skill_text = (directory / "SKILL.md").read_text(encoding="utf-8")
            metadata, instructions = self._parse_frontmatter(skill_text)
            extension_metadata = metadata.get("metadata", {})
            if not isinstance(extension_metadata, dict):
                extension_metadata = {}
            policy = yaml.safe_load((directory / "tool-policy.yaml").read_text(encoding="utf-8"))
            evals = yaml.safe_load((directory / "evals.yaml").read_text(encoding="utf-8"))
            skills[name] = Skill(
                name=name,
                version=str(extension_metadata.get("version", "0.1.0")),
                description=str(metadata.get("description", "")),
                instructions=instructions.strip(),
                prompt=(directory / "prompt.md").read_text(encoding="utf-8").strip(),
                input_schema=json.loads(
                    (directory / "input.schema.json").read_text(encoding="utf-8")
                ),
                output_schema=json.loads(
                    (directory / "output.schema.json").read_text(encoding="utf-8")
                ),
                tools=list(policy.get("tools", [])),
                approval_required=list(policy.get("approval_required", [])),
                evals=evals or {},
            )
        self._skills = skills

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple:
        if not text.startswith("---"):
            return {}, text
        _, frontmatter, body = text.split("---", 2)
        return yaml.safe_load(frontmatter) or {}, body

    def get(self, name: str) -> Skill:
        if not self._skills:
            self.load()
        if name not in self._skills:
            raise KeyError(f"Skill 未启用或不存在：{name}")
        return self._skills[name]

    def list(self) -> List[Skill]:
        if not self._skills:
            self.load()
        return list(self._skills.values())

    def route(self, message: str) -> Skill:
        lowered = message.lower()
        rules = [
            ("cv", ["cv", "简历", "resume"]),
            ("ps", ["ps", "个人陈述", "文书", "statement"]),
            ("application-timeline", ["时间线", "任务", "截止", "timeline", "deadline"]),
            ("shortlist-builder", ["选校", "冲刺", "主申", "稳妥", "shortlist"]),
            ("program-compare", ["比较", "对比", "哪个好", "compare"]),
            ("program-research", ["项目", "学校", "官网", "学费", "申请要求", "program"]),
        ]
        for key, keywords in rules:
            if any(keyword in lowered for keyword in keywords):
                name = "cv-planner" if key == "cv" else "ps-planner" if key == "ps" else key
                return self.get(name)
        return self.get("applicant-profile")

    @staticmethod
    def validate_input(skill: Skill, value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{skill.name} 输入必须是 JSON 对象")
        validator = Draft202012Validator(skill.input_schema)
        errors = sorted(validator.iter_errors(value), key=lambda item: list(item.path))
        if errors:
            details = "; ".join(
                f"{'.'.join(map(str, error.path)) or '$'}: {error.message}"
                for error in errors[:5]
            )
            raise ValueError(f"{skill.name} 输入不符合 Schema：{details}")
        return value

    @staticmethod
    def validate_output(skill: Skill, value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{skill.name} 输出必须是 JSON 对象")
        validator = Draft202012Validator(skill.output_schema)
        errors = sorted(validator.iter_errors(value), key=lambda item: list(item.path))
        if errors:
            details = "; ".join(
                f"{'.'.join(map(str, error.path)) or '$'}: {error.message}"
                for error in errors[:5]
            )
            raise ValueError(f"{skill.name} 输出不符合 Schema：{details}")
        return value


def parse_skill_output(skill: Skill, text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{skill.name} 最终输出不是合法 JSON：{exc}") from exc
    return SkillRegistry.validate_output(skill, value)


def ground_skill_output(
    skill: Skill, value: Dict[str, Any], grounded_programs: Dict[str, Dict[str, Any]],
    grounded_urls: List[str],
) -> Dict[str, Any]:
    """Remove catalog/source claims that did not come back from an executed tool."""
    if skill.name != "program-research":
        return value
    programs = []
    for candidate in value.get("programs", []):
        fact = grounded_programs.get(candidate.get("id", ""))
        if not fact:
            continue
        programs.append({
            "id": fact["id"], "university": fact["university"], "name": fact["name"],
            "field": fact.get("field", ""), "degree": fact.get("degree", ""),
            "country": fact.get("country", ""), "official_url": fact["official_url"],
            "reason": "专业方向和目标国家符合当前画像筛选条件；具体门槛以官网证据为准",
        })
    allowed_urls = set(grounded_urls) | {
        item["official_url"] for item in grounded_programs.values() if item.get("official_url")
    }
    sources = [item for item in value.get("sources", []) if item.get("url") in allowed_urls]
    unverified = list(value.get("unverified", []))
    if not programs:
        unverified.append("本次工具调用没有返回可引用的项目，未展示模型自行补充的项目")
        summary = "本次没有从项目目录或已读取的官网来源中取得可引用项目，请调整专业或检索条件。"
    else:
        summary = f"已从项目目录找到 {len(programs)} 个匹配项目；截止日期、学费和材料要求须以官网证据状态为准。"
    grounded = {**value, "programs": programs, "sources": sources,
                "unverified": list(dict.fromkeys(unverified)), "summary": summary}
    return SkillRegistry.validate_output(skill, grounded)


def local_skill_output(skill: Skill, context: Dict[str, Any]) -> Dict[str, Any]:
    profile = context.get("profile", {})
    experiences = [item for item in context.get("experiences", []) if item.get("confirmed")]
    summaries = {
        "applicant-profile": "已读取画像；请确认缺失的申请目标、目标国家和目标专业后再用于申请决策。",
        "program-research": "请在项目探索中按画像方向筛选，并逐项核验官网；没有逐字证据的截止日期均为待确认。",
        "program-compare": "请先选择需要比较的项目；系统会区分官网要求、匹配建议和申请风险，未核验字段不会作为事实。",
        "shortlist-builder": "选校会给出冲刺、主申和相对稳健分层及组合风险，不承诺录取概率。",
        "cv-planner": "CV 只使用经历库中已确认的真实经历。",
        "ps-planner": "PS 只使用已确认的真实素材，并根据项目题目生成提纲。",
        "application-timeline": "时间线必须基于已核验截止日期生成。",
    }
    summary = summaries[skill.name]
    if skill.name == "applicant-profile":
        missing = [
            key for key in ["school", "major", "gpa", "target_countries", "target_fields", "intake"]
            if not profile.get(key)
        ]
        return {"known": profile, "missing": missing, "questions": [f"请补充 {key}" for key in missing[:3]], "summary": summary}
    if skill.name == "program-research":
        return {"programs": [], "sources": [], "unverified": ["尚未执行官网核验"], "summary": summary}
    if skill.name == "program-compare":
        return {"dimensions": ["专业匹配", "截止日期", "费用", "材料", "证据状态"], "programs": [], "risks": ["请选择项目"], "summary": summary}
    if skill.name == "shortlist-builder":
        return {"items": [], "portfolio_risks": ["尚未选择候选项目"], "summary": summary}
    selected = [{"experience_id": item["id"], "reason": "已确认且可追溯"} for item in experiences[:5]]
    if skill.name == "cv-planner":
        return {"selected_experiences": selected, "order": [item["experience_id"] for item in selected], "focus": "突出与目标项目相关的真实成果", "gaps": [] if selected else ["缺少已确认经历"], "summary": summary}
    if skill.name == "ps-planner":
        return {"prompt_analysis": {"status": "missing", "prompt": "", "requirements": [], "word_limit": None}, "selected_evidence": [{"experience_id": item["id"], "use": "能力或动机证据"} for item in experiences[:3]], "outline": ["申请动机", "能力证据", "项目匹配", "职业目标"], "gaps": [] if experiences else ["缺少已确认素材"], "summary": summary}
    return {"deadline": "", "tasks": [], "risks": ["截止日期尚未核验"], "summary": summary}


skill_registry = SkillRegistry()
