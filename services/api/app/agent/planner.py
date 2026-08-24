from typing import Any, Dict, List, Optional

from app.agent.skills import Skill

SKILL_STEPS = {
    "applicant-profile": ["读取现有画像", "识别已知与缺失信息", "提出下一步或保存候选事实"],
    "program-research": ["拆解检索条件", "搜索候选项目", "核对官网证据", "汇总结果与待确认项"],
    "program-compare": ["确定比较维度", "读取项目与要求", "校验硬性条件", "输出差异与风险"],
    "shortlist-builder": ["读取画像与约束", "筛选候选项目", "评分并分层", "验证理由和风险"],
    "cv-planner": ["读取项目要求", "检索已确认经历", "选择并排序经历", "检查真实性与缺口"],
    "ps-planner": ["解析项目与题目", "检索已确认经历", "匹配真实素材", "生成提纲并检查贴题"],
    "application-timeline": ["读取截止日期", "拆解申请里程碑", "计算任务日期", "检查依赖与风险"],
}


PROGRAM_RESEARCH_RECIPE = [
    {
        "step_type": "parse_search_goal",
        "name": "解析检索目标与硬约束",
        "allowed_tools": [],
        "acceptance_criteria": ["生成 GoalSpec", "学校、项目、专业和学位层级可追踪"],
    },
    {
        "step_type": "discover_official_catalog_and_programs",
        "name": "逐步定位目录、读取白名单并独立核验项目身份",
        "allowed_tools": ["discover_official_programs"],
        "acceptance_criteria": [
            "只定位官方目录，不在该阶段生成项目 URL",
            "目录属于目标学校官方域名并可读取",
            "未指定学校时按国家与 QS 分批选择学校并逐校遍历",
            "程序先提取目录中的真实链接再交给模型",
            "模型只能从输入 URL 白名单中选择项目或下一层目录",
            "每个候选都与实际读取的来源目录绑定",
            "候选项目满足 GoalSpec 硬约束并排除当前页面项目",
            "不足目标数量时继续下一目录或下一批学校",
            "拒绝课程、新闻、活动、通用页面及非官方网站",
            "验证模型与检索模型不同",
            "验证模型只读取原始目录与项目页面证据",
            "课程页、新闻页、错误学位和错误项目必须拒绝",
        ],
    },
    {
        "step_type": "verify_official_requirements",
        "name": "读取官网并核验逐字证据",
        "allowed_tools": [
            "fetch_official_source",
            "extract_program_requirements",
            "verify_program_official",
        ],
        "acceptance_criteria": ["来源为项目官网", "关键要求包含可追溯原文证据"],
    },
    {
        "step_type": "independent_verify_result",
        "name": "独立复核项目身份与官网证据",
        "allowed_tools": [],
        "acceptance_criteria": [
            "最终项目必须同时通过检索身份与官网证据校验",
            "官网链接必须与已核验证据来源绑定",
        ],
    },
    {
        "step_type": "match_applicant_profile",
        "name": "结合画像评估匹配关系",
        "allowed_tools": [],
        "acceptance_criteria": ["只使用已确认画像事实", "未知信息不推断"],
    },
    {
        "step_type": "render_recommendations",
        "name": "发布通过最终审计的推荐结果",
        "allowed_tools": [],
        "acceptance_criteria": ["只展示已通过身份和官网证据验收的项目"],
    },
]

MATERIAL_WRITING_RECIPE = [
    {
        "step_type": "resolve_writing_intent",
        "name": "识别讨论、生成、修改或采用意图",
        "allowed_tools": [],
        "acceptance_criteria": ["写操作意图明确", "普通讨论不创建文稿版本"],
    },
    {
        "step_type": "resolve_material_context",
        "name": "构建并验收 Context Manifest",
        "allowed_tools": [],
        "acceptance_criteria": ["历史、官网要求和所选资源已实际加载", "来源哈希可追踪"],
    },
    {
        "step_type": "generate_or_edit_draft",
        "name": "生成回复或修改当前文稿",
        "allowed_tools": [],
        "acceptance_criteria": ["只使用 Manifest 内的事实和材料"],
    },
    {
        "step_type": "validate_draft",
        "name": "校验内容完整性与来源边界",
        "allowed_tools": [],
        "acceptance_criteria": ["无虚构来源", "修改基于当前版本", "正文发生有效变化"],
    },
    {
        "step_type": "independent_verify_draft",
        "name": "独立复核要求覆盖与事实来源",
        "allowed_tools": [],
        "acceptance_criteria": [
            "官网要求得到覆盖",
            "无超出 Context Manifest 的事实",
            "无来源冲突或未经支持的陈述",
        ],
    },
    {
        "step_type": "present_candidate_version",
        "name": "保存并展示候选版本",
        "allowed_tools": [],
        "acceptance_criteria": ["版本、父版本和来源关系已持久化"],
    },
    {
        "step_type": "adopt_version",
        "name": "用户确认后采用到申请包",
        "allowed_tools": [],
        "acceptance_criteria": ["用户明确确认", "采用版本属于当前申请材料"],
    },
]


def build_material_writing_plan(goal_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "id": f"step-{index + 1}",
            **item,
            "status": "pending",
            "dependencies": [] if index == 0 else [f"step-{index}"],
            "expected_output": "；".join(item["acceptance_criteria"]),
            "goal_constraints": list(goal_spec.get("hard_constraints") or []),
        }
        for index, item in enumerate(MATERIAL_WRITING_RECIPE)
    ]


def build_plan(
    skill: Skill, goal: str, goal_spec: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    if skill.name == "program-research":
        recipe = PROGRAM_RESEARCH_RECIPE
        return [
            {
                "id": f"step-{index + 1}",
                **item,
                "status": "pending",
                "dependencies": [] if index == 0 else [f"step-{index}"],
                "expected_output": "；".join(item["acceptance_criteria"]),
                "goal_constraints": list((goal_spec or {}).get("hard_constraints") or []),
            }
            for index, item in enumerate(recipe)
        ]
    steps = SKILL_STEPS.get(skill.name, ["分析目标", "执行任务", "验证输出"])
    return [
        {
            "id": f"step-{index + 1}",
            "name": name,
            "status": "pending",
            "dependencies": [] if index == 0 else [f"step-{index}"],
            "expected_output": f"{name}的可验证结果",
        }
        for index, name in enumerate(steps)
    ]


def should_replan(error_type: str) -> bool:
    return error_type in {
        "no_result",
        "source_conflict",
        "goal_changed",
        "hard_constraint_failed",
        "tool_timeout",
        "tool_error",
        "invalid_arguments",
        "permission_denied",
    }


def replan_after_failure(
    plan: List[Dict[str, Any]], error_type: str, tool_name: str
) -> List[Dict[str, Any]]:
    """Preserve verified work and annotate the remaining plan with a bounded recovery step."""
    updated = [dict(item) for item in plan]
    recovery = {
        "id": f"recovery-{sum(1 for item in updated if str(item['id']).startswith('recovery-')) + 1}",
        "name": f"处理 {tool_name} 的 {error_type}",
        "status": "pending",
        "dependencies": [
            item["id"] for item in updated if item.get("status") in {"completed", "accepted"}
        ],
        "expected_output": "改用更安全的参数、替代工具，或明确返回待确认项",
        "recovery": True,
    }
    updated.append(recovery)
    return updated
