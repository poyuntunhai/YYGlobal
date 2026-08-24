import hashlib
import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Sequence, Tuple

RECENT_MESSAGE_LIMIT = 12
RECENT_CONTENT_LIMIT = 4_000
RECENT_TOTAL_LIMIT = 32_000
SUMMARY_LIMIT = 8_000
DECISION_LIMIT = 40

DECISION_PATTERNS = (
    r"\b(?:must|should|prefer|use|keep|remove|avoid|focus|change)\b",
    r"(?:必须|需要|不要|不能|优先|重点|使用|选择|保留|删除|改成|改为|确认|决定)",
)
NEGATIVE_DECISION = re.compile(
    r"(?:不要|不能|删除|移除|避免|不再|不使用|do not|don't|remove|avoid)", re.I
)
DECISION_ACTION = re.compile(
    r"(?:必须|需要|不要|不能|优先|重点|使用|选择|保留|删除|移除|改成|改为|确认|决定|"
    r"must|should|prefer|use|keep|remove|avoid|focus|change|do not|don't)",
    re.I,
)


def _value(item: Any, key: str, default: Any = "") -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalized_messages(messages: Sequence[Any]) -> List[Dict[str, str]]:
    return [
        {
            "id": str(_value(item, "id", "")),
            "role": str(_value(item, "role", "")),
            "content": str(_value(item, "content", "")).strip(),
        }
        for item in messages
    ]


def _bounded_recent(messages: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    selected = []
    used = 0
    for item in reversed(messages[-RECENT_MESSAGE_LIMIT:]):
        content = item["content"][:RECENT_CONTENT_LIMIT]
        remaining = RECENT_TOTAL_LIMIT - used
        if remaining <= 0:
            break
        content = content[:remaining]
        selected.append({"id": item["id"], "role": item["role"], "content": content})
        used += len(content)
    return list(reversed(selected))


def _append_summary(previous: str, messages: Sequence[Dict[str, str]]) -> str:
    additions = [
        f"[{item['role']}] {item['content'][:600]}"
        for item in messages
        if item["content"]
    ]
    combined = "\n".join([value for value in [previous.strip(), *additions] if value])
    if len(combined) <= SUMMARY_LIMIT:
        return combined
    return combined[:2_000].rstrip() + "\n…\n" + combined[-5_995:].lstrip()


def _extract_decisions(
    previous: Iterable[str], messages: Sequence[Dict[str, str]]
) -> List[str]:
    decisions = list(previous)
    for item in messages:
        if item["role"] != "user":
            continue
        for sentence in re.split(r"(?<=[。！？.!?])\s*|\n+", item["content"]):
            sentence = " ".join(sentence.split())
            if sentence and any(re.search(pattern, sentence, re.I) for pattern in DECISION_PATTERNS):
                decisions.append(sentence[:500])
    return list(dict.fromkeys(decisions))[-DECISION_LIMIT:]


def _decision_conflicts(
    stable_decisions: Sequence[str], recent_messages: Sequence[Dict[str, str]]
) -> List[Dict[str, str]]:
    recent_decisions = _extract_decisions([], recent_messages)
    conflicts = []
    for older in stable_decisions:
        older_negative = bool(NEGATIVE_DECISION.search(older))
        older_subject = re.sub(r"\W+", "", DECISION_ACTION.sub("", older)).lower()
        if not older_subject:
            continue
        for newer in recent_decisions:
            if older_negative == bool(NEGATIVE_DECISION.search(newer)):
                continue
            newer_subject = re.sub(r"\W+", "", DECISION_ACTION.sub("", newer)).lower()
            similarity = SequenceMatcher(None, older_subject, newer_subject).ratio()
            if similarity >= 0.62:
                conflicts.append({
                    "previous": older,
                    "latest": newer,
                    "resolution": "latest_user_instruction_wins",
                })
    return conflicts[-10:]


def build_resource_index(
    selected_resource_ids: Iterable[str],
    resource_snapshots: Sequence[Dict[str, Any]],
    previous_index: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    snapshots = {str(item.get("resource_id")): item for item in resource_snapshots}
    index = {}
    issues = []
    for resource_id in sorted(set(selected_resource_ids)):
        snapshot = snapshots.get(resource_id)
        if not snapshot:
            index[resource_id] = {"status": "missing", "content_hash": ""}
            issues.append(f"resource_missing:{resource_id}")
            continue
        readable = bool(snapshot.get("readable", True))
        content_hash = str(snapshot.get("content_hash") or "")
        previous_hash = str((previous_index.get(resource_id) or {}).get("content_hash") or "")
        status = "loaded" if readable else "unreadable"
        if previous_hash and content_hash and previous_hash != content_hash:
            status = "changed"
            issues.append(f"resource_changed:{resource_id}")
        if not readable:
            issues.append(f"resource_unreadable:{resource_id}")
        index[resource_id] = {
            "status": status,
            "content_hash": content_hash,
            "label": str(snapshot.get("label") or ""),
            "kind": str(snapshot.get("kind") or ""),
        }
    return index, issues


def build_conversation_memory(
    messages: Sequence[Any],
    previous_state: Dict[str, Any],
    selected_resource_ids: Iterable[str],
    resource_snapshots: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    normalized = _normalized_messages(messages)
    older = normalized[:-RECENT_MESSAGE_LIMIT]
    previous_count = int(previous_state.get("summarized_message_count") or 0)
    if previous_count < 0 or previous_count > len(older):
        previous_count = 0
        previous_summary = ""
        previous_decisions = []
    else:
        previous_summary = str(previous_state.get("stable_summary") or "")
        previous_decisions = list(previous_state.get("decisions") or [])
    unsummarized = older[previous_count:]
    stable_summary = _append_summary(previous_summary, unsummarized)
    decisions = _extract_decisions(previous_decisions, unsummarized)
    recent = _bounded_recent(normalized)
    decision_conflicts = _decision_conflicts(decisions, recent)
    resource_index, resource_issues = build_resource_index(
        selected_resource_ids,
        resource_snapshots,
        dict(previous_state.get("resource_index") or {}),
    )
    state = {
        "version": 1,
        "stable_summary": stable_summary,
        "summary_hash": _hash(stable_summary),
        "decisions": decisions,
        "summarized_message_count": len(older),
        "recent_message_count": len(recent),
        "total_message_count": len(normalized),
        "resource_index": resource_index,
        "resource_issues": resource_issues,
        "decision_conflicts": decision_conflicts,
    }
    state["memory_hash"] = _hash(state)
    return state, recent
