import hashlib
import json
from typing import Any, Dict, Iterable, List, Sequence


def content_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_context_manifest_data(
    *,
    goal_spec: Dict[str, Any],
    profile: Dict[str, Any],
    conversation_id: str,
    history: Sequence[Any],
    conversation_memory: Dict[str, Any],
    program: Dict[str, Any],
    material_kind: str,
    slot_key: str,
    selected_resource_ids: Iterable[str],
    experiences: Sequence[Dict[str, Any]],
    documents: Sequence[Dict[str, Any]],
    drafts: Sequence[Dict[str, Any]],
    official_context: Dict[str, Any],
    current_draft: Dict[str, Any],
) -> Dict[str, Any]:
    selected = sorted(set(selected_resource_ids))
    experience_refs = [
        {
            "id": item.get("id"),
            "confirmed": bool(item.get("confirmed", True)),
            "content_hash": content_hash(item),
        }
        for item in experiences
    ]
    document_refs = [
        {
            "id": item.get("id"),
            "filename": item.get("filename", ""),
            "kind": item.get("kind", ""),
            "readable": bool(str(item.get("content") or "").strip()),
            "content_hash": content_hash(item.get("content", "")),
        }
        for item in documents
    ]
    draft_refs = [
        {
            "id": item.get("id"),
            "program_id": item.get("source_program_id") or item.get("program_id"),
            "version": item.get("version"),
            "content_hash": content_hash(item.get("content", "")),
        }
        for item in drafts
    ]
    evidence_refs = [
        {
            "id": item.get("evidence_id") or item.get("id"),
            "source_id": item.get("source_id"),
            "url": item.get("url") or item.get("locator"),
            "quote_hash": content_hash(item.get("quote", "")),
        }
        for item in official_context.get("evidence", [])
    ]
    current_ref = (
        {
            "id": current_draft.get("id"),
            "version": current_draft.get("version"),
            "content_hash": content_hash(current_draft.get("content", "")),
        }
        if current_draft
        else {}
    )
    manifest = {
        "intent": goal_spec.get("intent", "chat"),
        "program": program,
        "material_kind": material_kind,
        "slot_key": slot_key,
        "selected_resource_ids": selected,
        "profile_ref": {
            "id": profile.get("id"),
            "version": profile.get("updated_at"),
            "confirmed": bool(profile.get("confirmed")),
            "content_hash": content_hash(profile),
        },
        "conversation_ref": {
            "id": conversation_id,
            "message_count": int(
                conversation_memory.get("total_message_count") or len(history)
            ),
            "recent_message_count": len(history),
            "summarized_message_count": int(
                conversation_memory.get("summarized_message_count") or 0
            ),
            "stable_summary_hash": conversation_memory.get("summary_hash", ""),
            "memory_hash": conversation_memory.get("memory_hash", ""),
            "resource_issues": list(
                conversation_memory.get("resource_issues") or []
            ),
            "decision_conflicts": list(
                conversation_memory.get("decision_conflicts") or []
            ),
            "content_hash": content_hash(
                {
                    "stable_summary": conversation_memory.get("stable_summary", ""),
                    "decisions": conversation_memory.get("decisions", []),
                    "recent": [
                        {"role": item.get("role"), "content": item.get("content")}
                        for item in history
                    ],
                }
            ),
        },
        "experience_refs": experience_refs,
        "document_refs": document_refs,
        "draft_refs": draft_refs,
        "official_evidence_refs": evidence_refs,
        "current_draft_ref": current_ref,
    }
    manifest["content_hashes"] = {
        "manifest": content_hash(manifest),
        "official_requirement": content_hash(official_context),
    }
    return manifest


def validate_context_manifest(
    manifest: Dict[str, Any], goal_spec: Dict[str, Any]
) -> Dict[str, Any]:
    issues: List[str] = []
    warnings: List[str] = []
    intent = str(goal_spec.get("intent") or "chat")
    selected = set(manifest.get("selected_resource_ids") or [])
    loaded_documents = {f"document:{item.get('id')}" for item in manifest.get("document_refs", [])}
    loaded_drafts = {f"draft:{item.get('id')}" for item in manifest.get("draft_refs", [])}
    requested_documents = {item for item in selected if item.startswith("document:")}
    requested_drafts = {item for item in selected if item.startswith("draft:")}
    if requested_documents - loaded_documents:
        issues.append("selected_documents_not_loaded")
    if requested_drafts - loaded_drafts:
        issues.append("selected_drafts_not_loaded")
    if any(not item.get("readable") for item in manifest.get("document_refs", [])):
        issues.append("selected_document_has_no_readable_content")
    if not manifest.get("program", {}).get("id"):
        issues.append("program_missing")
    if not manifest.get("material_kind") or not manifest.get("slot_key"):
        issues.append("material_identity_missing")
    if intent in {"generate", "edit"}:
        if not manifest.get("profile_ref", {}).get("confirmed"):
            issues.append("confirmed_profile_missing")
        if not manifest.get("experience_refs"):
            issues.append("confirmed_experiences_missing")
        if not manifest.get("official_evidence_refs"):
            issues.append("official_requirement_evidence_missing")
    if intent == "edit" and not manifest.get("current_draft_ref", {}).get("id"):
        issues.append("current_draft_missing")
    resource_issues = list(
        manifest.get("conversation_ref", {}).get("resource_issues") or []
    )
    if any(value.startswith(("resource_missing:", "resource_unreadable:")) for value in resource_issues):
        issues.append("conversation_resource_unavailable")
    if any(value.startswith("resource_changed:") for value in resource_issues):
        warnings.append("conversation_resource_changed")
    if manifest.get("conversation_ref", {}).get("decision_conflicts"):
        warnings.append("conversation_decision_conflict_resolved_by_latest_instruction")
    return {
        "verdict": "accepted" if not issues else "rejected",
        "issues": issues,
        "warnings": warnings,
        "checked_resources": len(requested_documents | requested_drafts),
        "history_message_count": manifest.get("conversation_ref", {}).get("message_count", 0),
    }
