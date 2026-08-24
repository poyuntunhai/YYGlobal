from io import BytesIO

from sqlalchemy import select

from app.agent.goals import parse_material_goal
from app.agent.material_context import validate_context_manifest
from app.agent.provider import provider
from app.core.config import settings
from app.core.database import SessionLocal
from app.models.entities import (
    AgentRun,
    EvidenceChunk,
    MaterialDraft,
    Message,
    ProgramRequirement,
    ProgramSource,
)


def test_material_goal_and_context_validator_block_incomplete_generation():
    goal = parse_material_goal(
        "请生成一篇完整的 PS 正文", "ps", "program-1", "statement", False
    )
    assert goal["intent"] == "generate"
    evaluation = validate_context_manifest(
        {
            "program": {"id": "program-1"},
            "material_kind": "ps",
            "slot_key": "statement",
            "selected_resource_ids": ["document:missing"],
            "profile_ref": {"confirmed": False},
            "conversation_ref": {"message_count": 1},
            "experience_refs": [],
            "document_refs": [],
            "draft_refs": [],
            "official_evidence_refs": [],
            "current_draft_ref": {},
        },
        goal,
    )
    assert evaluation["verdict"] == "rejected"
    assert {
        "selected_documents_not_loaded",
        "confirmed_profile_missing",
        "confirmed_experiences_missing",
        "official_requirement_evidence_missing",
    } <= set(evaluation["issues"])


async def _set_verified_ps_requirement(program_id: str, official_url: str) -> None:
    quote = "Submit a statement of purpose describing your preparation and goals."
    async with SessionLocal() as session:
        requirement = await session.scalar(
            select(ProgramRequirement).where(
                ProgramRequirement.program_id == program_id
            )
        )
        requirement.materials = ["Statement of Purpose / Essays"]
        requirement.verified = True
        source = ProgramSource(
            program_id=program_id,
            url=official_url,
            title="Official application requirements",
            source_type="official",
            content=quote,
            status="verified",
        )
        session.add(source)
        await session.flush()
        evidence = EvidenceChunk(
            program_id=program_id,
            source_id=source.id,
            field="materials",
            quote=quote,
            locator=official_url,
            confidence=0.99,
        )
        session.add(evidence)
        requirement.source_ids = [source.id]
        await session.commit()


def _confirmed_profile(client):
    response = client.put(
        "/api/profile",
        json={
            "full_name": "Context Test Student",
            "current_school": "Test University",
            "current_major": "Computer Science",
            "degree": "Bachelor",
            "gpa": 3.8,
            "gpa_scale": 4.0,
            "language_scores": {"TOEFL": 108},
            "target_countries": ["United States"],
            "target_fields": ["Computer Science"],
            "intake": "2027 Fall",
            "budget": 70000,
            "preferences": {},
            "confirmed": True,
            "experiences": [
                {
                    "kind": "project",
                    "title": "Grounded Retrieval Project",
                    "organization": "Test Lab",
                    "description": "Built and evaluated a retrieval system.",
                    "tags": ["RAG"],
                    "confirmed": True,
                }
            ],
        },
    )
    assert response.status_code == 200
    return response.json()["experiences"][0]["id"]


async def test_material_recipe_manifest_lineage_and_adoption(client, monkeypatch):
    experience_id = _confirmed_profile(client)
    programs = client.get("/api/programs?personalized=false").json()
    program = programs[0]
    await _set_verified_ps_requirement(program["id"], program["official_url"])
    shortlist = client.post(
        "/api/shortlists",
        json={"name": "P0-B material test", "program_ids": [program["id"]]},
    )
    assert shortlist.status_code == 201
    package = client.post(
        f"/api/application-packages/{program['id']}/refresh"
    ).json()
    ps_row = next(item for item in package["checklist"] if item["category"] == "ps")

    upload = client.post(
        "/api/documents",
        files={
            "file": (
                "project-notes.md",
                BytesIO(b"The retrieval project improved evidence traceability."),
                "text/markdown",
            )
        },
        data={"kind": "other"},
    )
    assert upload.status_code == 201
    document_id = upload.json()["id"]

    seen_payloads = []

    async def fake_generate(payload):
        seen_payloads.append(payload)
        current = payload.get("current_draft") or {}
        suffix = " Revised with a clearer final goal." if current else " First complete version."
        return {
            "response_type": "draft",
            "message": "Generated from the supplied context.",
            "title": "Grounded Statement of Purpose",
            "content": (
                str(current.get("content") or "I built a grounded retrieval system.")
                + suffix
            ),
            "source_experience_ids": [experience_id],
            "warnings": [],
            "model_info": {"provider": "test", "model": "deterministic"},
        }

    async def accepted_verify(payload, candidate):
        return {
            "verdict": "accepted",
            "summary": "Grounded in the supplied context.",
            "requirement_coverage": ["statement of purpose"],
            "unsupported_claims": [],
            "conflicts": [],
            "issues": [],
            "confidence": 1.0,
            "mode": "test-independent-verifier",
        }

    monkeypatch.setattr(provider, "generate_material", fake_generate)
    monkeypatch.setattr(provider, "verify_material", accepted_verify)
    conversation = client.post(
        "/api/writing-conversations",
        json={
            "program_id": program["id"],
            "slot_key": ps_row["slot_key"],
            "material_kind": "ps",
            "title": "P0-B test conversation",
            "resource_ids": [
                "profile",
                "confirmed_experiences",
                "official_requirements",
                f"document:{document_id}",
            ],
        },
    )
    assert conversation.status_code == 201
    conversation_id = conversation.json()["id"]

    first = client.post(
        f"/api/writing-conversations/{conversation_id}/messages",
        json={"message": "请生成一篇完整的 PS 正文"},
    )
    assert first.status_code == 200, first.text
    first_body = first.json()
    first_draft = first_body["latest_draft"]
    assert f"document:{document_id}" in first_body["resource_ids"]
    assert first_draft["parent_id"] is None
    assert first_draft["context_manifest_id"]
    assert first_draft["source_refs"]["document_ids"] == [document_id]
    assert first_draft["source_refs"]["experience_ids"] == [experience_id]

    manifest = client.get(
        f"/api/context-manifests/{first_draft['context_manifest_id']}"
    )
    assert manifest.status_code == 200
    manifest_body = manifest.json()
    assert manifest_body["completeness"]["verdict"] == "accepted"
    assert manifest_body["document_refs"][0]["readable"] is True
    assert manifest_body["official_evidence_refs"]
    assert seen_payloads[0]["context_manifest"]["id"] == manifest_body["id"]

    second = client.post(
        f"/api/writing-conversations/{conversation_id}/messages",
        json={"message": "请修改当前文稿，让结尾更简洁"},
    )
    assert second.status_code == 200, second.text
    second_draft = second.json()["latest_draft"]
    assert second_draft["parent_id"] == first_draft["id"]
    assert second_draft["root_id"] == first_draft["id"]
    assert second_draft["version_number"] == 2
    assert second_draft["revision_type"] == "ai_revision"
    assert second_draft["source_refs"]["current_draft_id"] == first_draft["id"]
    assert seen_payloads[1]["conversation_history"]
    assert seen_payloads[1]["reference_documents"][0]["id"] == document_id
    assert seen_payloads[1]["current_draft"]["id"] == first_draft["id"]

    async def unchanged_generate(payload):
        current = payload["current_draft"]
        return {
            "response_type": "draft",
            "message": "No effective change.",
            "title": current["title"],
            "content": current["content"],
            "source_experience_ids": [experience_id],
            "warnings": [],
            "model_info": {"provider": "test", "model": "unchanged"},
        }

    monkeypatch.setattr(provider, "generate_material", unchanged_generate)
    unchanged = client.post(
        f"/api/writing-conversations/{conversation_id}/messages",
        json={"message": "请修改当前文稿，但保持原文不变"},
    )
    assert unchanged.status_code == 200
    assert unchanged.json()["latest_draft"]["id"] == second_draft["id"]
    versions = client.get(
        f"/api/material-drafts?program_id={program['id']}&slot_key={ps_row['slot_key']}"
    ).json()
    assert {item["id"] for item in versions} == {
        first_draft["id"],
        second_draft["id"],
    }

    run_id = second_draft["model_info"]["agent_run_id"]
    trace = client.get(f"/api/agent-runs/{run_id}/trace")
    assert trace.status_code == 200
    assert [item["status"] for item in trace.json()["steps"]] == [
        "accepted",
        "accepted",
        "accepted",
        "accepted",
        "accepted",
        "accepted",
        "pending",
    ]

    reviewed = client.patch(
        f"/api/material-drafts/{second_draft['id']}", json={"status": "reviewed"}
    )
    assert reviewed.status_code == 201
    package = client.post(
        f"/api/application-packages/{program['id']}/refresh"
    ).json()
    adopted = client.patch(
        f"/api/application-packages/{package['id']}/materials",
        json={
            "material_key": ps_row["material_key"],
            "status": "ready",
            "selected_asset_type": "draft",
            "selected_asset_id": second_draft["id"],
            "note": "P0-B explicit adoption test",
        },
    )
    assert adopted.status_code == 200, adopted.text
    adopted_trace = client.get(f"/api/agent-runs/{run_id}/trace").json()
    assert adopted_trace["steps"][-1]["status"] == "accepted"
    assert adopted_trace["run"]["stop_reason"] == "adopted"
    adopted_manifest = client.get(
        f"/api/context-manifests/{second_draft['context_manifest_id']}"
    ).json()
    assert adopted_manifest["status"] == "adopted"

    other_program = programs[1]
    await _set_verified_ps_requirement(
        other_program["id"], other_program["official_url"]
    )
    selected_other = client.post(
        "/api/shortlists",
        json={"name": "P0-B cross-project", "program_ids": [other_program["id"]]},
    )
    assert selected_other.status_code == 201
    other_package = client.post(
        f"/api/application-packages/{other_program['id']}/refresh"
    ).json()
    other_ps = next(
        item for item in other_package["checklist"] if item["category"] == "ps"
    )
    reused = client.patch(
        f"/api/application-packages/{other_package['id']}/materials",
        json={
            "material_key": other_ps["material_key"],
            "status": "ready",
            "selected_asset_type": "draft",
            "selected_asset_id": second_draft["id"],
            "note": "Explicitly reuse a reviewed historical version",
        },
    )
    assert reused.status_code == 200, reused.text
    refreshed_other = client.post(
        f"/api/application-packages/{other_program['id']}/refresh"
    ).json()
    refreshed_other_ps = next(
        item for item in refreshed_other["checklist"] if item["category"] == "ps"
    )
    assert refreshed_other_ps["selected_asset_id"] == second_draft["id"]
    assert refreshed_other_ps["status"] == "ready"
    monkeypatch.setattr(provider, "generate_material", fake_generate)
    other_conversation = client.post(
        "/api/writing-conversations",
        json={
            "program_id": other_program["id"],
            "slot_key": other_ps["slot_key"],
            "material_kind": "ps",
            "title": "Cross-project reuse",
            "resource_ids": [
                "profile",
                "confirmed_experiences",
                "official_requirements",
                f"draft:{second_draft['id']}",
            ],
        },
    ).json()
    cross = client.post(
        f"/api/writing-conversations/{other_conversation['id']}/messages",
        json={"message": "请参考所选历史版本生成一篇完整的 PS 正文"},
    )
    assert cross.status_code == 200, cross.text
    cross_draft = cross.json()["latest_draft"]
    assert cross_draft["derived_from_id"] == second_draft["id"]
    assert cross_draft["parent_id"] is None
    assert second_draft["id"] in cross_draft["source_refs"]["reference_draft_ids"]
    cross_manifest = client.get(
        f"/api/context-manifests/{cross_draft['context_manifest_id']}"
    ).json()
    assert cross_manifest["draft_refs"][0]["id"] == second_draft["id"]


async def test_independent_verifier_blocks_bad_draft_and_retries_revisable_draft(
    client, monkeypatch
):
    experience_id = _confirmed_profile(client)
    program = client.get("/api/programs?personalized=false").json()[0]
    await _set_verified_ps_requirement(program["id"], program["official_url"])
    client.post(
        "/api/shortlists",
        json={"name": "P0-C verifier", "program_ids": [program["id"]]},
    )
    package = client.post(
        f"/api/application-packages/{program['id']}/refresh"
    ).json()
    ps_row = next(item for item in package["checklist"] if item["category"] == "ps")

    async def new_conversation(title):
        return client.post(
            "/api/writing-conversations",
            json={
                "program_id": program["id"],
                "slot_key": ps_row["slot_key"],
                "material_kind": "ps",
                "title": title,
                "resource_ids": [
                    "profile",
                    "confirmed_experiences",
                    "official_requirements",
                ],
            },
        ).json()

    generation_calls = []

    async def fake_generate(payload):
        generation_calls.append(payload)
        return {
            "response_type": "draft",
            "message": "candidate",
            "title": "Verified draft",
            "content": f"Grounded statement version {len(generation_calls)}.",
            "source_experience_ids": [experience_id],
            "warnings": [],
            "model_info": {"provider": "test", "model": "writer"},
        }

    async def reject_verify(payload, candidate):
        return {
            "verdict": "rejected",
            "summary": "Unsupported claim.",
            "requirement_coverage": [],
            "unsupported_claims": ["invented claim"],
            "conflicts": [],
            "issues": ["unsupported_claim"],
            "confidence": 1.0,
            "mode": "test-verifier",
        }

    monkeypatch.setattr(provider, "generate_material", fake_generate)
    monkeypatch.setattr(provider, "verify_material", reject_verify)
    async with SessionLocal() as session:
        baseline_draft_ids = set(
            (
                await session.scalars(
                    select(MaterialDraft.id).where(
                        MaterialDraft.program_id == program["id"],
                        MaterialDraft.slot_key == ps_row["slot_key"],
                    )
                )
            ).all()
        )
    rejected_conversation = await new_conversation("Rejected candidate")
    rejected = client.post(
        f"/api/writing-conversations/{rejected_conversation['id']}/messages",
        json={"message": "请生成一篇完整的 PS 正文"},
    )
    assert rejected.status_code == 422
    assert "独立验证" in rejected.json()["detail"]
    async with SessionLocal() as session:
        rejected_draft_ids = set(
            (
                await session.scalars(
                    select(MaterialDraft.id).where(
                        MaterialDraft.program_id == program["id"],
                        MaterialDraft.slot_key == ps_row["slot_key"],
                    )
                )
            ).all()
        )
        assert rejected_draft_ids == baseline_draft_ids
        failed_run = await session.scalar(
            select(AgentRun).where(
                AgentRun.conversation_id == rejected_conversation["id"]
            )
        )
        assert failed_run.status == "failed"
        assert failed_run.stop_reason == "independent_verification_failed"

    generation_calls.clear()
    verification_calls = []

    async def revise_then_accept(payload, candidate):
        verification_calls.append(candidate)
        verdict = "revise" if len(verification_calls) == 1 else "accepted"
        return {
            "verdict": verdict,
            "summary": "Revise once." if verdict == "revise" else "Accepted.",
            "requirement_coverage": ["preparation", "goals"],
            "unsupported_claims": [],
            "conflicts": [],
            "issues": ["clarify_goal"] if verdict == "revise" else [],
            "confidence": 0.95,
            "mode": "test-verifier",
        }

    monkeypatch.setattr(provider, "verify_material", revise_then_accept)
    retried_conversation = await new_conversation("Retried candidate")
    retried = client.post(
        f"/api/writing-conversations/{retried_conversation['id']}/messages",
        json={"message": "请生成一篇完整的 PS 正文"},
    )
    assert retried.status_code == 200, retried.text
    draft = retried.json()["latest_draft"]
    assert len(generation_calls) == 2
    assert len(verification_calls) == 2
    assert draft["content"].endswith("version 2.")
    assert draft["model_info"]["verification"]["verdict"] == "accepted"
    trace = client.get(
        f"/api/agent-runs/{draft['model_info']['agent_run_id']}/trace"
    ).json()
    verifier_step = next(
        item
        for item in trace["steps"]
        if item["name"] == "独立复核要求覆盖与事实来源"
    )
    assert verifier_step["status"] == "accepted"
    assert verifier_step["result"]["attempt_count"] == 2


async def test_long_material_conversation_persists_summary_and_recent_window(
    client, monkeypatch
):
    experience_id = _confirmed_profile(client)
    program = client.get("/api/programs?personalized=false").json()[0]
    await _set_verified_ps_requirement(program["id"], program["official_url"])
    client.post(
        "/api/shortlists",
        json={"name": "P0-D memory", "program_ids": [program["id"]]},
    )
    package = client.post(
        f"/api/application-packages/{program['id']}/refresh"
    ).json()
    ps_row = next(item for item in package["checklist"] if item["category"] == "ps")
    conversation = client.post(
        "/api/writing-conversations",
        json={
            "program_id": program["id"],
            "slot_key": ps_row["slot_key"],
            "material_kind": "ps",
            "title": "Long memory test",
            "resource_ids": [
                "profile",
                "confirmed_experiences",
                "official_requirements",
            ],
        },
    ).json()
    async with SessionLocal() as session:
        for index in range(14):
            session.add(Message(
                owner_id=settings.local_owner_id,
                conversation_id=conversation["id"],
                role="user" if index % 2 == 0 else "assistant",
                content=(
                    "必须保留检索项目经历作为文书主线。"
                    if index == 0
                    else f"historical message {index}"
                ),
            ))
        await session.commit()

    seen_payloads = []

    async def fake_generate(payload):
        seen_payloads.append(payload)
        return {
            "response_type": "draft",
            "message": "candidate",
            "title": "Long-memory draft",
            "content": "I built a grounded retrieval project and will continue this work.",
            "source_experience_ids": [experience_id],
            "warnings": [],
            "model_info": {"provider": "test", "model": "writer"},
        }

    async def accepted_verify(payload, candidate):
        return {
            "verdict": "accepted",
            "summary": "Memory and sources are grounded.",
            "requirement_coverage": ["preparation", "goals"],
            "unsupported_claims": [],
            "conflicts": [],
            "issues": [],
            "confidence": 1.0,
            "mode": "test-verifier",
        }

    monkeypatch.setattr(provider, "generate_material", fake_generate)
    monkeypatch.setattr(provider, "verify_material", accepted_verify)
    response = client.post(
        f"/api/writing-conversations/{conversation['id']}/messages",
        json={"message": "请生成一篇完整的 PS 正文"},
    )
    assert response.status_code == 200, response.text
    memory_state = response.json()["memory_state"]
    assert memory_state["total_message_count"] == 15
    assert memory_state["summarized_message_count"] == 3
    assert "必须保留检索项目经历" in memory_state["stable_summary"]
    assert len(seen_payloads[0]["conversation_history"]) == 12
    assert seen_payloads[0]["conversation_memory"]["decisions"]
    draft = response.json()["latest_draft"]
    manifest = client.get(
        f"/api/context-manifests/{draft['context_manifest_id']}"
    ).json()
    assert manifest["conversation_ref"]["message_count"] == 15
    assert manifest["conversation_ref"]["recent_message_count"] == 12
    assert manifest["conversation_ref"]["summarized_message_count"] == 3
    assert manifest["conversation_ref"]["stable_summary_hash"]
