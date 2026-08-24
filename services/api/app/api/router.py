import asyncio
import json
import re
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.conversation_memory import build_conversation_memory
from app.agent.goals import parse_material_goal
from app.agent.guardrails import check_user_input, verify_output
from app.agent.harness import harness
from app.agent.material_context import (
    build_context_manifest_data,
    content_hash,
    validate_context_manifest,
)
from app.agent.material_workflow import (
    set_material_step,
    start_material_run,
    sync_material_plan,
)
from app.agent.mcp import demo_mcp
from app.agent.memory import persist_confirmed_memory
from app.agent.program_recommendation_workflow import run_online_program_recommendation
from app.agent.provider import provider
from app.agent.skills import skill_registry
from app.agent.tools import tool_registry
from app.agent.verifier import verify_material_candidate
from app.core.config import settings
from app.core.database import get_session
from app.models.entities import (
    AgentRun,
    AgentStep,
    ApplicantProfile,
    Application,
    ApplicationPackage,
    ContextManifest,
    Conversation,
    Document,
    EvidenceChunk,
    Experience,
    MaterialArtifact,
    MaterialDraft,
    MaterialPlan,
    MCPConnection,
    Memory,
    Message,
    Program,
    ProgramSource,
    RecommendationGoal,
    Shortlist,
    ShortlistItem,
    Task,
    ToolCall,
)
from app.schemas.api import (
    AgentRunResponse,
    ApplicationCreate,
    ApplicationPackageResponse,
    ApplicationResponse,
    ApplicationUpdate,
    AssistantConversationCreate,
    AssistantConversationResponse,
    AssistantConversationUpdate,
    BatchVerifyResponse,
    ChatRequest,
    ContextManifestResponse,
    DocumentConfirmRequest,
    DocumentResponse,
    EvidenceResponse,
    HealthResponse,
    MaterialArtifactCreate,
    MaterialArtifactResponse,
    MaterialArtifactUpdate,
    MaterialAssetPreviewResponse,
    MaterialDraftGenerate,
    MaterialDraftResponse,
    MaterialDraftUpdate,
    MaterialPlanCreate,
    MaterialPlanResponse,
    MaterialPreflightRequest,
    MaterialPreflightResponse,
    MCPServerResponse,
    MemoryResponse,
    PackageMaterialUpdate,
    ProfileResponse,
    ProfileUpdate,
    ProgramRecommendationResponse,
    ProgramResponse,
    ProgramVerifyResponse,
    RecommendationGoalResponse,
    RecommendationGoalUpdate,
    RequirementResponse,
    ShortlistCreate,
    ShortlistItemResponse,
    ShortlistItemsUpdate,
    ShortlistResponse,
    SkillResponse,
    SourceResponse,
    TaskCreate,
    TaskResponse,
    TaskUpdate,
    TimelineCreate,
    TraceResponse,
    UniversityOptionResponse,
    WritingConversationCreate,
    WritingConversationResponse,
    WritingConversationUpdate,
    WritingMessageCreate,
    WritingMessageResponse,
)
from app.services.business import (
    add_shortlist_programs,
    consolidate_shortlists,
    create_material_plan,
    create_shortlist,
    create_task,
    create_timeline,
    get_or_create_application_package,
    get_or_create_recommendation_goal,
    get_program,
    get_requirement,
    material_slots,
    profile_with_experiences,
    refresh_application_package,
    remove_shortlist_program,
    score_recommendation,
    search_programs_for_profile,
    university_options,
    update_profile,
    update_recommendation_goal,
)
from app.services.documents import (
    SUPPORTED_MIME_TYPES,
    extract_text,
    infer_document_data,
    sha256_bytes,
)
from app.services.material_exports import (
    content_disposition,
    docx_export,
    export_filename,
    pdf_export,
)
from app.services.requirements import verify_program_official
from app.services.web import UnsafeUrlError

router = APIRouter()
active_writing_tasks: Dict[str, asyncio.Task] = {}
active_chat_tasks: Dict[str, asyncio.Task] = {}


def profile_response(profile: Any, experiences: List[Any]) -> ProfileResponse:
    data = ProfileResponse.model_validate(profile).model_dump()
    data["experiences"] = [
        {
            "id": item.id,
            "kind": item.kind,
            "title": item.title,
            "organization": item.organization,
            "start_date": item.start_date,
            "end_date": item.end_date,
            "description": item.description,
            "tags": item.tags,
            "confirmed": item.confirmed,
        }
        for item in experiences
    ]
    return ProfileResponse(**data)


async def serialize_program(session: AsyncSession, program: Program) -> ProgramResponse:
    requirement = await get_requirement(session, program.id)
    sources = list(
        (
            await session.scalars(
                select(ProgramSource).where(ProgramSource.program_id == program.id)
            )
        ).all()
    )
    evidence = list(
        (
            await session.scalars(
                select(EvidenceChunk).where(EvidenceChunk.program_id == program.id)
            )
        ).all()
    )
    payload = ProgramResponse.model_validate(program).model_dump()
    payload["requirement"] = (
        RequirementResponse.model_validate(requirement) if requirement else None
    )
    payload["sources"] = [SourceResponse.model_validate(item) for item in sources]
    payload["evidence"] = [EvidenceResponse.model_validate(item) for item in evidence]
    return ProgramResponse(**payload)


@router.get("/health", response_model=HealthResponse)
async def health(session: AsyncSession = Depends(get_session)) -> HealthResponse:
    await session.scalar(select(1))
    return HealthResponse(
        status="ok",
        database="ok",
        llm_mode=provider.mode,
        version="0.1.0",
    )


@router.get("/profile", response_model=ProfileResponse)
async def get_profile_route(session: AsyncSession = Depends(get_session)) -> ProfileResponse:
    profile, experiences = await profile_with_experiences(session)
    return profile_response(profile, experiences)


@router.put("/profile", response_model=ProfileResponse)
async def put_profile_route(
    payload: ProfileUpdate, session: AsyncSession = Depends(get_session)
) -> ProfileResponse:
    profile, experiences = await update_profile(session, payload)
    if payload.confirmed:
        await persist_confirmed_memory(
            session,
            key="applicant_profile",
            value=payload.model_dump(exclude={"experiences"}),
            source_type="user_confirmed",
            source_id=profile.id,
        )
    return profile_response(profile, experiences)


@router.get("/recommendation-goal", response_model=RecommendationGoalResponse)
async def get_recommendation_goal(
    session: AsyncSession = Depends(get_session),
) -> RecommendationGoalResponse:
    goal = await get_or_create_recommendation_goal(session)
    return RecommendationGoalResponse.model_validate(goal)


@router.put("/recommendation-goal", response_model=RecommendationGoalResponse)
async def put_recommendation_goal(
    payload: RecommendationGoalUpdate,
    session: AsyncSession = Depends(get_session),
) -> RecommendationGoalResponse:
    goal = await update_recommendation_goal(session, payload)
    await persist_confirmed_memory(
        session,
        key="recommendation_goal",
        value=payload.model_dump(),
        source_type="user_confirmed",
        source_id=goal.id,
    )
    return RecommendationGoalResponse.model_validate(goal)


@router.get("/profile/export")
async def export_profile(session: AsyncSession = Depends(get_session)) -> JSONResponse:
    profile, experiences = await profile_with_experiences(session)
    memories = list(
        (
            await session.scalars(
                select(Memory).where(
                    Memory.owner_id == settings.local_owner_id,
                    Memory.active.is_(True),
                )
            )
        ).all()
    )
    payload = {
        "export_version": "1.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile_response(profile, experiences).model_dump(mode="json"),
        "memories": [MemoryResponse.model_validate(item).model_dump(mode="json") for item in memories],
    }
    return JSONResponse(
        payload,
        headers={"Content-Disposition": 'attachment; filename="yyglobal-profile.json"'},
    )


@router.delete("/profile", status_code=204)
async def clear_profile(
    confirm: str = Query(""), session: AsyncSession = Depends(get_session)
) -> None:
    if confirm != "DELETE_MY_P0_DATA":
        raise HTTPException(400, "必须提供 confirm=DELETE_MY_P0_DATA")

    document_paths = list(
        (
            await session.scalars(
                select(Document.path).where(Document.owner_id == settings.local_owner_id)
            )
        ).all()
    )
    run_ids = list(
        (
            await session.scalars(
                select(AgentRun.id).where(AgentRun.owner_id == settings.local_owner_id)
            )
        ).all()
    )
    shortlist_ids = list(
        (
            await session.scalars(
                select(Shortlist.id).where(Shortlist.owner_id == settings.local_owner_id)
            )
        ).all()
    )
    conversation_ids = list(
        (
            await session.scalars(
                select(Conversation.id).where(Conversation.owner_id == settings.local_owner_id)
            )
        ).all()
    )
    if run_ids:
        await session.execute(delete(ToolCall).where(ToolCall.run_id.in_(run_ids)))
        await session.execute(delete(AgentStep).where(AgentStep.run_id.in_(run_ids)))
    if shortlist_ids:
        await session.execute(
            delete(ShortlistItem).where(ShortlistItem.shortlist_id.in_(shortlist_ids))
        )
    if conversation_ids:
        await session.execute(
            delete(Message).where(Message.conversation_id.in_(conversation_ids))
        )
    for model in (
        Application,
        ApplicationPackage,
        MaterialArtifact,
        MaterialDraft,
        MaterialPlan,
        Task,
        Memory,
        AgentRun,
        Conversation,
        Shortlist,
        Experience,
        Document,
        RecommendationGoal,
        ApplicantProfile,
    ):
        await session.execute(delete(model).where(model.owner_id == settings.local_owner_id))
    await session.commit()
    for value in document_paths:
        path = Path(value)
        if path.is_file() and settings.upload_dir in path.parents:
            path.unlink(missing_ok=True)


@router.post("/documents", response_model=DocumentResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    kind: str = Form("other"),
    session: AsyncSession = Depends(get_session),
) -> DocumentResponse:
    mime_type = file.content_type or "application/octet-stream"
    if mime_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(415, "仅支持 PDF、DOCX、TXT、Markdown、PNG 和 JPEG")
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"文件不能超过 {settings.max_upload_mb}MB")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload").suffix.lower()[:10]
    path = settings.upload_dir / f"{uuid4().hex}{suffix}"
    path.write_bytes(content)
    text, status = extract_text(path, mime_type)
    extracted_data = infer_document_data(text, kind)
    if provider.available:
        try:
            extracted_data = await provider.extract_document(path, mime_type, kind, text)
            if not text.strip():
                text = str(extracted_data.get("summary") or "").strip()
            status = (
                "parsed_multimodal"
                if mime_type.startswith("image/") or not text.strip()
                else "parsed_ai"
            )
        except Exception as exc:
            extracted_data["ai_extraction_error"] = str(exc)[:300]
            if not text.strip():
                status = "needs_multimodal_review"
    item = Document(
        owner_id=settings.local_owner_id,
        filename=Path(file.filename or "upload").name[:255],
        mime_type=mime_type,
        kind=kind,
        path=str(path),
        sha256=sha256_bytes(content),
        parse_status=status,
        extracted_text=text[:200_000],
        extracted_data=extracted_data,
    )
    session.add(item)
    await session.commit()
    await session.refresh(item)
    return DocumentResponse.model_validate(item)


@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    session: AsyncSession = Depends(get_session),
) -> List[DocumentResponse]:
    items = list((await session.scalars(
        select(Document).where(
            Document.owner_id == settings.local_owner_id
        ).order_by(Document.created_at.desc())
    )).all())
    return [DocumentResponse.model_validate(item) for item in items]


@router.get("/documents/{document_id}/download")
async def download_document(
    document_id: str,
    session: AsyncSession = Depends(get_session),
) -> Response:
    item = await session.get(Document, document_id)
    if not item or item.owner_id != settings.local_owner_id:
        raise HTTPException(404, "材料不存在")
    path = Path(item.path)
    if not path.is_file():
        raise HTTPException(404, "材料文件不存在")
    return Response(
        content=path.read_bytes(),
        media_type=item.mime_type,
        headers={"Content-Disposition": content_disposition(item.filename)},
    )


@router.get("/documents/{document_id}/content")
async def view_document_content(
    document_id: str,
    session: AsyncSession = Depends(get_session),
) -> Response:
    item = await session.get(Document, document_id)
    if not item or item.owner_id != settings.local_owner_id:
        raise HTTPException(404, "材料不存在")
    path = Path(item.path)
    if not path.is_file():
        raise HTTPException(404, "材料文件不存在")
    return Response(content=path.read_bytes(), media_type=item.mime_type)


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    session: AsyncSession = Depends(get_session),
) -> Response:
    item = await session.get(Document, document_id)
    if not item or item.owner_id != settings.local_owner_id:
        raise HTTPException(404, "材料不存在")
    path = Path(item.path)
    await session.execute(delete(MaterialArtifact).where(
        MaterialArtifact.owner_id == settings.local_owner_id,
        MaterialArtifact.document_id == item.id,
    ))
    await session.delete(item)
    await session.commit()
    if path.is_file() and settings.upload_dir in path.parents:
        path.unlink(missing_ok=True)
    return Response(status_code=204)


@router.post("/documents/{document_id}/confirm", response_model=ProfileResponse)
async def confirm_document_candidates(
    document_id: str,
    payload: DocumentConfirmRequest,
    session: AsyncSession = Depends(get_session),
) -> ProfileResponse:
    document = await session.get(Document, document_id)
    if not document or document.owner_id != settings.local_owner_id:
        raise HTTPException(404, "材料不存在")
    allowed = {"gpa", "gpa_scale", "language_scores"}
    unknown = set(payload.accepted_fields) - allowed
    if unknown:
        raise HTTPException(400, f"不支持自动写入的候选字段：{', '.join(sorted(unknown))}")
    profile, experiences = await profile_with_experiences(session)
    candidates = document.extracted_data or {}
    by_field = {
        item.get("field"): item for item in candidates.get("candidate_facts", [])
        if isinstance(item, dict) and item.get("field")
    }
    accepted: Dict[str, Any] = {}
    if "gpa" in payload.accepted_fields and "gpa" in by_field:
        raw = by_field["gpa"].get("value")
        match = re.search(r"\d+(?:\.\d+)?", str(raw))
        if match:
            profile.gpa = float(match.group())
            accepted["gpa"] = profile.gpa
    if "gpa_scale" in payload.accepted_fields and "gpa" in by_field:
        raw = f"{by_field['gpa'].get('value', '')}/{by_field['gpa'].get('scale', '')}"
        numbers = re.findall(r"\d+(?:\.\d+)?", raw)
        if len(numbers) >= 2:
            profile.gpa_scale = float(numbers[1])
            accepted["gpa_scale"] = profile.gpa_scale
    if "language_scores" in payload.accepted_fields and candidates.get("language_scores"):
        profile.language_scores = candidates["language_scores"]
        accepted["language_scores"] = profile.language_scores
    if not accepted:
        raise HTTPException(400, "所选字段没有可确认的候选值")
    await session.commit()
    await persist_confirmed_memory(
        session,
        key=f"document_candidates:{document.id}",
        value={"accepted": accepted, "filename": document.filename},
        source_type="document_user_confirmed",
        source_id=document.id,
    )
    profile, experiences = await profile_with_experiences(session)
    return profile_response(profile, experiences)


@router.get("/programs", response_model=List[ProgramResponse])
async def list_programs(
    q: str = Query("", max_length=200),
    country: str = Query("", max_length=100),
    field: str = Query("", max_length=160),
    personalized: bool = Query(True),
    session: AsyncSession = Depends(get_session),
) -> List[ProgramResponse]:
    items = await search_programs_for_profile(
        session, q, country, field, use_profile=personalized
    )
    return [await serialize_program(session, item) for item in items]


@router.get("/programs/universities", response_model=List[UniversityOptionResponse])
async def list_university_options(
    countries: List[str] = Query(default_factory=list),
    max_qs_rank: Optional[int] = Query(None, ge=1, le=300),
    session: AsyncSession = Depends(get_session),
) -> List[UniversityOptionResponse]:
    return [
        UniversityOptionResponse(**item)
        for item in await university_options(session, countries, max_qs_rank)
    ]


@router.post(
    "/programs/recommendations",
    response_model=List[ProgramRecommendationResponse],
)
async def recommend_programs(
    q: str = Query("", max_length=200),
    limit: int = Query(5, ge=1, le=5),
    exclude_ids: str = Query("", max_length=10000),
    session: AsyncSession = Depends(get_session),
) -> List[ProgramRecommendationResponse]:
    profile, _ = await profile_with_experiences(session)
    goal = await get_or_create_recommendation_goal(session)
    if not q and (
        not profile.confirmed
        or not goal.target_fields
        or (not goal.target_countries and not goal.target_university)
    ):
        raise HTTPException(400, "请先完成画像，再开始项目推荐")

    excluded_program_ids = {
        item.strip() for item in exclude_ids.split(",") if item.strip()
    }
    try:
        programs, _ = await run_online_program_recommendation(
            session, goal, excluded_program_ids=excluded_program_ids, limit=limit
        )
    except TimeoutError as exc:
        raise HTTPException(
            504,
            "官网目录定位超时，本次不会重复执行完整检索，请稍后重试。",
        ) from exc
    output = []
    refreshed_profile, experiences = await profile_with_experiences(session)
    refreshed_goal = await get_or_create_recommendation_goal(session)
    for program in programs:
        requirement = await get_requirement(session, program.id)
        score, reasons = await score_recommendation(
            refreshed_profile, experiences, program, requirement, refreshed_goal
        )
        output.append(
            ProgramRecommendationResponse(
                program=await serialize_program(session, program),
                score=score,
                reasons=reasons,
                verification_status="verified",
                verification_error="",
            )
        )
    output.sort(key=lambda item: (-item.score, item.program.university, item.program.name))
    return output


@router.get("/programs/{program_id}", response_model=ProgramResponse)
async def program_detail(
    program_id: str, session: AsyncSession = Depends(get_session)
) -> ProgramResponse:
    item = await get_program(session, program_id)
    if not item:
        raise HTTPException(404, "项目不存在")
    return await serialize_program(session, item)


@router.post("/programs/{program_id}/verify", response_model=ProgramVerifyResponse)
async def verify_program(
    program_id: str, session: AsyncSession = Depends(get_session)
) -> ProgramVerifyResponse:
    program = await get_program(session, program_id)
    if not program:
        raise HTTPException(404, "项目不存在")
    try:
        source, extracted = await verify_program_official(session, program)
    except (UnsafeUrlError, Exception) as exc:
        raise HTTPException(502, f"官网读取失败：{exc}") from exc
    requirement = await get_requirement(session, program.id)
    return ProgramVerifyResponse(
        program_id=program.id,
        source_id=source.id,
        status="verified" if requirement and requirement.verified else "fetched_needs_review",
        fetched_at=source.fetched_at,
        content_hash=source.content_hash,
        extracted=extracted,
    )


@router.post("/programs/verify-matched", response_model=BatchVerifyResponse)
async def verify_profile_matched_programs(
    limit: int = Query(5, ge=1, le=10),
    session: AsyncSession = Depends(get_session),
) -> BatchVerifyResponse:
    profile, _ = await profile_with_experiences(session)
    goal = await get_or_create_recommendation_goal(session)
    if not profile.confirmed or not goal.target_fields or not goal.target_countries:
        raise HTTPException(400, "请先确认画像，并在项目推荐中填写选校目标")
    matched = await search_programs_for_profile(session, use_profile=True)
    results: List[Dict[str, Any]] = []
    for program in matched[:limit]:
        try:
            source, extracted = await verify_program_official(session, program)
            requirement = await get_requirement(session, program.id)
            results.append({
                "program_id": program.id,
                "university": program.university,
                "name": program.name,
                "status": "verified" if requirement and requirement.verified else "fetched_needs_review",
                "source_id": source.id,
                "evidence_count": len(extracted.get("evidence", [])),
                "error": "",
            })
        except Exception as exc:
            results.append({
                "program_id": program.id,
                "university": program.university,
                "name": program.name,
                "status": "failed",
                "source_id": "",
                "evidence_count": 0,
                "error": str(exc)[:300],
            })
    return BatchVerifyResponse(
        matched_count=len(matched),
        attempted_count=len(results),
        verified_count=sum(item["status"] == "verified" for item in results),
        needs_review_count=sum(item["status"] == "fetched_needs_review" for item in results),
        failed_count=sum(item["status"] == "failed" for item in results),
        results=results,
    )


async def serialize_shortlist(session: AsyncSession, item: Shortlist) -> ShortlistResponse:
    rows = list(
        (
            await session.scalars(
                select(ShortlistItem)
                .where(ShortlistItem.shortlist_id == item.id)
                .order_by(ShortlistItem.score.desc())
            )
        ).all()
    )
    output = []
    for row in rows:
        program = await session.get(Program, row.program_id)
        if program:
            output.append(
                ShortlistItemResponse(
                    id=row.id,
                    program=await serialize_program(session, program),
                    tier=row.tier,
                    score=row.score,
                    rationale=row.rationale,
                    risks=row.risks,
                )
            )
    return ShortlistResponse(
        id=item.id,
        name=item.name,
        rationale=item.rationale,
        items=output,
        created_at=item.created_at,
    )


@router.get("/shortlists", response_model=List[ShortlistResponse])
async def list_shortlists(session: AsyncSession = Depends(get_session)) -> List[ShortlistResponse]:
    await consolidate_shortlists(session)
    items = list(
        (
            await session.scalars(
                select(Shortlist)
                .where(Shortlist.owner_id == settings.local_owner_id)
                .order_by(Shortlist.created_at.desc())
            )
        ).all()
    )
    return [await serialize_shortlist(session, item) for item in items]


@router.post("/shortlists", response_model=ShortlistResponse, status_code=201)
async def post_shortlist(
    payload: ShortlistCreate, session: AsyncSession = Depends(get_session)
) -> ShortlistResponse:
    item = await create_shortlist(session, payload.name, payload.program_ids)
    return await serialize_shortlist(session, item)


@router.post("/shortlists/items", response_model=ShortlistResponse)
async def add_shortlist_items(
    payload: ShortlistItemsUpdate,
    session: AsyncSession = Depends(get_session),
) -> ShortlistResponse:
    item = await add_shortlist_programs(session, payload.program_ids)
    return await serialize_shortlist(session, item)


@router.delete("/shortlists/{shortlist_id}/items/{program_id}", status_code=204)
async def delete_shortlist_item(
    shortlist_id: str,
    program_id: str,
    session: AsyncSession = Depends(get_session),
) -> None:
    if not await remove_shortlist_program(session, shortlist_id, program_id):
        raise HTTPException(404, "选校清单中没有该项目")


async def serialize_application_package(
    session: AsyncSession, item: ApplicationPackage
) -> ApplicationPackageResponse:
    program = await get_program(session, item.program_id)
    if not program:
        raise HTTPException(409, "申请包关联的项目不存在")
    return ApplicationPackageResponse(
        id=item.id,
        program=await serialize_program(session, program),
        shortlist_id=item.shortlist_id,
        official_verified=item.official_verified,
        checklist=item.checklist,
        gaps=item.gaps,
        ready=item.ready,
        plan_confirmed=item.plan_confirmed,
        status=item.status,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


@router.get("/application-packages", response_model=List[ApplicationPackageResponse])
async def list_application_packages(
    session: AsyncSession = Depends(get_session),
) -> List[ApplicationPackageResponse]:
    items = list((await session.scalars(
        select(ApplicationPackage).where(
            ApplicationPackage.owner_id == settings.local_owner_id
        ).order_by(ApplicationPackage.updated_at.desc())
    )).all())
    output = []
    for item in items:
        item = await get_or_create_application_package(session, item.program_id, item.shortlist_id)
        output.append(await serialize_application_package(session, item))
    await session.commit()
    return output


@router.post(
    "/application-packages/{program_id}/refresh",
    response_model=ApplicationPackageResponse,
)
async def refresh_package(
    program_id: str, session: AsyncSession = Depends(get_session)
) -> ApplicationPackageResponse:
    if not await get_program(session, program_id):
        raise HTTPException(404, "项目不存在")
    item = await get_or_create_application_package(session, program_id)
    await session.commit()
    await session.refresh(item)
    return await serialize_application_package(session, item)


@router.patch(
    "/application-packages/{package_id}/materials",
    response_model=ApplicationPackageResponse,
)
async def update_package_material(
    package_id: str,
    payload: PackageMaterialUpdate,
    session: AsyncSession = Depends(get_session),
) -> ApplicationPackageResponse:
    item = await session.get(ApplicationPackage, package_id)
    if not item or item.owner_id != settings.local_owner_id:
        raise HTTPException(404, "项目申请包不存在")
    allowed = {"ready", "needs_edit", "unverified", "missing", "manual_review"}
    if payload.status not in allowed:
        raise HTTPException(422, "不支持的材料适配状态")
    await refresh_application_package(session, item)
    # Rebuild every JSON row so SQLAlchemy reliably persists nested changes.
    checklist = [dict(row) for row in (item.checklist or [])]
    target = next(
        (row for row in checklist if row.get("material_key") == payload.material_key), None
    )
    if not target:
        raise HTTPException(404, "申请包中没有该材料要求")
    candidates = target.get("candidate_assets", [])
    if payload.status == "ready":
        selected = next((
            row
            for row in candidates
            if row.get("type") == payload.selected_asset_type
            and row.get("id") == payload.selected_asset_id
        ), None)
        valid = selected is not None
        if selected and selected.get("type") == "draft":
            valid = selected.get("status") == "reviewed"
        if selected and selected.get("type") == "artifact":
            valid = selected.get("status") in {"ready", "submitted"}
        if not valid:
            raise HTTPException(422, "所选材料不存在、尚未确认，或不属于当前项目")
    target.update({
        "status": payload.status,
        "selected_asset_type": payload.selected_asset_type,
        "selected_asset_id": payload.selected_asset_id,
        "note": payload.note,
    })
    item.checklist = [dict(row) for row in checklist]
    item.plan_confirmed = False
    item.ready = False
    item.status = "materials_in_progress"
    item.gaps = [
        f"待处理：{row.get('name')}" for row in checklist if row.get("status") != "ready"
    ]
    if payload.status == "ready" and payload.selected_asset_type == "draft":
        selected_draft = await session.get(MaterialDraft, payload.selected_asset_id)
        if selected_draft and selected_draft.owner_id == settings.local_owner_id:
            run_id = str((selected_draft.model_info or {}).get("agent_run_id") or "")
            if run_id:
                run = await session.get(AgentRun, run_id)
                run_steps = list(
                    (
                        await session.scalars(
                            select(AgentStep)
                            .where(AgentStep.run_id == run_id)
                            .order_by(AgentStep.position)
                        )
                    ).all()
                )
                if run and run_steps:
                    set_material_step(
                        run_steps,
                        "adopt_version",
                        "accepted",
                        {
                            "draft_id": selected_draft.id,
                            "package_id": item.id,
                            "material_key": payload.material_key,
                            "confirmed_by_user": True,
                        },
                    )
                    sync_material_plan(run, run_steps)
                    run.stop_reason = "adopted"
            if selected_draft.context_manifest_id:
                manifest = await session.get(
                    ContextManifest, selected_draft.context_manifest_id
                )
                if manifest:
                    manifest.status = "adopted"
    await session.commit()
    await session.refresh(item)
    return await serialize_application_package(session, item)


@router.post(
    "/application-packages/{package_id}/confirm-plan",
    response_model=ApplicationPackageResponse,
)
async def confirm_package_plan(
    package_id: str,
    session: AsyncSession = Depends(get_session),
) -> ApplicationPackageResponse:
    item = await session.get(ApplicationPackage, package_id)
    if not item or item.owner_id != settings.local_owner_id:
        raise HTTPException(404, "项目申请包不存在")
    await refresh_application_package(session, item)
    checklist = [dict(row) for row in (item.checklist or [])]
    if not checklist:
        raise HTTPException(422, "当前项目还没有可确认的官网材料清单")
    unresolved = [
        row.get("name")
        for row in checklist
        if row.get("status") != "ready" or not row.get("selected_asset_id")
    ]
    if unresolved:
        raise HTTPException(422, f"仍有材料没有完成准备：{'、'.join(str(value) for value in unresolved[:6])}")
    item.plan_confirmed = True
    item.ready = True
    item.status = "ready"
    await session.commit()
    await session.refresh(item)
    return await serialize_application_package(session, item)


@router.get("/material-assets/{asset_type}/{asset_id}/preview", response_model=MaterialAssetPreviewResponse)
async def preview_material_asset(
    asset_type: str,
    asset_id: str,
    session: AsyncSession = Depends(get_session),
) -> MaterialAssetPreviewResponse:
    if asset_type == "artifact":
        artifact = await session.get(MaterialArtifact, asset_id)
        if not artifact or artifact.owner_id != settings.local_owner_id:
            raise HTTPException(404, "材料版本不存在")
        asset_id = artifact.document_id
        asset_type = "document"
    if asset_type == "document":
        item = await session.get(Document, asset_id)
        if not item or item.owner_id != settings.local_owner_id:
            raise HTTPException(404, "材料不存在")
        return MaterialAssetPreviewResponse(
            title=item.filename,
            kind=item.kind,
            mime_type=item.mime_type,
            content=item.extracted_text,
            raw_url=f"/api/documents/{item.id}/content",
        )
    if asset_type == "draft":
        item = await session.get(MaterialDraft, asset_id)
        if not item or item.owner_id != settings.local_owner_id:
            raise HTTPException(404, "文稿版本不存在")
        return MaterialAssetPreviewResponse(
            title=f"{item.title} v{item.version_number}",
            kind=item.kind,
            mime_type="text/markdown",
            content=item.content,
        )
    raise HTTPException(422, "不支持的材料资源类型")


@router.get("/material-plans", response_model=List[MaterialPlanResponse])
async def list_material_plans(
    session: AsyncSession = Depends(get_session),
) -> List[MaterialPlanResponse]:
    items = list(
        (
            await session.scalars(
                select(MaterialPlan)
                .where(MaterialPlan.owner_id == settings.local_owner_id)
                .order_by(MaterialPlan.created_at.desc())
            )
        ).all()
    )
    return [MaterialPlanResponse.model_validate(item) for item in items]


@router.post("/material-plans", response_model=MaterialPlanResponse, status_code=201)
async def post_material_plan(
    payload: MaterialPlanCreate, session: AsyncSession = Depends(get_session)
) -> MaterialPlanResponse:
    try:
        item = await create_material_plan(session, payload.program_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return MaterialPlanResponse.model_validate(item)


MATERIAL_KINDS = {"cv", "ps"}
MATERIAL_SCOPES = {"general", "program"}
MATERIAL_STATUSES = {"draft", "ready", "submitted"}


async def serialize_artifact(
    session: AsyncSession, item: MaterialArtifact
) -> MaterialArtifactResponse:
    document = await session.get(Document, item.document_id)
    if not document:
        raise HTTPException(409, "材料版本关联的文件不存在")
    return MaterialArtifactResponse(
        id=item.id,
        document_id=item.document_id,
        program_id=item.program_id,
        kind=item.kind,
        scope=item.scope,
        version_name=item.version_name,
        language=item.language,
        status=item.status,
        notes=item.notes,
        filename=document.filename,
        parse_status=document.parse_status,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


def validate_artifact_values(
    kind: str, scope: str, status: str, program_id: Optional[str]
) -> None:
    if kind not in MATERIAL_KINDS:
        raise HTTPException(422, "材料类型只支持 cv 或 ps")
    if scope not in MATERIAL_SCOPES:
        raise HTTPException(422, "版本范围只支持 general 或 program")
    if status not in MATERIAL_STATUSES:
        raise HTTPException(422, "版本状态只支持 draft、ready 或 submitted")
    if scope == "program" and not program_id:
        raise HTTPException(422, "学校定制版本必须关联项目")
    if scope == "general" and program_id:
        raise HTTPException(422, "通用版本不能关联具体项目")


@router.get("/material-artifacts", response_model=List[MaterialArtifactResponse])
async def list_material_artifacts(
    kind: str = Query(""), session: AsyncSession = Depends(get_session)
) -> List[MaterialArtifactResponse]:
    statement = select(MaterialArtifact).where(
        MaterialArtifact.owner_id == settings.local_owner_id
    )
    if kind:
        statement = statement.where(MaterialArtifact.kind == kind)
    items = list((await session.scalars(statement.order_by(MaterialArtifact.updated_at.desc()))).all())
    return [await serialize_artifact(session, item) for item in items]


@router.post("/material-artifacts", response_model=MaterialArtifactResponse, status_code=201)
async def create_material_artifact(
    payload: MaterialArtifactCreate, session: AsyncSession = Depends(get_session)
) -> MaterialArtifactResponse:
    validate_artifact_values(payload.kind, payload.scope, payload.status, payload.program_id)
    document = await session.get(Document, payload.document_id)
    if not document or document.owner_id != settings.local_owner_id:
        raise HTTPException(404, "文件不存在")
    if document.kind != payload.kind:
        raise HTTPException(422, "文件类型与材料版本类型不一致")
    if payload.program_id and not await get_program(session, payload.program_id):
        raise HTTPException(404, "项目不存在")
    item = MaterialArtifact(owner_id=settings.local_owner_id, **payload.model_dump())
    session.add(item)
    await session.commit()
    await session.refresh(item)
    return await serialize_artifact(session, item)


@router.patch("/material-artifacts/{artifact_id}", response_model=MaterialArtifactResponse)
async def update_material_artifact(
    artifact_id: str,
    payload: MaterialArtifactUpdate,
    session: AsyncSession = Depends(get_session),
) -> MaterialArtifactResponse:
    item = await session.get(MaterialArtifact, artifact_id)
    if not item or item.owner_id != settings.local_owner_id:
        raise HTTPException(404, "材料版本不存在")
    values = payload.model_dump(exclude_unset=True)
    kind = item.kind
    scope = str(values.get("scope", item.scope))
    status = str(values.get("status", item.status))
    program_id = values.get("program_id", item.program_id)
    validate_artifact_values(kind, scope, status, program_id)
    if program_id and not await get_program(session, str(program_id)):
        raise HTTPException(404, "项目不存在")
    for key, value in values.items():
        setattr(item, key, value)
    await session.commit()
    await session.refresh(item)
    return await serialize_artifact(session, item)


@router.post("/material-artifacts/preflight", response_model=MaterialPreflightResponse)
async def material_preflight(
    payload: MaterialPreflightRequest, session: AsyncSession = Depends(get_session)
) -> MaterialPreflightResponse:
    item = await session.get(MaterialArtifact, payload.artifact_id)
    if not item or item.owner_id != settings.local_owner_id:
        raise HTTPException(404, "材料版本不存在")
    document = await session.get(Document, item.document_id)
    program = await get_program(session, payload.program_id)
    if not document or not program:
        raise HTTPException(404, "文件或项目不存在")
    checks = [
        {"name": "文件解析成功", "passed": document.parse_status not in {"failed", "pending"}},
        {"name": "材料类型正确", "passed": item.kind in MATERIAL_KINDS},
        {"name": "语言已标注", "passed": bool(item.language.strip())},
        {
            "name": "项目版本匹配",
            "passed": item.scope == "general" or item.program_id == payload.program_id,
        },
        {"name": "版本已确认可提交", "passed": item.status in {"ready", "submitted"}},
    ]
    warnings = []
    if item.scope == "general":
        warnings.append(f"当前是通用 {item.kind.upper()}，提交 {program.name} 前建议确认学校定制内容。")
    if item.status == "submitted":
        warnings.append("该版本已标记为已提交，请确认是否需要重复使用。")
    return MaterialPreflightResponse(
        ready_to_upload=all(bool(check["passed"]) for check in checks),
        artifact_id=item.id,
        program_id=program.id,
        checks=checks,
        warnings=warnings,
    )


@router.get("/material-drafts", response_model=List[MaterialDraftResponse])
async def list_material_drafts(
    kind: str = Query(""),
    program_id: str = Query(""),
    slot_key: str = Query(""),
    session: AsyncSession = Depends(get_session),
) -> List[MaterialDraft]:
    statement = select(MaterialDraft).where(
        MaterialDraft.owner_id == settings.local_owner_id
    )
    if kind:
        statement = statement.where(MaterialDraft.kind == kind)
    if program_id:
        statement = statement.where(MaterialDraft.program_id == program_id)
    if slot_key:
        statement = statement.where(MaterialDraft.slot_key == slot_key)
    return list(
        (await session.scalars(statement.order_by(MaterialDraft.updated_at.desc()))).all()
    )


@router.get("/material-drafts/{draft_id}/export")
async def export_material_draft(
    draft_id: str,
    format: str = Query("docx", pattern="^(docx|pdf)$"),
    session: AsyncSession = Depends(get_session),
) -> Response:
    item = await session.get(MaterialDraft, draft_id)
    if not item or item.owner_id != settings.local_owner_id:
        raise HTTPException(404, "文稿不存在")
    program = await get_program(session, item.program_id) if item.program_id else None
    filename = export_filename(item, program, format)
    if format == "pdf":
        content = pdf_export(item, program)
        media_type = "application/pdf"
    else:
        content = docx_export(item, program)
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": content_disposition(filename)},
    )


@router.post("/material-drafts/generate", response_model=MaterialDraftResponse, status_code=201)
async def generate_material_draft(
    payload: MaterialDraftGenerate, session: AsyncSession = Depends(get_session)
) -> MaterialDraft:
    if payload.kind not in {"cv", "ps", "recommendation"}:
        raise HTTPException(422, "材料类型只支持 cv、ps 或 recommendation")
    if payload.language not in {"English", "Chinese"}:
        raise HTTPException(422, "生成语言只支持 English 或 Chinese")
    if check_user_input(payload.prompt):
        raise HTTPException(400, "附加题目包含不安全的指令内容")
    profile, experiences = await profile_with_experiences(session)
    # This legacy one-shot endpoint has no conversation. It always uses the
    # confirmed profile and confirmed experiences; conversational resource
    # selection is handled by /writing-conversations/{id}/messages.
    selected_resources = {"profile", "confirmed_experiences"}
    confirmed_all = [item for item in experiences if item.confirmed]
    selected_experience_ids = {
        value.removeprefix("experience:")
        for value in selected_resources
        if value.startswith("experience:")
    }
    include_all_experiences = "confirmed_experiences" in selected_resources or not selected_experience_ids
    confirmed = [
        item for item in confirmed_all
        if include_all_experiences or item.id in selected_experience_ids
    ]
    if not profile.confirmed:
        raise HTTPException(400, "请先填写并确认申请画像")
    if not confirmed:
        raise HTTPException(400, "请先在画像中添加并确认至少一段真实经历")
    program = None
    if payload.program_id:
        program = await get_program(session, payload.program_id)
        if not program:
            raise HTTPException(404, "项目不存在")
        selected_package = await session.scalar(
            select(ApplicationPackage).where(
                ApplicationPackage.owner_id == settings.local_owner_id,
                ApplicationPackage.program_id == payload.program_id,
            )
        )
        if not selected_package:
            raise HTTPException(400, "该项目尚未加入选校清单，请先选择项目并建立申请包")
    if payload.kind == "ps" and not program:
        raise HTTPException(422, "生成完整 PS 必须选择目标项目")
    generation_input = {
        "kind": payload.kind,
        "language": payload.language,
        "prompt": payload.prompt,
        "profile": {
            "full_name": profile.full_name,
            "current_school": profile.current_school,
            "current_major": profile.current_major,
            "degree": profile.degree,
            "gpa": profile.gpa,
            "gpa_scale": profile.gpa_scale,
            "language_scores": profile.language_scores,
            "target_countries": profile.target_countries,
            "target_fields": profile.target_fields,
            "intake": profile.intake,
        },
        "confirmed_experiences": [
            {
                "id": item.id,
                "kind": item.kind,
                "title": item.title,
                "organization": item.organization,
                "start_date": item.start_date,
                "end_date": item.end_date,
                "description": item.description,
                "tags": item.tags,
            }
            for item in confirmed
        ],
        "program": (
            {
                "id": program.id,
                "university": program.university,
                "name": program.name,
                "field": program.field,
                "official_url": program.official_url,
            }
            if program else None
        ),
    }
    try:
        generated = await provider.generate_material(generation_input)
    except Exception as exc:
        raise HTTPException(502, f"大模型生成失败：{str(exc)[:300]}") from exc
    allowed_ids = {item.id for item in confirmed}
    source_ids = [
        value for value in generated.get("source_experience_ids", []) if value in allowed_ids
    ]
    content = str(generated.get("content", "")).strip()
    findings = verify_output(content)
    if not content or findings:
        raise HTTPException(422, "生成内容未通过完整性或安全检查")
    title = str(generated.get("title") or f"{payload.kind.upper()} draft")[:240]
    version_id = str(uuid4())
    item = MaterialDraft(
        id=version_id,
        owner_id=settings.local_owner_id,
        program_id=program.id if program else None,
        slot_key=payload.slot_key,
        parent_id=None,
        derived_from_id=None,
        root_id=version_id,
        version_number=1,
        revision_type="generated",
        change_summary="由 AI 根据当前确认画像、经历和目标项目首次生成",
        kind=payload.kind,
        title=title,
        language=payload.language,
        prompt=payload.prompt,
        content=content,
        source_experience_ids=source_ids,
        warnings=list(generated.get("warnings", [])),
        model_info=dict(generated.get("model_info", {})),
        status="draft",
    )
    session.add(item)
    await session.commit()
    await session.refresh(item)
    return item


@router.patch("/material-drafts/{draft_id}", response_model=MaterialDraftResponse, status_code=201)
async def update_material_draft(
    draft_id: str,
    payload: MaterialDraftUpdate,
    session: AsyncSession = Depends(get_session),
) -> MaterialDraft:
    item = await session.get(MaterialDraft, draft_id)
    if not item or item.owner_id != settings.local_owner_id:
        raise HTTPException(404, "文稿不存在")
    values = payload.model_dump(exclude_unset=True)
    if values.get("status") not in {None, "draft", "reviewed"}:
        raise HTTPException(422, "文稿状态只支持 draft 或 reviewed")
    content = values.get("content")
    if content and verify_output(content):
        raise HTTPException(422, "文稿内容未通过安全检查")
    root_id = item.root_id or item.id
    latest_version = await session.scalar(
        select(func.max(MaterialDraft.version_number)).where(
            MaterialDraft.owner_id == settings.local_owner_id,
            MaterialDraft.root_id == root_id,
        )
    )
    changed = []
    if values.get("title") is not None and values["title"] != item.title:
        changed.append("标题")
    if values.get("content") is not None and values["content"] != item.content:
        changed.append("正文")
    if values.get("status") is not None and values["status"] != item.status:
        changed.append("复核状态")
    if not changed:
        raise HTTPException(400, "内容没有变化，无需创建新版本")
    if changed == ["复核状态"]:
        item.status = str(values["status"])
        await session.commit()
        await session.refresh(item)
        return item
    new_item = MaterialDraft(
        owner_id=item.owner_id,
        program_id=item.program_id,
        slot_key=item.slot_key,
        parent_id=item.id,
        derived_from_id=item.derived_from_id,
        root_id=root_id,
        version_number=int(latest_version or item.version_number) + 1,
        revision_type="manual_edit",
        change_summary=f"人工修改：{'、'.join(changed)}",
        kind=item.kind,
        title=str(values.get("title", item.title)),
        language=item.language,
        prompt=item.prompt,
        content=str(values.get("content", item.content)),
        source_experience_ids=item.source_experience_ids,
        warnings=item.warnings,
        model_info={**item.model_info, "edited_from": item.id},
        context_manifest_id=item.context_manifest_id,
        source_refs=item.source_refs,
        status=str(values.get("status", item.status)),
    )
    session.add(new_item)
    await session.commit()
    await session.refresh(new_item)
    return new_item


@router.post("/material-drafts/{draft_id}/restore", response_model=MaterialDraftResponse, status_code=201)
async def restore_material_draft(
    draft_id: str,
    session: AsyncSession = Depends(get_session),
) -> MaterialDraft:
    item = await session.get(MaterialDraft, draft_id)
    if not item or item.owner_id != settings.local_owner_id:
        raise HTTPException(404, "文稿不存在")
    root_id = item.root_id or item.id
    latest_version = await session.scalar(select(func.max(MaterialDraft.version_number)).where(
        MaterialDraft.owner_id == settings.local_owner_id,
        MaterialDraft.root_id == root_id,
    ))
    restored = MaterialDraft(
        owner_id=item.owner_id,
        program_id=item.program_id,
        slot_key=item.slot_key,
        parent_id=item.id,
        derived_from_id=item.derived_from_id,
        root_id=root_id,
        version_number=int(latest_version or item.version_number) + 1,
        revision_type="restored",
        change_summary=f"基于 v{item.version_number} 恢复为新版本",
        kind=item.kind,
        title=item.title,
        language=item.language,
        prompt=item.prompt,
        content=item.content,
        source_experience_ids=item.source_experience_ids,
        warnings=item.warnings,
        model_info={**item.model_info, "restored_from": item.id},
        context_manifest_id=item.context_manifest_id,
        source_refs=item.source_refs,
        status="draft",
    )
    session.add(restored)
    await session.commit()
    await session.refresh(restored)
    return restored


@router.get(
    "/context-manifests/{manifest_id}", response_model=ContextManifestResponse
)
async def get_context_manifest(
    manifest_id: str,
    session: AsyncSession = Depends(get_session),
) -> ContextManifest:
    item = await session.get(ContextManifest, manifest_id)
    if not item or item.owner_id != settings.local_owner_id:
        raise HTTPException(404, "上下文清单不存在")
    return item


async def serialize_writing_conversation(
    session: AsyncSession, conversation: Conversation
) -> WritingConversationResponse:
    messages = list((await session.scalars(
        select(Message).where(
            Message.conversation_id == conversation.id,
            Message.owner_id == settings.local_owner_id,
        ).order_by(Message.created_at)
    )).all())
    draft_candidates = list((await session.scalars(
        select(MaterialDraft).where(
            MaterialDraft.owner_id == settings.local_owner_id,
            MaterialDraft.program_id == conversation.program_id,
            MaterialDraft.slot_key == conversation.slot_key,
        ).order_by(MaterialDraft.updated_at.desc())
    )).all())
    latest_draft = next((
        draft for draft in draft_candidates
        if draft.model_info.get("conversation_id") == conversation.id
    ), None)
    return WritingConversationResponse(
        id=conversation.id,
        title=conversation.title,
        program_id=str(conversation.program_id),
        slot_key=conversation.slot_key,
        material_kind=conversation.material_kind,
        resource_ids=conversation.resource_ids,
        memory_state=conversation.memory_state,
        messages=[WritingMessageResponse.model_validate(item) for item in messages],
        latest_draft=(MaterialDraftResponse.model_validate(latest_draft) if latest_draft else None),
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.get("/writing-conversations", response_model=List[WritingConversationResponse])
async def list_writing_conversations(
    program_id: str = Query(""),
    slot_key: str = Query(""),
    session: AsyncSession = Depends(get_session),
) -> List[WritingConversationResponse]:
    statement = select(Conversation).where(
        Conversation.owner_id == settings.local_owner_id,
        Conversation.program_id.is_not(None),
    )
    if program_id:
        statement = statement.where(Conversation.program_id == program_id)
    if slot_key:
        statement = statement.where(Conversation.slot_key == slot_key)
    items = list((await session.scalars(statement.order_by(Conversation.updated_at.desc()))).all())
    return [await serialize_writing_conversation(session, item) for item in items]


@router.get("/assistant-conversations", response_model=List[AssistantConversationResponse])
async def list_assistant_conversations(
    session: AsyncSession = Depends(get_session),
) -> List[AssistantConversationResponse]:
    items = list((await session.scalars(
        select(Conversation).where(
            Conversation.owner_id == settings.local_owner_id
        ).order_by(Conversation.pinned.desc(), Conversation.updated_at.desc())
    )).all())
    output = []
    for item in items:
        messages = list((await session.scalars(
            select(Message).where(
                Message.owner_id == settings.local_owner_id,
                Message.conversation_id == item.id,
            ).order_by(Message.created_at)
        )).all())
        program_label = ""
        if item.program_id:
            program = await get_program(session, item.program_id)
            if program:
                program_label = f"{program.university} · {program.name}"
        output.append(AssistantConversationResponse(
            id=item.id,
            title=item.title,
            scene=("material" if item.material_kind else "application"),
            program_id=item.program_id,
            program_label=program_label,
            slot_key=item.slot_key,
            material_kind=item.material_kind,
            pinned=item.pinned,
            resource_ids=item.resource_ids,
            memory_state=item.memory_state,
            messages=[WritingMessageResponse.model_validate(message) for message in messages],
            created_at=item.created_at,
            updated_at=item.updated_at,
        ))
    return output


@router.post("/assistant-conversations", response_model=AssistantConversationResponse, status_code=201)
async def create_assistant_conversation(
    payload: AssistantConversationCreate,
    session: AsyncSession = Depends(get_session),
) -> AssistantConversationResponse:
    item = Conversation(
        owner_id=settings.local_owner_id,
        title=payload.title.strip(),
        resource_ids=payload.resource_ids,
    )
    session.add(item)
    await session.commit()
    await session.refresh(item)
    return AssistantConversationResponse(
        id=item.id,
        title=item.title,
        scene="application",
        resource_ids=item.resource_ids,
        memory_state=item.memory_state,
        messages=[],
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


@router.patch("/assistant-conversations/{conversation_id}", response_model=AssistantConversationResponse)
async def update_assistant_conversation(
    conversation_id: str,
    payload: AssistantConversationUpdate,
    session: AsyncSession = Depends(get_session),
) -> AssistantConversationResponse:
    item = await session.get(Conversation, conversation_id)
    if not item or item.owner_id != settings.local_owner_id:
        raise HTTPException(404, "对话不存在")
    values = payload.model_dump(exclude_unset=True)
    if "title" in values:
        values["title"] = str(values["title"]).strip()
        if not values["title"]:
            raise HTTPException(422, "对话名称不能为空")
    for key, value in values.items():
        setattr(item, key, value)
    await session.commit()
    await session.refresh(item)
    messages = list((await session.scalars(select(Message).where(
        Message.owner_id == settings.local_owner_id,
        Message.conversation_id == item.id,
    ).order_by(Message.created_at))).all())
    program_label = ""
    if item.program_id:
        program = await get_program(session, item.program_id)
        if program:
            program_label = f"{program.university} · {program.name}"
    return AssistantConversationResponse(
        id=item.id,
        title=item.title,
        scene=("material" if item.material_kind else "application"),
        program_id=item.program_id,
        program_label=program_label,
        slot_key=item.slot_key,
        material_kind=item.material_kind,
        pinned=item.pinned,
        resource_ids=item.resource_ids,
        memory_state=item.memory_state,
        messages=[WritingMessageResponse.model_validate(message) for message in messages],
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


@router.delete("/assistant-conversations/{conversation_id}", status_code=204)
async def delete_assistant_conversation(
    conversation_id: str,
    session: AsyncSession = Depends(get_session),
) -> Response:
    item = await session.get(Conversation, conversation_id)
    if not item or item.owner_id != settings.local_owner_id:
        raise HTTPException(404, "对话不存在")
    await session.execute(delete(Message).where(
        Message.owner_id == settings.local_owner_id,
        Message.conversation_id == item.id,
    ))
    await session.delete(item)
    await session.commit()
    return Response(status_code=204)


@router.post("/writing-conversations", response_model=WritingConversationResponse, status_code=201)
async def create_writing_conversation(
    payload: WritingConversationCreate,
    session: AsyncSession = Depends(get_session),
) -> WritingConversationResponse:
    if payload.material_kind not in {"cv", "ps", "recommendation"}:
        raise HTTPException(422, "不支持的材料类型")
    if not await get_program(session, payload.program_id):
        raise HTTPException(404, "项目不存在")
    item = Conversation(
        owner_id=settings.local_owner_id,
        title=payload.title,
        program_id=payload.program_id,
        slot_key=payload.slot_key,
        material_kind=payload.material_kind,
        resource_ids=payload.resource_ids,
    )
    session.add(item)
    await session.commit()
    await session.refresh(item)
    return await serialize_writing_conversation(session, item)


@router.patch("/writing-conversations/{conversation_id}", response_model=WritingConversationResponse)
async def update_writing_conversation(
    conversation_id: str,
    payload: WritingConversationUpdate,
    session: AsyncSession = Depends(get_session),
) -> WritingConversationResponse:
    item = await session.get(Conversation, conversation_id)
    if not item or item.owner_id != settings.local_owner_id or not item.program_id:
        raise HTTPException(404, "文书对话不存在")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(item, key, value)
    await session.commit()
    await session.refresh(item)
    return await serialize_writing_conversation(session, item)


@router.post(
    "/writing-conversations/{conversation_id}/messages",
    response_model=WritingConversationResponse,
)
async def send_writing_message(
    conversation_id: str,
    payload: WritingMessageCreate,
    session: AsyncSession = Depends(get_session),
) -> WritingConversationResponse:
    conversation = await session.get(Conversation, conversation_id)
    if not conversation or conversation.owner_id != settings.local_owner_id or not conversation.program_id:
        raise HTTPException(404, "文书对话不存在")
    if check_user_input(payload.message):
        raise HTTPException(400, "消息包含不安全的指令内容")
    current_task = asyncio.current_task()
    previous_task = active_writing_tasks.get(conversation_id)
    if previous_task and not previous_task.done() and previous_task is not current_task:
        raise HTTPException(409, "当前对话仍有内容正在生成")
    if current_task:
        active_writing_tasks[conversation_id] = current_task
    try:
        return await _generate_writing_reply(conversation, payload, session)
    except asyncio.CancelledError:
        await asyncio.shield(session.rollback())
        raise HTTPException(409, "本次生成已停止") from None
    finally:
        if active_writing_tasks.get(conversation_id) is current_task:
            active_writing_tasks.pop(conversation_id, None)


@router.post("/writing-conversations/{conversation_id}/cancel")
async def cancel_writing_generation(
    conversation_id: str,
    session: AsyncSession = Depends(get_session),
) -> Dict[str, bool]:
    conversation = await session.get(Conversation, conversation_id)
    if not conversation or conversation.owner_id != settings.local_owner_id:
        raise HTTPException(404, "文书对话不存在")
    task = active_writing_tasks.get(conversation_id)
    if not task or task.done():
        return {"cancelled": False}
    task.cancel()
    return {"cancelled": True}


async def _generate_writing_reply(
    conversation: Conversation,
    payload: WritingMessageCreate,
    session: AsyncSession,
) -> WritingConversationResponse:
    profile, experiences = await profile_with_experiences(session)
    pending_resources = set(conversation.resource_ids or [])
    selected_resources = set(pending_resources)
    previous_messages = list((await session.scalars(select(Message).where(
        Message.owner_id == settings.local_owner_id,
        Message.conversation_id == conversation.id,
    ).order_by(Message.created_at))).all())
    for message in previous_messages:
        if message.role != "user":
            continue
        for source in message.sources or []:
            if source.get("type") == "document" and source.get("id"):
                selected_resources.add(f"document:{source['id']}")
            elif source.get("type") in {"draft", "reference_draft"} and source.get("id"):
                selected_resources.add(f"draft:{source['id']}")
    confirmed_all = [item for item in experiences if item.confirmed]
    selected_experience_ids = {
        value.removeprefix("experience:")
        for value in selected_resources
        if value.startswith("experience:")
    }
    include_all_experiences = "confirmed_experiences" in selected_resources
    confirmed = [
        item for item in confirmed_all
        if include_all_experiences or item.id in selected_experience_ids
    ]
    program = await get_program(session, conversation.program_id)
    requirement = await get_requirement(session, conversation.program_id)
    if not program:
        raise HTTPException(404, "项目不存在")
    document_ids = {
        value.removeprefix("document:")
        for value in selected_resources
        if value.startswith("document:")
    }
    draft_ids = {
        value.removeprefix("draft:")
        for value in selected_resources
        if value.startswith("draft:")
    }
    reference_documents = list((await session.scalars(select(Document).where(
        Document.owner_id == settings.local_owner_id,
        Document.id.in_(document_ids),
    ))).all()) if document_ids else []
    for item in reference_documents:
        if not item.extracted_text.strip() and provider.available and Path(item.path).is_file():
            try:
                extracted = await provider.extract_document(Path(item.path), item.mime_type, item.kind, "")
                summary = str(extracted.get("summary") or "").strip()
                if summary:
                    item.extracted_data = extracted
                    item.extracted_text = summary[:200_000]
                    item.parse_status = "parsed_multimodal"
            except Exception:
                pass
    reference_drafts = list((await session.scalars(select(MaterialDraft).where(
        MaterialDraft.owner_id == settings.local_owner_id,
        MaterialDraft.id.in_(draft_ids),
    ))).all()) if draft_ids else []
    if "historical_drafts" in selected_resources:
        historical_drafts = list((await session.scalars(select(MaterialDraft).where(
            MaterialDraft.owner_id == settings.local_owner_id,
            MaterialDraft.status == "reviewed",
        ).order_by(MaterialDraft.updated_at.desc()))).all())
        latest_by_root: Dict[str, MaterialDraft] = {}
        for item in historical_drafts:
            root_key = item.root_id or item.id
            if root_key not in latest_by_root:
                latest_by_root[root_key] = item
        known_draft_ids = {item.id for item in reference_drafts}
        reference_drafts.extend(
            item for item in latest_by_root.values() if item.id not in known_draft_ids
        )
    active_memories = list((await session.scalars(select(Memory).where(
        Memory.owner_id == settings.local_owner_id,
        Memory.active.is_(True),
    ).order_by(Memory.updated_at.desc()))).all())
    memories_by_key: Dict[str, Memory] = {}
    for item in active_memories:
        memories_by_key.setdefault(f"{item.memory_type}:{item.key}", item)
    official_context: Dict[str, Any] = {}
    if requirement:
        current_slot = next((
            item for item in material_slots([str(value) for value in requirement.materials])
            if item.get("slot_key") == conversation.slot_key
        ), None)
        evidence_rows = list((await session.scalars(select(EvidenceChunk).where(
            EvidenceChunk.program_id == program.id,
            EvidenceChunk.field == "materials",
        ).order_by(EvidenceChunk.confidence.desc()))).all())
        source_ids = {item.source_id for item in evidence_rows}
        source_rows = list((await session.scalars(select(ProgramSource).where(
            ProgramSource.id.in_(source_ids)
        ))).all()) if source_ids else []
        source_urls = {item.id: item.url for item in source_rows}
        exact_requirement = current_slot.get("official_name") if current_slot else ""
        requirement_terms = {
            term for term in re.findall(r"[a-zA-Z]{3,}|[\u4e00-\u9fff]{2,}", exact_requirement.lower())
            if term not in {"the", "and", "for", "with", "your", "申请", "材料"}
        }
        evidence_payload = [{
            "evidence_id": item.id,
            "source_id": item.source_id,
            "quote": item.quote,
            "locator": item.locator,
            "url": source_urls.get(item.source_id, program.official_url),
            "confidence": item.confidence,
        } for item in evidence_rows]
        relevant_evidence = [item for item in evidence_payload if any(
            term in str(item["quote"]).lower() for term in requirement_terms
        )]
        relevant_quotes = {str(item["quote"]) for item in relevant_evidence}
        official_context = {
            "slot_key": conversation.slot_key,
            "material_name": current_slot.get("name") if current_slot else conversation.material_kind,
            "exact_requirement": exact_requirement,
            "all_material_requirements": requirement.materials,
            "verified": requirement.verified,
            "official_url": program.official_url,
            "evidence": relevant_evidence,
            "general_evidence": [
                item for item in evidence_payload if str(item["quote"]) not in relevant_quotes
            ],
        }
    user_message = Message(
        owner_id=settings.local_owner_id,
        conversation_id=conversation.id,
        role="user",
        content=payload.message,
        sources=[
            *[{"type": "document", "id": value.removeprefix("document:")} for value in pending_resources if value.startswith("document:")],
            *[{"type": "reference_draft", "id": value.removeprefix("draft:")} for value in pending_resources if value.startswith("draft:")],
        ],
    )
    session.add(user_message)
    await session.flush()
    history = list((await session.scalars(
        select(Message).where(Message.conversation_id == conversation.id).order_by(Message.created_at)
    )).all())
    conversation_drafts = list((await session.scalars(select(MaterialDraft).where(
        MaterialDraft.owner_id == settings.local_owner_id,
        MaterialDraft.program_id == program.id,
        MaterialDraft.slot_key == conversation.slot_key,
    ).order_by(MaterialDraft.updated_at.desc()))).all())
    current_draft = next((
        item for item in conversation_drafts
        if item.model_info.get("conversation_id") == conversation.id
    ), None)
    resource_snapshots = [
        *[
            {
                "resource_id": f"document:{item.id}",
                "label": item.filename,
                "kind": item.kind,
                "readable": bool(
                    (item.extracted_text or "").strip()
                    or str((item.extracted_data or {}).get("summary") or "").strip()
                ),
                "content_hash": content_hash(
                    item.extracted_text
                    or str((item.extracted_data or {}).get("summary") or "")
                ),
            }
            for item in reference_documents
        ],
        *[
            {
                "resource_id": f"draft:{item.id}",
                "label": item.title,
                "kind": item.kind,
                "readable": bool(item.content.strip()),
                "content_hash": content_hash(item.content),
            }
            for item in reference_drafts
        ],
    ]
    if "profile" in selected_resources:
        resource_snapshots.append({
            "resource_id": "profile",
            "label": "已确认画像",
            "kind": "profile",
            "readable": bool(profile.confirmed),
            "content_hash": content_hash({
                "id": profile.id,
                "updated_at": profile.updated_at.isoformat(),
                "confirmed": profile.confirmed,
            }),
        })
    if "confirmed_experiences" in selected_resources:
        resource_snapshots.append({
            "resource_id": "confirmed_experiences",
            "label": "已确认经历",
            "kind": "experiences",
            "readable": bool(confirmed),
            "content_hash": content_hash([
                {"id": item.id, "updated_at": item.updated_at.isoformat()}
                for item in confirmed
            ]),
        })
    resource_snapshots.extend({
        "resource_id": f"experience:{item.id}",
        "label": item.title,
        "kind": "experience",
        "readable": True,
        "content_hash": content_hash({
            "id": item.id,
            "updated_at": item.updated_at.isoformat(),
            "description": item.description,
        }),
    } for item in confirmed)
    if "official_requirements" in selected_resources:
        resource_snapshots.append({
            "resource_id": "official_requirements",
            "label": "官网要求与原文证据",
            "kind": "official_requirements",
            "readable": bool(official_context.get("evidence")),
            "content_hash": content_hash(official_context),
        })
    if "historical_drafts" in selected_resources:
        resource_snapshots.append({
            "resource_id": "historical_drafts",
            "label": "历史文稿",
            "kind": "draft_collection",
            "readable": bool(reference_drafts),
            "content_hash": content_hash([
                {"id": item.id, "updated_at": item.updated_at.isoformat()}
                for item in reference_drafts
            ]),
        })
    conversation_memory, recent_history = build_conversation_memory(
        history,
        conversation.memory_state or {},
        selected_resources,
        resource_snapshots,
    )
    conversation.memory_state = conversation_memory
    generation_input = {
        "interaction_mode": "assistant",
        "kind": conversation.material_kind,
        "language": "English",
        "prompt": payload.message,
        "conversation_history": [
            {"role": item["role"], "content": item["content"]}
            for item in recent_history
        ],
        "conversation_memory": conversation_memory,
        "resource_ids": conversation.resource_ids,
        "slot_key": conversation.slot_key,
        "official_requirements": official_context,
        "profile": ({
            "id": profile.id,
            "updated_at": profile.updated_at.isoformat(),
            "confirmed": profile.confirmed,
            "full_name": profile.full_name,
            "current_school": profile.current_school,
            "current_major": profile.current_major,
            "degree": profile.degree,
            "gpa": profile.gpa,
            "gpa_scale": profile.gpa_scale,
            "language_scores": profile.language_scores,
            "target_fields": profile.target_fields,
            "target_countries": profile.target_countries,
            "intake": profile.intake,
            "budget": profile.budget,
            "preferences": profile.preferences,
        } if "profile" in selected_resources else {}),
        "memories": [{
            "type": item.memory_type,
            "key": item.key,
            "value": item.value,
            "source_type": item.source_type,
        } for item in memories_by_key.values()],
        "confirmed_experiences": [{
            "id": item.id, "kind": item.kind, "title": item.title,
            "organization": item.organization, "start_date": item.start_date,
            "end_date": item.end_date, "description": item.description, "tags": item.tags,
        } for item in confirmed],
        "reference_documents": [{
            "id": item.id,
            "filename": item.filename,
            "kind": item.kind,
            "content": (item.extracted_text or str((item.extracted_data or {}).get("summary") or ""))[:20_000],
        } for item in reference_documents],
        "reference_drafts": [{
            "id": item.id,
            "title": item.title,
            "kind": item.kind,
            "source_program_id": item.program_id,
            "content": item.content[:20_000],
        } for item in reference_drafts],
        "current_draft": ({
            "id": current_draft.id,
            "version": current_draft.version_number,
            "title": current_draft.title,
            "content": current_draft.content,
            "status": current_draft.status,
        } if current_draft else {}),
        "program": {
            "id": program.id, "university": program.university, "name": program.name,
            "field": program.field, "official_url": program.official_url,
        },
    }
    goal_spec = parse_material_goal(
        payload.message,
        conversation.material_kind,
        program.id,
        conversation.slot_key,
        bool(current_draft),
    )
    run, run_steps = await start_material_run(
        session, conversation, payload.message, goal_spec
    )
    set_material_step(
        run_steps,
        "resolve_writing_intent",
        "accepted",
        {"goal_spec": goal_spec},
    )
    manifest_data = build_context_manifest_data(
        goal_spec=goal_spec,
        profile=generation_input["profile"],
        conversation_id=conversation.id,
        history=generation_input["conversation_history"],
        conversation_memory=conversation_memory,
        program=generation_input["program"],
        material_kind=conversation.material_kind,
        slot_key=conversation.slot_key,
        selected_resource_ids=selected_resources,
        experiences=[{**item, "confirmed": True} for item in generation_input["confirmed_experiences"]],
        documents=generation_input["reference_documents"],
        drafts=generation_input["reference_drafts"],
        official_context=official_context,
        current_draft=generation_input["current_draft"],
    )
    completeness = validate_context_manifest(manifest_data, goal_spec)
    manifest = ContextManifest(
        owner_id=settings.local_owner_id,
        agent_run_id=run.id,
        conversation_id=conversation.id,
        program_id=program.id,
        material_kind=conversation.material_kind,
        slot_key=conversation.slot_key,
        intent=goal_spec["intent"],
        profile_ref=manifest_data["profile_ref"],
        conversation_ref=manifest_data["conversation_ref"],
        experience_refs=manifest_data["experience_refs"],
        document_refs=manifest_data["document_refs"],
        draft_refs=manifest_data["draft_refs"],
        official_evidence_refs=manifest_data["official_evidence_refs"],
        current_draft_ref=manifest_data["current_draft_ref"],
        content_hashes=manifest_data["content_hashes"],
        completeness=completeness,
        status="accepted" if completeness["verdict"] == "accepted" else "partial",
    )
    session.add(manifest)
    await session.flush()
    generation_input["context_manifest"] = {
        **manifest_data,
        "id": manifest.id,
        "completeness": completeness,
    }
    run.context_snapshot = {
        "goal_spec": goal_spec,
        "context_manifest_id": manifest.id,
        "context_manifest_hash": manifest.content_hashes.get("manifest"),
    }
    context_accepted = completeness["verdict"] == "accepted" or goal_spec["intent"] == "chat"
    set_material_step(
        run_steps,
        "resolve_material_context",
        "accepted" if context_accepted else "rejected",
        {"manifest_id": manifest.id, "completeness": completeness},
    )
    if not context_accepted:
        for step_type in (
            "generate_or_edit_draft",
            "validate_draft",
            "independent_verify_draft",
            "present_candidate_version",
            "adopt_version",
        ):
            set_material_step(
                run_steps, step_type, "blocked", {"reason": "context_incomplete"}
            )
        run.status = "failed"
        run.stop_reason = "context_incomplete"
        run.final_output = "生成前上下文不完整"
        sync_material_plan(run, run_steps)
        await session.commit()
        raise HTTPException(
            422,
            f"生成前上下文不完整：{', '.join(completeness['issues'])}",
        )
    set_material_step(
        run_steps,
        "generate_or_edit_draft",
        "running",
        {"manifest_id": manifest.id},
    )
    sync_material_plan(run, run_steps)
    await session.commit()
    try:
        generated = await provider.generate_material(generation_input)
    except Exception as exc:
        set_material_step(
            run_steps,
            "generate_or_edit_draft",
            "failed",
            {"error": str(exc)[:300]},
        )
        run.status = "failed"
        run.stop_reason = "generation_error"
        for step_type in (
            "validate_draft",
            "independent_verify_draft",
            "present_candidate_version",
            "adopt_version",
        ):
            set_material_step(
                run_steps, step_type, "blocked", {"reason": "generation_error"}
            )
        sync_material_plan(run, run_steps)
        await session.commit()
        raise HTTPException(502, f"大模型生成失败：{str(exc)[:300]}") from exc
    response_type = str(generated.get("response_type", "chat"))
    set_material_step(
        run_steps,
        "generate_or_edit_draft",
        "accepted",
        {"response_type": response_type},
    )
    context_sources = [
        *[{"type": "document", "id": item.id, "label": item.filename} for item in reference_documents],
        *[{"type": "reference_draft", "id": item.id, "label": item.title} for item in reference_drafts],
        *([{
            "type": "official_requirement",
            "url": program.official_url,
            "evidence_count": len(official_context.get("evidence", [])),
            "general_evidence_count": len(official_context.get("general_evidence", [])),
        }] if official_context else []),
        {"type": "conversation_history", "message_count": len(history)},
        {"type": "memory", "count": len(memories_by_key)},
        {"type": "context_manifest", "id": manifest.id},
        *([{"type": "current_draft", "id": current_draft.id, "version": current_draft.version_number}] if current_draft else []),
    ]
    if response_type == "chat":
        message_content = str(generated.get("message", "")).strip()
        if not message_content or verify_output(message_content):
            raise HTTPException(422, "对话回复未通过完整性或安全检查")
        session.add(Message(
            owner_id=settings.local_owner_id,
            conversation_id=conversation.id,
            role="assistant",
            content=message_content,
            sources=[{"type": "response_mode", "value": "chat"}, *context_sources],
        ))
        for step_type in (
            "validate_draft",
            "independent_verify_draft",
            "present_candidate_version",
            "adopt_version",
        ):
            set_material_step(run_steps, step_type, "skipped", {"reason": "chat_response"})
        manifest.status = "consumed" if completeness["verdict"] == "accepted" else "partial"
        run.status = "completed"
        run.stop_reason = "success"
        run.final_output = message_content
        run.structured_output = {
            "response_type": "chat",
            "context_manifest_id": manifest.id,
        }
        sync_material_plan(run, run_steps)
        conversation.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(conversation)
        return await serialize_writing_conversation(session, conversation)
    if response_type != "draft":
        for step_type in (
            "validate_draft",
            "independent_verify_draft",
            "present_candidate_version",
            "adopt_version",
        ):
            set_material_step(
                run_steps, step_type, "blocked", {"reason": "invalid_response_type"}
            )
        run.status = "failed"
        run.stop_reason = "invalid_response_type"
        sync_material_plan(run, run_steps)
        await session.commit()
        raise HTTPException(422, "模型返回了无法识别的材料响应类型")
    if goal_spec["intent"] not in {"generate", "edit"}:
        for step_type in (
            "validate_draft",
            "independent_verify_draft",
            "present_candidate_version",
            "adopt_version",
        ):
            set_material_step(
                run_steps, step_type, "blocked", {"reason": "write_intent_not_confirmed"}
            )
        run.status = "failed"
        run.stop_reason = "write_intent_not_confirmed"
        sync_material_plan(run, run_steps)
        await session.commit()
        raise HTTPException(422, "本轮未识别到明确的生成或修改意图，未创建文稿版本")
    content = str(generated.get("content", "")).strip()
    if not content or verify_output(content):
        set_material_step(
            run_steps,
            "validate_draft",
            "rejected",
            {"reason": "content_integrity_failed"},
        )
        run.status = "failed"
        run.stop_reason = "draft_validation_failed"
        for step_type in (
            "independent_verify_draft",
            "present_candidate_version",
            "adopt_version",
        ):
            set_material_step(
                run_steps,
                step_type,
                "blocked",
                {"reason": "draft_validation_failed"},
            )
        sync_material_plan(run, run_steps)
        await session.commit()
        raise HTTPException(422, "生成内容未通过完整性或安全检查")
    base_draft = next((
        item for item in conversation_drafts
        if item.model_info.get("conversation_id") == conversation.id
    ), None)
    if not base_draft:
        base_draft = next((
            item for item in reference_drafts
            if item.program_id == program.id and item.slot_key == conversation.slot_key
        ), None)
    cross_project_source = next((
        item for item in reference_drafts
        if item.program_id and item.program_id != program.id
    ), None)
    if base_draft and content == base_draft.content:
        set_material_step(
            run_steps,
            "validate_draft",
            "rejected",
            {"reason": "content_unchanged", "base_draft_id": base_draft.id},
        )
        set_material_step(
            run_steps,
            "independent_verify_draft",
            "skipped",
            {"reason": "content_unchanged"},
        )
        set_material_step(
            run_steps,
            "present_candidate_version",
            "skipped",
            {"reason": "content_unchanged"},
        )
        set_material_step(
            run_steps,
            "adopt_version",
            "skipped",
            {"reason": "no_candidate_version"},
        )
        manifest.status = "consumed"
        run.status = "completed"
        run.stop_reason = "no_change"
        run.final_output = "正文没有变化，未创建重复版本。"
        run.structured_output = {
            "response_type": "chat",
            "context_manifest_id": manifest.id,
            "reason": "content_unchanged",
        }
        sync_material_plan(run, run_steps)
        session.add(
            Message(
                owner_id=settings.local_owner_id,
                conversation_id=conversation.id,
                role="assistant",
                content="正文没有发生变化，因此没有创建重复版本。",
                sources=[
                    {"type": "response_mode", "value": "chat"},
                    *context_sources,
                ],
            )
        )
        await session.commit()
        await session.refresh(conversation)
        return await serialize_writing_conversation(session, conversation)
    allowed_source_ids = {item.id for item in confirmed}
    generated_source_ids = [
        value
        for value in generated.get("source_experience_ids", [])
        if value in allowed_source_ids
    ]
    set_material_step(
        run_steps,
        "validate_draft",
        "accepted",
        {
            "manifest_id": manifest.id,
            "base_draft_id": base_draft.id if base_draft else None,
            "source_experience_ids": generated_source_ids,
        },
    )
    verification_attempts = []
    verification_report = {}
    for attempt in range(2):
        set_material_step(
            run_steps,
            "independent_verify_draft",
            "running",
            {"attempt": attempt + 1, "manifest_id": manifest.id},
        )
        sync_material_plan(run, run_steps)
        await session.commit()
        try:
            semantic_report = await provider.verify_material(generation_input, generated)
        except Exception as exc:
            set_material_step(
                run_steps,
                "independent_verify_draft",
                "failed",
                {"error": str(exc)[:300], "attempt": attempt + 1},
            )
            run.status = "failed"
            run.stop_reason = "verification_error"
            manifest.status = "rejected"
            for step_type in ("present_candidate_version", "adopt_version"):
                set_material_step(
                    run_steps,
                    step_type,
                    "blocked",
                    {"reason": "verification_error"},
                )
            sync_material_plan(run, run_steps)
            await session.commit()
            raise HTTPException(502, f"独立验证器执行失败：{str(exc)[:300]}") from exc
        verification_report = verify_material_candidate(
            generation_input, generated, semantic_report
        )
        verification_attempts.append(verification_report)
        if verification_report.get("verdict") == "accepted":
            break
        if attempt == 0 and verification_report.get("verdict") == "revise":
            retry_input = {
                **generation_input,
                "prompt": payload.message,
                "verification_feedback": verification_report,
                "revision_instruction": (
                    "根据独立验证报告修订候选文稿；不得引入新的事实或来源。"
                ),
            }
            try:
                generated = await provider.generate_material(retry_input)
            except Exception as exc:
                set_material_step(
                    run_steps,
                    "independent_verify_draft",
                    "failed",
                    {"error": str(exc)[:300], "attempt": attempt + 1},
                )
                run.status = "failed"
                run.stop_reason = "verification_revision_error"
                manifest.status = "rejected"
                for step_type in ("present_candidate_version", "adopt_version"):
                    set_material_step(
                        run_steps,
                        step_type,
                        "blocked",
                        {"reason": "verification_revision_error"},
                    )
                sync_material_plan(run, run_steps)
                await session.commit()
                raise HTTPException(502, f"验证后自动修订失败：{str(exc)[:300]}") from exc
            if str(generated.get("response_type") or "") != "draft":
                verification_report = {
                    "verdict": "rejected",
                    "issues": ["revision_did_not_return_draft"],
                }
                break
            content = str(generated.get("content") or "").strip()
            if not content or verify_output(content) or (
                base_draft and content == base_draft.content
            ):
                verification_report = {
                    "verdict": "rejected",
                    "issues": ["revision_content_invalid_or_unchanged"],
                }
                break
            generated_source_ids = [
                value
                for value in generated.get("source_experience_ids", [])
                if value in allowed_source_ids
            ]
            generation_input = retry_input
            continue
        break
    if verification_report.get("verdict") != "accepted":
        set_material_step(
            run_steps,
            "independent_verify_draft",
            "rejected",
            {
                "verification_report": verification_report,
                "attempts": verification_attempts,
            },
        )
        for step_type in ("present_candidate_version", "adopt_version"):
            set_material_step(
                run_steps,
                step_type,
                "blocked",
                {"reason": "independent_verification_failed"},
            )
        manifest.status = "rejected"
        run.status = "failed"
        run.stop_reason = "independent_verification_failed"
        run.final_output = "候选文稿未通过独立验证，未创建新版本。"
        run.structured_output = {
            "response_type": "verification_report",
            "context_manifest_id": manifest.id,
            "verification": verification_report,
            "attempts": verification_attempts,
        }
        sync_material_plan(run, run_steps)
        await session.commit()
        issues = ", ".join(verification_report.get("issues") or [])
        raise HTTPException(422, f"候选文稿未通过独立验证：{issues or '需要修订'}")
    set_material_step(
        run_steps,
        "independent_verify_draft",
        "accepted",
        {
            "verification_report": verification_report,
            "attempt_count": len(verification_attempts),
        },
    )
    root_id = (base_draft.root_id or base_draft.id) if base_draft else None
    latest_version = await session.scalar(select(func.max(MaterialDraft.version_number)).where(
        MaterialDraft.owner_id == settings.local_owner_id,
        MaterialDraft.root_id == root_id,
    )) if root_id else None
    version = int(latest_version or 0) + 1
    draft = MaterialDraft(
        owner_id=settings.local_owner_id,
        program_id=program.id,
        slot_key=conversation.slot_key,
        parent_id=base_draft.id if base_draft else None,
        derived_from_id=cross_project_source.id if not base_draft and cross_project_source else None,
        root_id=root_id,
        version_number=version,
        revision_type=("ai_revision" if base_draft else "derived" if cross_project_source else "generated"),
        change_summary=(
            f"AI 基于 v{base_draft.version_number} 修改：{payload.message[:100]}"
            if base_draft else
            f"基于其他项目文稿创建：{payload.message[:100]}"
            if cross_project_source else
            f"由 AI 首次生成：{payload.message[:100]}"
        ),
        kind=conversation.material_kind,
        title=str(generated.get("title") or conversation.title)[:240],
        language="English",
        prompt=payload.message,
        content=content,
        source_experience_ids=generated_source_ids,
        warnings=list(generated.get("warnings", [])),
        model_info={
            **dict(generated.get("model_info", {})),
            "conversation_id": conversation.id,
            "agent_run_id": run.id,
            "context_manifest_id": manifest.id,
            "verification": verification_report,
        },
        context_manifest_id=manifest.id,
        source_refs={
            "experience_ids": generated_source_ids,
            "loaded_experience_ids": [item.id for item in confirmed],
            "document_ids": [item.id for item in reference_documents],
            "reference_draft_ids": [item.id for item in reference_drafts],
            "official_evidence_ids": [
                item.get("evidence_id")
                for item in official_context.get("evidence", [])
                if item.get("evidence_id")
            ],
            "current_draft_id": current_draft.id if current_draft else None,
            "conversation_id": conversation.id,
        },
        status="draft",
    )
    session.add(draft)
    await session.flush()
    if not draft.root_id:
        draft.root_id = draft.id
    assistant_message = Message(
        owner_id=settings.local_owner_id,
        conversation_id=conversation.id,
        role="assistant",
        content=content,
        sources=[
            {"type": "response_mode", "value": "draft"},
            {"type": "draft", "id": draft.id, "version": version},
            *context_sources,
        ],
    )
    session.add(assistant_message)
    set_material_step(
        run_steps,
        "present_candidate_version",
        "accepted",
        {"draft_id": draft.id, "version": version, "root_id": draft.root_id},
    )
    set_material_step(
        run_steps,
        "adopt_version",
        "pending",
        {"requires_user_confirmation": True, "draft_id": draft.id},
    )
    manifest.status = "consumed"
    run.status = "completed"
    run.stop_reason = "candidate_ready"
    run.final_output = str(generated.get("message") or "已生成候选文稿版本。")
    run.structured_output = {
        "response_type": "draft",
        "draft_id": draft.id,
        "context_manifest_id": manifest.id,
        "verification": verification_report,
        "requires_user_confirmation": True,
    }
    sync_material_plan(run, run_steps)
    conversation.updated_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(conversation)
    return await serialize_writing_conversation(session, conversation)


APPLICATION_STATUSES = {
    "planning",
    "materials",
    "ready",
    "submitted",
    "interview",
    "offer",
    "rejected",
    "withdrawn",
}


def validate_application_status(status: str) -> None:
    if status not in APPLICATION_STATUSES:
        raise HTTPException(422, f"不支持的申请状态：{status}")


@router.get("/applications", response_model=List[ApplicationResponse])
async def list_applications(
    session: AsyncSession = Depends(get_session),
) -> List[ApplicationResponse]:
    items = list(
        (
            await session.scalars(
                select(Application)
                .where(Application.owner_id == settings.local_owner_id)
                .order_by(Application.updated_at.desc())
            )
        ).all()
    )
    return [ApplicationResponse.model_validate(item) for item in items]


@router.post("/applications", response_model=ApplicationResponse, status_code=201)
async def create_application(
    payload: ApplicationCreate, session: AsyncSession = Depends(get_session)
) -> ApplicationResponse:
    validate_application_status(payload.status)
    if not await get_program(session, payload.program_id):
        raise HTTPException(404, "项目不存在")
    package = await session.scalar(select(ApplicationPackage).where(
        ApplicationPackage.owner_id == settings.local_owner_id,
        ApplicationPackage.program_id == payload.program_id,
    ))
    if package:
        await refresh_application_package(session, package)
    if not package or not package.ready:
        raise HTTPException(400, "该项目申请包尚未就绪，请先完成并确认全部必需材料")
    existing = await session.scalar(
        select(Application).where(
            Application.owner_id == settings.local_owner_id,
            Application.program_id == payload.program_id,
        )
    )
    if existing:
        raise HTTPException(409, "该项目已经在申请看板中")
    item = Application(owner_id=settings.local_owner_id, **payload.model_dump())
    session.add(item)
    await session.commit()
    await session.refresh(item)
    return ApplicationResponse.model_validate(item)


@router.patch("/applications/{application_id}", response_model=ApplicationResponse)
async def update_application(
    application_id: str,
    payload: ApplicationUpdate,
    session: AsyncSession = Depends(get_session),
) -> ApplicationResponse:
    item = await session.get(Application, application_id)
    if not item or item.owner_id != settings.local_owner_id:
        raise HTTPException(404, "申请记录不存在")
    values = payload.model_dump(exclude_unset=True)
    if "status" in values and values["status"] is not None:
        validate_application_status(str(values["status"]))
        if values["status"] in {"ready", "submitted", "interview", "offer"}:
            package = await session.scalar(select(ApplicationPackage).where(
                ApplicationPackage.owner_id == settings.local_owner_id,
                ApplicationPackage.program_id == item.program_id,
            ))
            if not package or not package.ready:
                raise HTTPException(400, "项目申请包尚未就绪，不能进入提交阶段")
    for key, value in values.items():
        setattr(item, key, value)
    await session.commit()
    await session.refresh(item)
    return ApplicationResponse.model_validate(item)


@router.get("/tasks", response_model=List[TaskResponse])
async def list_tasks(
    status: str = Query(""), session: AsyncSession = Depends(get_session)
) -> List[TaskResponse]:
    statement = select(Task).where(Task.owner_id == settings.local_owner_id)
    if status:
        statement = statement.where(Task.status == status)
    items = list((await session.scalars(statement.order_by(Task.due_date))).all())
    return [TaskResponse.model_validate(item) for item in items]


@router.post("/tasks", response_model=TaskResponse, status_code=201)
async def post_task(
    payload: TaskCreate, session: AsyncSession = Depends(get_session)
) -> TaskResponse:
    return TaskResponse.model_validate(await create_task(session, payload))


@router.post("/tasks/timeline", response_model=List[TaskResponse], status_code=201)
async def post_timeline(
    payload: TimelineCreate, session: AsyncSession = Depends(get_session)
) -> List[TaskResponse]:
    try:
        items = await create_timeline(session, payload.program_id)
    except ValueError as exc:
        status = 404 if str(exc) == "项目不存在" else 400
        raise HTTPException(status, str(exc)) from exc
    return [TaskResponse.model_validate(item) for item in items]


@router.patch("/tasks/{task_id}", response_model=TaskResponse)
async def patch_task(
    task_id: str, payload: TaskUpdate, session: AsyncSession = Depends(get_session)
) -> TaskResponse:
    item = await session.get(Task, task_id)
    if not item or item.owner_id != settings.local_owner_id:
        raise HTTPException(404, "任务不存在")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(item, key, value)
    await session.commit()
    await session.refresh(item)
    return TaskResponse.model_validate(item)


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest, session: AsyncSession = Depends(get_session)
) -> StreamingResponse:
    async def stream():
        queue: asyncio.Queue = asyncio.Queue()
        task: asyncio.Task

        async def emit(event: str, data: Dict[str, Any]) -> None:
            if event == "run.started" and data.get("conversation_id"):
                active_chat_tasks[str(data["conversation_id"])] = task
            await queue.put((event, data))

        async def execute() -> None:
            try:
                await harness.run(session, payload.message, payload.conversation_id, emit)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                await queue.put(("run.failed", {"error": str(exc)}))
            finally:
                for conversation_id, active_task in list(active_chat_tasks.items()):
                    if active_task is task:
                        active_chat_tasks.pop(conversation_id, None)
                await queue.put(None)

        task = asyncio.create_task(execute())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                event, data = item
                yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
            await task
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.post("/chat/{conversation_id}/cancel")
async def cancel_chat_generation(
    conversation_id: str,
    session: AsyncSession = Depends(get_session),
) -> Dict[str, bool]:
    conversation = await session.get(Conversation, conversation_id)
    if not conversation or conversation.owner_id != settings.local_owner_id:
        raise HTTPException(404, "对话不存在")
    task = active_chat_tasks.get(conversation_id)
    if not task or task.done():
        return {"cancelled": False}
    task.cancel()
    return {"cancelled": True}


@router.get("/agent-runs/{run_id}", response_model=AgentRunResponse)
async def get_agent_run(
    run_id: str, session: AsyncSession = Depends(get_session)
) -> AgentRunResponse:
    item = await session.get(AgentRun, run_id)
    if not item or item.owner_id != settings.local_owner_id:
        raise HTTPException(404, "Agent Run 不存在")
    return AgentRunResponse.model_validate(item)


@router.get("/agent-runs/{run_id}/trace", response_model=TraceResponse)
async def get_agent_trace(
    run_id: str, session: AsyncSession = Depends(get_session)
) -> TraceResponse:
    item = await session.get(AgentRun, run_id)
    if not item or item.owner_id != settings.local_owner_id:
        raise HTTPException(404, "Agent Run 不存在")
    steps = list(
        (
            await session.scalars(
                select(AgentStep).where(AgentStep.run_id == run_id).order_by(AgentStep.position)
            )
        ).all()
    )
    calls = list(
        (
            await session.scalars(
                select(ToolCall).where(ToolCall.run_id == run_id).order_by(ToolCall.created_at)
            )
        ).all()
    )
    return TraceResponse(
        run=AgentRunResponse.model_validate(item),
        steps=[
            {
                "id": step.id,
                "position": step.position,
                "name": step.name,
                "status": step.status,
                "result": step.result,
            }
            for step in steps
        ],
        tool_calls=[
            {
                "id": call.id,
                "step_id": call.step_id,
                "tool_name": call.tool_name,
                "status": call.status,
                "arguments": call.arguments,
                "result": call.result,
                "error": call.error,
                "duration_ms": call.duration_ms,
            }
            for call in calls
        ],
    )


@router.get("/memories", response_model=List[MemoryResponse])
async def list_memories(session: AsyncSession = Depends(get_session)) -> List[MemoryResponse]:
    items = list(
        (
            await session.scalars(
                select(Memory)
                .where(Memory.owner_id == settings.local_owner_id, Memory.active.is_(True))
                .order_by(Memory.updated_at.desc())
            )
        ).all()
    )
    unique: Dict[str, Memory] = {}
    for item in items:
        unique.setdefault(f"{item.memory_type}:{item.key}", item)
    return [MemoryResponse.model_validate(item) for item in unique.values()]


@router.delete("/memories/{memory_id}", status_code=204)
async def delete_memory(memory_id: str, session: AsyncSession = Depends(get_session)) -> None:
    item = await session.get(Memory, memory_id)
    if not item or item.owner_id != settings.local_owner_id:
        raise HTTPException(404, "记忆不存在")
    item.active = False
    await session.commit()


@router.get("/skills", response_model=List[SkillResponse])
async def list_skills() -> List[SkillResponse]:
    return [
        SkillResponse(
            name=item.name,
            version=item.version,
            description=item.description,
            tools=item.tools,
            output_schema=item.output_schema,
            enabled=True,
        )
        for item in skill_registry.list()
    ]


@router.get("/mcp/servers", response_model=List[MCPServerResponse])
async def list_mcp_servers(session: AsyncSession = Depends(get_session)) -> List[MCPServerResponse]:
    items = list((await session.scalars(select(MCPConnection))).all())
    return [
        MCPServerResponse(
            name=item.name,
            transport=item.transport,
            endpoint=item.endpoint,
            read_only=item.read_only,
            status=item.status,
            tools=item.tools,
        )
        for item in items
    ]


@router.post("/mcp/demo/call")
async def call_demo_mcp(
    payload: Dict[str, Any], session: AsyncSession = Depends(get_session)
) -> Dict[str, Any]:
    name = str(payload.get("name", ""))
    allowed = {item.name for item in demo_mcp.list_tools()}
    if name not in allowed:
        raise HTTPException(403, "MCP 工具不在只读白名单中")
    registry_name = {
        "catalog.search_programs": "mcp_catalog_search",
    }.get(name)
    if not registry_name:
        raise HTTPException(501, "该 MCP 示例工具尚未映射到统一 Tool Registry")
    run = AgentRun(
        owner_id=settings.local_owner_id,
        skill_name="mcp-demo",
        skill_version="0.1.0",
        goal=f"调用只读 MCP 工具 {name}",
        status="running",
        plan=[
            {
                "id": "step-1",
                "name": "通过统一 Tool Registry 调用 MCP",
                "status": "running",
                "dependencies": [],
                "expected_output": "只读目录结果与 ToolCall Trace",
            }
        ],
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)
    result = await tool_registry.execute(
        session,
        run.id,
        registry_name,
        payload.get("arguments", {}),
        [registry_name],
    )
    run.status = "completed"
    run.stop_reason = "success"
    run.final_output = "只读 MCP 调用完成"
    run.plan = [{**run.plan[0], "status": "completed"}]
    await session.commit()
    return {
        "server": demo_mcp.name,
        "tool": name,
        "registry_tool": registry_name,
        "run_id": run.id,
        "result": result,
    }
