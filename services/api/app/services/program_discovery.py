import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Set
from urllib.parse import parse_qs, unquote, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.entities import Program
from app.services.business import canonical_countries, program_matches_fields
from app.services.rankings import qs_rank_for_university, ranked_university_options
from app.services.web import FetchedPage, fetch_page

COUNTRY_OFFICIAL_SUFFIXES = {
    "United States": (".edu",),
    "United Kingdom": (".ac.uk",),
    "Canada": (".ca",),
    "Australia": (".edu.au",),
    "Singapore": (".edu.sg",),
    "Hong Kong": (".edu.hk", "hku.hk", "ust.hk", "hkust.edu.hk"),
    "New Zealand": (".ac.nz",),
    "Ireland": (".ie",),
}

COMMON_UNIVERSITY_WORDS = {
    "university", "college", "institute", "technology", "national", "the",
    "of", "and", "school", "hong", "kong", "singapore", "london",
}

# Keep the user's original field while expanding common programme naming
# variants. These aliases are discovery terms; the concrete programme page is
# still verified before it can be released.
BUILTIN_FIELD_ALIASES = {
    "人工智能": (
        "artificial intelligence",
        "applied ai",
        "machine learning",
        "intelligent systems",
    ),
    "artificial intelligence": (
        "artificial intelligence",
        "applied ai",
        "machine learning",
        "intelligent systems",
    ),
    "计算机科学": (
        "computer science",
        "computing",
        "software engineering",
        "computer engineering",
        "data science",
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "cybersecurity",
        "cyber security",
        "information security",
    ),
    "computer science": (
        "computer science",
        "computing",
        "software engineering",
        "computer engineering",
        "data science",
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "cybersecurity",
        "cyber security",
        "information security",
    ),
    "机械工程": (
        "mechanical engineering",
        "mechanics",
        "mechatronics",
        "manufacturing engineering",
    ),
    "mechanical engineering": (
        "mechanical engineering",
        "mechanics",
        "mechatronics",
        "manufacturing engineering",
    ),
    "国际教育": (
        "international education",
        "global education",
        "comparative education",
        "international higher education",
    ),
    "international education": (
        "international education",
        "global education",
        "comparative education",
        "international higher education",
    ),
}


def _normalized_match_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _field_aliases(field: str, suggested: Any = None) -> List[str]:
    values = [field, *BUILTIN_FIELD_ALIASES.get(field.strip().casefold(), ())]
    if isinstance(suggested, list):
        values.extend(str(item) for item in suggested[:8] if isinstance(item, str))
    aliases: List[str] = []
    seen = set()
    for value in values:
        normalized = _normalized_match_text(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            aliases.append(normalized)
    return aliases


def _match_field_text(text: str, aliases_by_field: Dict[str, List[str]]) -> str:
    normalized = f" {_normalized_match_text(text)} "
    best_field = ""
    best_length = 0
    for field, aliases in aliases_by_field.items():
        for alias in aliases:
            if f" {alias} " in normalized and len(alias) > best_length:
                best_field = field
                best_length = len(alias)
    return best_field


def _degree_text_matches(text: str, degree_level: str) -> bool:
    normalized = _normalized_match_text(text)
    patterns = {
        "master": r"\bmaster(?:s)?\b|\bmsc[a-z0-9]*\b|\bmcomp\b|\bmeng\b|\bmcs\b",
        "doctoral": r"\bphd\b|\bdoctoral\b|\bdoctorate\b",
        "undergraduate": r"\bbachelor(?:s)?\b|\bbsc\b|\bbeng\b|\bundergraduate\b",
    }
    pattern = patterns.get(degree_level)
    return bool(pattern and re.search(pattern, normalized))


def _institution_domain(url: str, country: str) -> str:
    hostname = (urlparse(url).hostname or "").casefold().removeprefix("www.")
    labels = hostname.split(".")
    suffix = ".".join(labels[-2:])
    if suffix in {"edu.sg", "edu.hk", "ac.uk", "edu.au", "ac.nz"}:
        return ".".join(labels[-3:]) if len(labels) >= 3 else hostname
    return ".".join(labels[-2:]) if len(labels) >= 2 else hostname

def _canonical_degree_level(value: str) -> str:
    normalized = value.casefold()
    if any(token in normalized for token in ("phd", "ph.d", "doctoral", "doctorate")):
        return "doctoral"
    if any(token in normalized for token in ("bachelor", "undergraduate", "bsc", "b.sc")):
        return "undergraduate"
    if any(token in normalized for token in ("master", "msc", "m.sc", "mcomp", "meng", "mcs")):
        return "master"
    return normalized.strip()


def _school_matches(actual: str, target: str) -> bool:
    actual_value = re.sub(r"[^a-z0-9]+", " ", actual.casefold()).strip()
    target_value = re.sub(r"[^a-z0-9]+", " ", target.casefold()).strip()
    if not target_value:
        return True
    actual_comparable = " ".join(
        token for token in actual_value.split() if token != "the"
    )
    target_comparable = " ".join(
        token for token in target_value.split() if token != "the"
    )
    if actual_comparable == target_comparable:
        return True
    target_tokens = {
        token for token in target_value.split()
        if token not in COMMON_UNIVERSITY_WORDS and len(token) >= 3
    }
    actual_tokens = set(actual_value.split())
    return bool(target_tokens) and target_tokens <= actual_tokens


def _program_name_tokens(value: str) -> Set[str]:
    normalized = value.casefold()
    normalized = re.sub(r"\bmsc\b", "master science", normalized)
    normalized = re.sub(r"\bmcomp\b", "master computing", normalized)
    return {
        token
        for token in re.findall(r"[a-z]{2,}", normalized)
        if token not in {"the", "of", "in", "programme", "program"}
    }


def _program_names_match(actual: str, target: str) -> bool:
    actual_tokens = _program_name_tokens(actual)
    target_tokens = _program_name_tokens(target)
    if not actual_tokens or not target_tokens:
        return False
    # Used for deduplication/exclusion, so partial overlap is unsafe: distinct
    # programmes commonly share generic words such as Master, AI, Data and
    # Science. MSc and "Master of Science" are normalized above, after which
    # the complete identity token set must match.
    return actual_tokens == target_tokens


def _trusted_qs_rank(university: str) -> Optional[int]:
    return qs_rank_for_university(university)


def normalize_program_url(value: str) -> str:
    parsed = urlparse(value.strip())
    scheme = "https" if parsed.scheme in {"http", "https"} else parsed.scheme
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    path = path.rstrip("/") or "/"
    return urlunparse((scheme, parsed.netloc.lower(), path, "", "", ""))


def official_domain_allowed(url: str, country: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    if not hostname:
        return False
    suffixes = COUNTRY_OFFICIAL_SUFFIXES.get(country, ())
    return any(hostname.endswith(suffix) for suffix in suffixes)


def _country_from_official_url(url: str) -> str:
    hostname = (urlparse(url).hostname or "").lower()
    for country, suffixes in COUNTRY_OFFICIAL_SUFFIXES.items():
        if any(hostname.endswith(suffix) for suffix in suffixes):
            return country
    return ""


def _page_supports_identity(page: FetchedPage, candidate: Dict[str, Any]) -> bool:
    searchable = f"{page.title}\n{page.text[:60_000]}".casefold()
    parsed = urlparse(page.url)
    page_identity = _normalized_match_text(f"{page.title} {parsed.path}")
    blocked_page_tokens = {
        "news", "event", "events", "giving", "about", "course", "courses",
        "article", "articles", "story", "stories", "press", "media",
    }
    if set(page_identity.split()) & blocked_page_tokens:
        return False
    path_segments = [segment for segment in parsed.path.split("/") if segment]
    host_labels = (parsed.hostname or "").casefold().split(".")
    project_host = any(
        re.search(r"msc|master|phd|doctoral|bachelor|programme|program", label)
        for label in host_labels[:-2]
    )
    # A department or faculty homepage is not a concrete program page, even if
    # its navigation happens to mention master's programs.
    generic_home_labels = {"www", "cs", "cse", "comp", "seng", "eng", "engineering", "graduate", "grad"}
    if len(path_segments) == 0 and not project_host:
        return False
    if (
        len(path_segments) <= 1
        and not project_host
        and bool(host_labels)
        and host_labels[0] in generic_home_labels
    ):
        return False
    university_tokens = [
        token.casefold()
        for token in re.findall(r"[A-Za-z]{4,}", str(candidate.get("university") or ""))
        if token.casefold() not in COMMON_UNIVERSITY_WORDS
    ]
    if university_tokens and not any(token in searchable for token in university_tokens):
        return False
    degree = str(candidate.get("degree") or "").casefold()
    degree_patterns = {
        "undergraduate": r"\bbachelor(?:'s)?\b|\bb\.?sc\.?\b|\bb\.?s\.?\b|\bundergraduate\b",
        "master": r"\bmaster(?:'s)?\b|\bm\.?sc\.?\b|\bm\.?s\.?\b|\bmcomp\b|\bmcs\b|\bmeng\b",
        "doctoral": r"\bph\.?d\.?\b|\bdoctoral\b|\bdoctorate\b",
    }
    expected_level = _canonical_degree_level(degree)
    candidate_name = str(candidate.get("name") or "")
    if not _degree_text_matches(candidate_name, expected_level):
        return False
    pattern = degree_patterns.get(expected_level)
    if pattern and not re.search(pattern, searchable):
        return False
    candidate_name = candidate_name.casefold()
    url_identity = f"{parsed.hostname or ''}{parsed.path}".casefold()
    primary_identity = f"{page.title}\n{page.text[:5_000]}".casefold()
    if re.search(r"\bmcomp\b|master of computing", candidate_name):
        if not re.search(r"\bmcomp\b|master of computing", primary_identity):
            return False
    elif re.search(r"\bmsc\b|master of science", candidate_name):
        if re.search(r"mcomp[-_/]|/mcomp(?:/|$)", url_identity):
            return False
    if re.search(r"\bmphil\b|master of philosophy", candidate_name):
        if not re.search(r"\bmphil\b|master of philosophy", primary_identity):
            return False
    if "artificial intelligence" in candidate_name:
        if "artificial intelligence" not in searchable:
            return False
        if not re.search(
            r"artificial[-_/]?intelligence|msc[-_]?ai|mscai|mcomp[-_]?ai|/mai(?:/|$)",
            url_identity,
        ):
            return False
    if "computer science" in candidate_name:
        if "computer science" not in searchable:
            return False
        if not re.search(r"computer[-_/]?science|msc[-_]?cs|msccs|/mcs(?:/|$)|/cs(?:/|$)", url_identity):
            return False
    return True


async def verify_program_identity_independently(
    page: FetchedPage, candidate: Dict[str, Any]
) -> Dict[str, Any]:
    discovery_model = settings.program_discovery_model or settings.dashscope_reasoning_model
    verification_model = (
        settings.program_verification_model or settings.dashscope_extraction_model
    )
    if verification_model == discovery_model:
        return {
            "verdict": "rejected",
            "reason_code": "verifier_model_not_independent",
            "message": "项目检索模型与验证模型必须不同。",
            "verification_model": verification_model,
        }
    client = AsyncOpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        timeout=settings.program_discovery_timeout_seconds,
    )
    prompt = {
        "task": "独立验证页面是否为候选项目的具体招生项目页",
        "candidate": {
            "university": candidate.get("university"),
            "name": candidate.get("name"),
            "degree": candidate.get("degree"),
            "field": candidate.get("field"),
            "official_url": candidate.get("official_url"),
            "source_catalog_url": candidate.get("_source_catalog_url"),
        },
        "page": {
            "url": page.url,
            "title": page.title,
            "text": page.text[:30_000],
        },
        "rules": [
            "课程、新闻、活动、About、Giving、学院首页和通用目录必须拒绝",
            "页面必须明确支持项目名称、学校和目标学位层级",
            "evidence_quote 必须逐字来自页面标题或正文",
            "无法确认时返回 uncertain，不能猜测",
        ],
        "output": {
            "verdict": "accepted|rejected|uncertain",
            "reason_code": "program_page|course_page|news_page|directory_page|wrong_program|wrong_degree|wrong_university|insufficient_evidence",
            "evidence_quote": "",
            "message": "",
        },
    }
    response = await client.chat.completions.create(
        model=verification_model,
        messages=[
            {
                "role": "system",
                "content": "你是独立项目身份审计模型。不要相信检索模型结论，只依据提供的官网页面证据判断。只输出 JSON。",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
    )
    result = _clean_json_object(response.choices[0].message.content or "{}")
    quote = str(result.get("evidence_quote") or "").strip()
    page_evidence = f"{page.title}\n{page.text}"
    quote_supported = bool(quote and quote.casefold() in page_evidence.casefold())
    accepted = result.get("verdict") == "accepted" and quote_supported
    return {
        "verdict": "accepted" if accepted else "rejected",
        "reason_code": (
            str(result.get("reason_code") or "program_page")
            if accepted
            else (
                "unsupported_verifier_quote"
                if result.get("verdict") == "accepted" and not quote_supported
                else str(result.get("reason_code") or "insufficient_evidence")
            )
        ),
        "evidence_quote": quote if quote_supported else "",
        "message": str(result.get("message") or ""),
        "verification_model": verification_model,
    }


def _clean_json_object(value: str) -> Dict[str, Any]:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    parsed = json.loads(cleaned)
    return parsed if isinstance(parsed, dict) else {}


async def _recover_catalog_urls(
    failed_urls: List[str],
    *,
    country: str,
    field_aliases: List[str],
    max_pages: int = 18,
) -> tuple[List[str], Dict[str, Any]]:
    """Recover real directory pages from an invalid model-suggested URL.

    The model is useful for identifying the institution and likely subdomain,
    but it is not trusted to invent a working path. This bounded crawl walks
    upward to real parent pages and follows only directory-like official links.
    """
    seeds: List[str] = []
    institution_domains: Set[str] = set()
    for value in failed_urls:
        parsed = urlparse(value)
        if not parsed.hostname:
            continue
        institution_domains.add(_institution_domain(value, country))
        parts = [part for part in parsed.path.split("/") if part]
        # Try the site root and a few real ancestors. This is generic across
        # university CMS structures and avoids school-specific URL templates.
        for length in range(min(3, len(parts)), -1, -1):
            path = "/" + "/".join(parts[:length]) if length else "/"
            candidate = normalize_program_url(
                urlunparse(("https", parsed.netloc, path, "", "", ""))
            )
            if candidate not in seeds:
                seeds.append(candidate)

    directory_terms = {
        "program", "programme", "programs", "programmes", "degree", "degrees",
        "graduate", "postgraduate", "master", "masters", "admission", "admissions",
        "academic", "academics", "faculty", "faculties", "school", "schools",
    }
    blocked_terms = {
        "news", "event", "events", "giving", "about", "alumni", "staff", "people",
        "course", "courses", "handbook", "scholarship", "research-news",
    }

    def link_score(link: Dict[str, Any]) -> int:
        label = str(link.get("label") or "")
        url = normalize_program_url(str(link.get("url") or ""))
        normalized = _normalized_match_text(f"{label} {urlparse(url).path}")
        tokens = set(normalized.split())
        if tokens & blocked_terms:
            return -20
        score = 2 * len(tokens & directory_terms)
        if any(alias in normalized for alias in field_aliases):
            score += 3
        if re.search(r"/program(?:me)?s?(?:/|$)", urlparse(url).path.casefold()):
            score += 4
        return score

    def is_directory_page(page: FetchedPage) -> bool:
        identity = _normalized_match_text(
            f"{page.title} {urlparse(page.url).path}"
        )
        program_like_links = sum(
            1
            for link in page.links
            if link_score(link) >= 4
        )
        return (
            any(term in identity.split() for term in ("program", "programme", "programs", "programmes"))
            and program_like_links >= 2
        )

    # Seed ancestors are useful starting points, but a concrete high-scoring
    # directory link discovered from one of them should pre-empt the remaining
    # generic parents.
    frontier: List[tuple[int, str]] = [(3, seed) for seed in seeds]
    visited: Set[str] = set()
    recovered: List[str] = []
    fetched_trace: List[Dict[str, Any]] = []
    while frontier and len(visited) < max_pages:
        frontier.sort(key=lambda item: item[0], reverse=True)
        _priority, current = frontier.pop(0)
        if current in visited:
            continue
        visited.add(current)
        if not official_domain_allowed(current, country):
            continue
        if institution_domains and _institution_domain(current, country) not in institution_domains:
            continue
        try:
            page = await fetch_page(current)
        except Exception as exc:
            fetched_trace.append({"url": current, "error": type(exc).__name__})
            continue
        final_url = normalize_program_url(page.url)
        fetched_trace.append(
            {"url": final_url, "title": page.title, "link_count": len(page.links)}
        )
        if is_directory_page(page) and final_url not in recovered:
            path_leaf = urlparse(final_url).path.rstrip("/").split("/")[-1].casefold()
            if path_leaf in {"program", "programme", "programs", "programmes"}:
                recovered = [final_url]
                break
            recovered.append(final_url)
            continue
        ranked_links = sorted(page.links, key=link_score, reverse=True)
        for link in ranked_links[:8]:
            if link_score(link) < 2:
                continue
            url = normalize_program_url(str(link.get("url") or ""))
            if (
                url
                and url not in visited
                and official_domain_allowed(url, country)
                and _institution_domain(url, country) in institution_domains
            ):
                frontier.append((link_score(link), url))
    recovered = list(dict.fromkeys(recovered))[:4]
    return recovered, {"failed_urls": failed_urls, "pages": fetched_trace, "recovered": recovered}


async def search_official_program_candidates(
    *,
    countries: List[str],
    fields: List[str],
    degree_level: str,
    target_university: str,
    max_qs_rank: Optional[int],
    excluded_urls: Set[str],
    max_candidates: int,
    school_offset: int = 0,
    include_trace: bool = False,
) -> Any:
    if not settings.dashscope_api_key:
        raise RuntimeError("未配置联网项目发现所需的 DASHSCOPE_API_KEY")
    if target_university:
        scopes = [(target_university, "", fields)]
    else:
        ranked_schools = ranked_university_options(countries, max_qs_rank)
        school_batch = ranked_schools[school_offset : school_offset + 6]
        scopes = [
            (str(school["university"]), str(school["country"]), fields)
            for school in school_batch
        ]
    if not scopes:
        scopes = [(target_university, "", fields)]
    per_scope_limit = max(5, min(12, max_candidates * 2))
    # One scope now represents one university (with all requested fields), so
    # the first school batch can safely locate its directories in parallel.
    semaphore = asyncio.Semaphore(min(6, max(4, len(scopes))))

    async def search_scope(
        school: str, country: str, scope_fields: List[str]
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        client = AsyncOpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            timeout=settings.program_discovery_timeout_seconds,
        )
        location_scope = (
            f"指定学校：{school}。学校优先，不使用国家限制。"
            if school else f"目标国家或地区：{country}。"
        )
        discovery_model = settings.program_discovery_model or settings.dashscope_reasoning_model
        locator_prompt = f"""
当前步骤只定位大学官方项目目录，不查找、不返回具体项目。
{location_scope}
目标专业：{json.dumps(scope_fields, ensure_ascii=False)}。
目标学位层级：{degree_level}。
QS 2026 排名上限：{max_qs_rank if max_qs_rank is not None else '不限'}。

必须遵守：
1. 只返回学校研究生项目总目录和相关学院的研究生项目目录。
2. URL 必须属于学校官网，不能使用排名站、中介、聚合站、新闻页或申请登录页。
3. university 包含学校标准英文名；country 使用标准英文国家或地区名。
4. qs_rank 使用 QS World University Rankings 2026，无法确认时填 null。
5. field_aliases 按用户输入的每个目标专业分别给出常见英文名称及直接相关项目名称；
   不要扩展到明显无关的学科，也不能丢失任何用户输入方向。

只返回 JSON 对象：
{{
  "university":"", "country":"", "city":"", "qs_rank":null,
  "field_aliases":{{"用户输入专业":["English alias or related programme name"]}},
  "catalogs":{{
    "university":[{{"url":"https://...", "title":""}}],
    "faculty":[{{"url":"https://...", "title":""}}]
  }}
}}
""".strip()
        async with semaphore:
            response = await client.chat.completions.create(
                model=discovery_model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是大学官方项目目录定位器。当前步骤禁止返回具体项目，只输出合法 JSON。",
                    },
                    {"role": "user", "content": locator_prompt},
                ],
                response_format={"type": "json_object"},
                extra_body={"enable_search": True},
            )
        locator_payload = _clean_json_object(response.choices[0].message.content or "{}")
        resolved_school = str(locator_payload.get("university") or school).strip()
        resolved_country_values = canonical_countries(
            [str(locator_payload.get("country") or country)]
        )
        resolved_country = resolved_country_values[0] if resolved_country_values else ""
        resolved_city = str(locator_payload.get("city") or "").strip()
        resolved_rank = _trusted_qs_rank(resolved_school)
        if resolved_rank is None and isinstance(locator_payload.get("qs_rank"), int):
            resolved_rank = locator_payload["qs_rank"]
        suggested_aliases = locator_payload.get("field_aliases")
        aliases_by_field = {
            field: _field_aliases(
                field,
                suggested_aliases.get(field)
                if isinstance(suggested_aliases, dict)
                else (suggested_aliases if len(scope_fields) == 1 else None),
            )
            for field in scope_fields
        }
        all_aliases = list(
            dict.fromkeys(
                alias
                for aliases in aliases_by_field.values()
                for alias in aliases
            )
        )
        trace: Dict[str, Any] = {
            "scope": {
                "university": resolved_school,
                "country": resolved_country,
                "fields": scope_fields,
            },
            "locator_model": discovery_model,
            "locator_catalogs": [],
            "catalog_rounds": [],
        }
        catalogs = locator_payload.get("catalogs")
        catalogs = catalogs if isinstance(catalogs, dict) else {}

        queue: List[tuple[str, str, int]] = []
        for catalog_type in ("university", "faculty"):
            rows = catalogs.get(catalog_type)
            if not isinstance(rows, list):
                continue
            for item in rows[:4]:
                if not isinstance(item, dict):
                    continue
                url = normalize_program_url(str(item.get("url") or ""))
                candidate_country = resolved_country or _country_from_official_url(url)
                if url and candidate_country and official_domain_allowed(url, candidate_country):
                    resolved_country = resolved_country or candidate_country
                    queue.append((catalog_type, url, 0))
                    trace["locator_catalogs"].append(
                        {"type": catalog_type, "url": url, "title": str(item.get("title") or "")}
                    )

        seen_catalogs: Set[str] = set()
        catalog_types: Dict[str, str] = {}
        programs: List[Dict[str, Any]] = []
        seen_program_urls: Set[str] = set()

        for catalog_round in range(3):
            current = [item for item in queue if item[2] == catalog_round]
            if not current:
                break
            links_for_model: List[Dict[str, str]] = []
            link_sources: Dict[str, str] = {}
            failed_catalog_urls: List[str] = []
            for catalog_type, catalog_url, _depth in current:
                if catalog_url in seen_catalogs:
                    continue
                seen_catalogs.add(catalog_url)
                try:
                    async with semaphore:
                        page = await fetch_page(catalog_url)
                except Exception as exc:
                    failed_catalog_urls.append(catalog_url)
                    trace["catalog_rounds"].append(
                        {"round": catalog_round + 1, "catalog_url": catalog_url, "error": type(exc).__name__}
                    )
                    continue
                final_catalog_url = normalize_program_url(page.url)
                catalog_types[final_catalog_url] = catalog_type
                for link in page.links:
                    url = normalize_program_url(str(link.get("url") or ""))
                    if (
                        not url
                        or url in excluded_urls
                        or not official_domain_allowed(url, resolved_country)
                        or _institution_domain(url, resolved_country)
                        != _institution_domain(final_catalog_url, resolved_country)
                    ):
                        continue
                    if url not in link_sources:
                        link_sources[url] = final_catalog_url
                        links_for_model.append(
                            {"url": url, "label": str(link.get("label") or "")[:240]}
                        )
            if failed_catalog_urls:
                try:
                    recovered_urls, recovery_trace = await asyncio.wait_for(
                        _recover_catalog_urls(
                            failed_catalog_urls,
                            country=resolved_country,
                            field_aliases=all_aliases,
                            max_pages=12,
                        ),
                        timeout=35,
                    )
                except (TimeoutError, asyncio.TimeoutError):
                    recovered_urls = []
                    recovery_trace = {
                        "failed_urls": failed_catalog_urls,
                        "recovered": [],
                        "error": "catalog_recovery_timeout",
                    }
                trace.setdefault("catalog_recovery", []).append(recovery_trace)
                for url in recovered_urls:
                    if url not in seen_catalogs:
                        path_leaf = urlparse(url).path.rstrip("/").split("/")[-1].casefold()
                        recovered_type = (
                            "university"
                            if path_leaf in {"program", "programme", "programs", "programmes"}
                            else "faculty"
                        )
                        queue.append((recovered_type, url, catalog_round + 1))
            if not links_for_model:
                continue
            links_for_model = links_for_model[:240]
            allowlist = {item["url"] for item in links_for_model}
            selector_prompt = f"""
以下链接由程序从大学官方项目目录中逐条提取。你只能选择输入中已有的 URL，禁止生成、
改写或补充任何 URL。

学校：{resolved_school}
国家或地区：{resolved_country}
目标专业及检索别名：{json.dumps(aliases_by_field, ensure_ascii=False)}
目标学位：{degree_level}

任务：
1. programs 选择符合任一目标专业和学位的独立招生项目页面。
2. next_catalog_urls 选择仍需继续进入的学院、系或研究生项目目录。
3. 新闻、课程、About、Giving、师资页、活动页、申请登录页不能作为项目。
4. 如果链接文字不能确认是独立项目，放入 next_catalog_urls 或忽略，不要猜测。
最多选择 {per_scope_limit} 个项目。

目录链接：
{json.dumps(links_for_model, ensure_ascii=False)}

只返回 JSON：
{{"programs":[{{"url":"输入中的URL","name":"项目正式名称"}}],
"next_catalog_urls":["输入中的URL"]}}
""".strip()
            async with semaphore:
                selected_response = await client.chat.completions.create(
                    model=discovery_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是官方目录阅读器，只能从输入 URL 白名单中选择项目和下一层目录。",
                        },
                        {"role": "user", "content": selector_prompt},
                    ],
                    response_format={"type": "json_object"},
                )
            selected = _clean_json_object(
                selected_response.choices[0].message.content or "{}"
            )
            selected_program_urls: List[str] = []
            for item in selected.get("programs", []):
                if not isinstance(item, dict):
                    continue
                official_url = normalize_program_url(str(item.get("url") or ""))
                if official_url not in allowlist or official_url in seen_program_urls:
                    continue
                source_catalog = link_sources[official_url]
                program_name = str(item.get("name") or "").strip()
                if not _degree_text_matches(program_name, degree_level):
                    continue
                matched_field = _match_field_text(
                    f"{program_name} {official_url}", aliases_by_field
                )
                if not matched_field:
                    continue
                seen_program_urls.add(official_url)
                selected_program_urls.append(official_url)
                university_catalog = next(
                    (
                        url for url, kind in catalog_types.items()
                        if kind == "university"
                    ),
                    "",
                )
                faculty_catalog = (
                    source_catalog if catalog_types.get(source_catalog) == "faculty" else ""
                )
                programs.append(
                    {
                        "university": resolved_school,
                        "name": program_name,
                        "degree": degree_level,
                        "country": resolved_country,
                        "city": resolved_city,
                        "field": matched_field,
                        "official_url": official_url,
                        "qs_rank": resolved_rank,
                        "_field_aliases": aliases_by_field[matched_field],
                        "_university_catalog_url": university_catalog,
                        "_faculty_catalog_url": faculty_catalog,
                        "_source_catalog_url": source_catalog,
                    }
                )
            next_catalog_urls: List[str] = []
            for value in selected.get("next_catalog_urls", []):
                url = normalize_program_url(str(value or ""))
                if url in allowlist and url not in seen_catalogs:
                    next_catalog_urls.append(url)
                    queue.append(("faculty", url, catalog_round + 1))
            trace["catalog_rounds"].append(
                {
                    "round": catalog_round + 1,
                    "catalog_urls": [url for _kind, url, _depth in current],
                    "links_sent_to_model": len(links_for_model),
                    "selected_program_urls": selected_program_urls,
                    "next_catalog_urls": next_catalog_urls,
                    "selector_model": discovery_model,
                }
            )
            if len(programs) >= per_scope_limit:
                break
        return programs[:per_scope_limit], trace

    results = await asyncio.gather(
        *(search_scope(school, country, scope_fields) for school, country, scope_fields in scopes),
        return_exceptions=True,
    )
    candidates: List[Dict[str, Any]] = []
    seen = set()
    successful_results = [result for result in results if isinstance(result, tuple)]
    traces = [result[1] for result in successful_results]
    result_lists = [result[0] for result in successful_results]
    max_result_length = max((len(result) for result in result_lists), default=0)
    # Interleave country/field scopes so one broad scope cannot consume the
    # whole verification budget before another target country is considered.
    for index in range(max_result_length):
        for result in result_lists:
            if index >= len(result):
                continue
            item = result[index]
            key = (
                normalize_program_url(str(item.get("official_url") or "")),
                str(item.get("university") or "").casefold(),
                str(item.get("name") or "").casefold(),
            )
            if key not in seen:
                seen.add(key)
                candidates.append(item)
    output = candidates[: max_candidates * 2]
    return (output, {"mode": "staged_catalog_allowlist", "scopes": traces}) if include_trace else output


async def repair_official_program_urls(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    semaphore = asyncio.Semaphore(3)

    def decode_yahoo_url(value: str) -> str:
        match = re.search(r"/RU=([^/]+)(?:/RK=|$)", value)
        return unquote(match.group(1)) if match else value

    def decode_duckduckgo_url(value: str) -> str:
        parsed = urlparse(value)
        return unquote(parse_qs(parsed.query).get("uddg", [value])[0])

    async def search_result_urls(candidate: Dict[str, Any]) -> List[str]:
        query = (
            f'"{candidate.get("university", "")}" '
            f'"{candidate.get("name", "")}" {candidate.get("degree", "")} official'
        )
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 Chrome/124 Safari/537.36"
            )
        }
        engines = [
            (
                "yahoo",
                "https://search.yahoo.com/search",
                {"p": query},
                "div.algo h3 a[href], ol.searchCenterMiddle a[href]",
            ),
            (
                "duckduckgo",
                "https://html.duckduckgo.com/html/",
                {"q": query},
                "a.result__a[href]",
            ),
        ]
        country_values = canonical_countries([str(candidate.get("country") or "")])
        country = country_values[0] if country_values else ""
        found: List[str] = []
        async with semaphore:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10), follow_redirects=True, headers=headers
            ) as client:
                for engine, endpoint, params, selector in engines:
                    attempts = 2 if engine == "yahoo" else 1
                    for attempt in range(attempts):
                        try:
                            response = await client.get(endpoint, params=params)
                            if response.status_code != 200:
                                if attempt + 1 < attempts:
                                    await asyncio.sleep(0.25)
                                continue
                            soup = BeautifulSoup(response.text, "html.parser")
                            for anchor in soup.select(selector):
                                href = str(anchor.get("href") or "")
                                href = (
                                    decode_yahoo_url(href)
                                    if engine == "yahoo"
                                    else decode_duckduckgo_url(href)
                                )
                                normalized = normalize_program_url(href)
                                if (
                                    normalized.startswith("https://")
                                    and official_domain_allowed(normalized, country)
                                    and normalized not in found
                                ):
                                    found.append(normalized)
                                    if len(found) >= 4:
                                        return found
                            if found:
                                return found
                            break
                        except (httpx.HTTPError, ValueError):
                            if attempt + 1 < attempts:
                                await asyncio.sleep(0.25)
                    if found:
                        return found
        return found

    async def repair_one(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
        urls = await search_result_urls(candidate)
        return [{**candidate, "official_url": url} for url in urls]

    repaired = await asyncio.gather(
        *(repair_one(item) for item in candidates[:8]),
        return_exceptions=True,
    )
    return [
        item
        for result in repaired
        if isinstance(result, list)
        for item in result
        if isinstance(item, dict)
    ]


async def enrich_candidates_from_catalogs(
    candidates: List[Dict[str, Any]],
    *,
    fields: List[str],
    degree_level: str,
    max_candidates: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Add candidates from at most two verified directory levels.

    Direct project search remains the primary path. Directories add coverage
    and user-checkable links, but failure to fetch a directory never removes a
    concrete project candidate.
    """
    if not candidates:
        return [], {"status": "no_direct_candidates", "catalogs": [], "errors": []}

    aliases_by_field: Dict[str, List[str]] = {
        field: _field_aliases(field) for field in fields
    }
    for item in candidates:
        field = str(item.get("field") or "").strip()
        aliases = item.get("_field_aliases")
        if field and isinstance(aliases, list):
            aliases_by_field[field] = _field_aliases(field, aliases)

    school_groups: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for item in candidates:
        country_values = canonical_countries([str(item.get("country") or "")])
        country = country_values[0] if country_values else ""
        school = str(item.get("university") or "").strip()
        if school and country:
            school_groups.setdefault((school, country), []).append(item)

    fetched: Dict[str, FetchedPage] = {}
    fetch_errors: List[Dict[str, str]] = []
    semaphore = asyncio.Semaphore(4)

    async def fetch_catalog(
        url: str, country: str, project_urls: List[str]
    ) -> Optional[FetchedPage]:
        normalized = normalize_program_url(url)
        if not normalized or not official_domain_allowed(normalized, country):
            return None
        project_domains = {
            _institution_domain(value, country) for value in project_urls if value
        }
        if project_domains and _institution_domain(normalized, country) not in project_domains:
            return None
        if normalized in fetched:
            return fetched[normalized]
        try:
            async with semaphore:
                page = await fetch_page(normalized)
        except Exception as exc:
            fetch_errors.append({"url": normalized, "reason": type(exc).__name__})
            return None
        fetched[normalized] = page
        return page

    catalog_meta: Dict[tuple[str, str], Dict[str, List[str]]] = {
        key: {"university": [], "faculty": []} for key in school_groups
    }
    catalog_inputs: Dict[tuple[str, str], Dict[str, Any]] = {}
    for key, items in school_groups.items():
        catalog_inputs[key] = {
            "project_urls": [str(item.get("official_url") or "") for item in items],
            "university_urls": list(
                dict.fromkeys(
                    normalize_program_url(str(item.get("_university_catalog_url") or ""))
                    for item in items
                    if str(item.get("_university_catalog_url") or "").strip()
                )
            )[:1],
            "faculty_urls": list(
                dict.fromkeys(
                    normalize_program_url(str(item.get("_faculty_catalog_url") or ""))
                    for item in items
                    if str(item.get("_faculty_catalog_url") or "").strip()
                )
            )[:3],
        }

    async def fetch_university_level(
        key: tuple[str, str], url: str
    ) -> tuple[tuple[str, str], Optional[FetchedPage]]:
        country = key[1]
        page = await fetch_catalog(url, country, catalog_inputs[key]["project_urls"])
        return key, page

    university_results = await asyncio.gather(
        *(
            fetch_university_level(key, url)
            for key, inputs in catalog_inputs.items()
            for url in inputs["university_urls"]
        )
    )
    university_pages: Dict[tuple[str, str], List[FetchedPage]] = {
        key: [] for key in school_groups
    }
    for key, page in university_results:
        if page:
            normalized = normalize_program_url(page.url)
            catalog_meta[key]["university"].append(normalized)
            university_pages[key].append(page)

    # Level two: follow only explicit, field-related faculty/programme links
    # from the school directory. All schools and faculty pages are fetched with
    # the shared bounded semaphore rather than serially.
    for key, pages in university_pages.items():
        discovered_faculty: List[tuple[int, str]] = []
        for page in pages:
            for link in getattr(page, "links", []):
                if not isinstance(link, dict):
                    continue
                url = normalize_program_url(str(link.get("url") or ""))
                label = str(link.get("label") or "")
                searchable = _normalized_match_text(f"{label} {urlparse(url).path}")
                if not url or "undergraduate" in searchable:
                    continue
                directory_marker = any(
                    marker in searchable
                    for marker in (
                        "graduate programmes",
                        "graduate programs",
                        "postgraduate programmes",
                        "postgraduate programs",
                        "masters programmes",
                        "masters programs",
                        "faculty",
                        "school",
                        "department",
                    )
                )
                if directory_marker and _match_field_text(searchable, aliases_by_field):
                    discovered_faculty.append((100 + len(searchable), url))
        faculty_urls = catalog_inputs[key]["faculty_urls"]
        faculty_urls.extend(
            url
            for _score, url in sorted(discovered_faculty, reverse=True)
            if url not in faculty_urls
        )
        catalog_inputs[key]["faculty_urls"] = faculty_urls[:3]

    async def fetch_faculty_level(
        key: tuple[str, str], url: str
    ) -> tuple[tuple[str, str], Optional[FetchedPage]]:
        country = key[1]
        page = await fetch_catalog(url, country, catalog_inputs[key]["project_urls"])
        return key, page

    faculty_results = await asyncio.gather(
        *(
            fetch_faculty_level(key, url)
            for key, inputs in catalog_inputs.items()
            for url in inputs["faculty_urls"]
        )
    )
    for key, page in faculty_results:
        if page:
            catalog_meta[key]["faculty"].append(normalize_program_url(page.url))
    for meta in catalog_meta.values():
        meta["university"] = list(dict.fromkeys(meta["university"]))
        meta["faculty"] = list(dict.fromkeys(meta["faculty"]))

    # Direct project candidates remain valid even if a directory is unavailable.
    # Attach only directory URLs that were actually fetched successfully.
    enriched: List[Dict[str, Any]] = []
    for item in candidates:
        country_values = canonical_countries([str(item.get("country") or "")])
        country = country_values[0] if country_values else ""
        key = (str(item.get("university") or "").strip(), country)
        meta = catalog_meta.get(key, {"university": [], "faculty": []})
        proposed_university = normalize_program_url(
            str(item.get("_university_catalog_url") or "")
        )
        proposed_faculty = normalize_program_url(str(item.get("_faculty_catalog_url") or ""))
        enriched.append(
            {
                **item,
                "_university_catalog_url": (
                    proposed_university
                    if proposed_university in meta["university"]
                    else (meta["university"][0] if meta["university"] else "")
                ),
                "_faculty_catalog_url": (
                    proposed_faculty
                    if proposed_faculty in meta["faculty"]
                    else (meta["faculty"][0] if meta["faculty"] else "")
                ),
            }
        )

    # Enumerate explicit project links from verified school/faculty directories.
    # There is no generic graph crawl beyond these two levels.
    for (school, country), items in school_groups.items():
        template = items[0]
        meta = catalog_meta.get((school, country), {"university": [], "faculty": []})
        for catalog_url in [*meta["university"], *meta["faculty"]]:
            page = fetched.get(catalog_url)
            if not page:
                continue
            for link in getattr(page, "links", []):
                if not isinstance(link, dict):
                    continue
                official_url = normalize_program_url(str(link.get("url") or ""))
                label = re.sub(r"\s+", " ", str(link.get("label") or "")).strip()
                searchable = f"{label} {urlparse(official_url).path}"
                matched_field = _match_field_text(searchable, aliases_by_field)
                if (
                    not official_url
                    or not label
                    or not matched_field
                    or not _degree_text_matches(searchable, degree_level)
                    or not official_domain_allowed(official_url, country)
                    or _institution_domain(official_url, country)
                    != _institution_domain(str(template.get("official_url") or ""), country)
                ):
                    continue
                enriched.append(
                    {
                        "university": school,
                        "name": label[:240],
                        "degree": degree_level,
                        "country": country,
                        "city": str(template.get("city") or ""),
                        "field": matched_field,
                        "official_url": official_url,
                        "qs_rank": template.get("qs_rank"),
                        "_university_catalog_url": (
                            meta["university"][0] if meta["university"] else ""
                        ),
                        "_faculty_catalog_url": (
                            catalog_url if catalog_url in meta["faculty"] else ""
                        ),
                        "_catalog_verified": True,
                    }
                )

    deduped: List[Dict[str, Any]] = []
    seen_urls = set()
    for item in enriched:
        url = normalize_program_url(str(item.get("official_url") or ""))
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(item)
    report = {
        "status": "completed",
        "mode": "direct_plus_two_level_catalog",
        "catalogs": [
            {
                "url": url,
                "title": page.title,
                "link_count": len(getattr(page, "links", [])),
            }
            for url, page in fetched.items()
        ],
        "errors": fetch_errors,
        "direct_candidate_count": len(candidates),
        "catalog_candidate_count": max(0, len(deduped) - len(candidates)),
    }
    return deduped[: max_candidates * 2], report


async def discover_official_programs(
    session: AsyncSession,
    *,
    countries: List[str],
    fields: List[str],
    degree_level: str = "master",
    target_university: str = "",
    max_qs_rank: Optional[int] = 100,
    excluded_program_ids: Optional[Set[str]] = None,
    max_candidates: int = 15,
    school_offset: int = 0,
    include_report: bool = False,
) -> Any:
    excluded_program_ids = excluded_program_ids or set()
    excluded_programs = list(
        (
            await session.scalars(
                select(Program).where(Program.id.in_(excluded_program_ids))
            )
        ).all()
    ) if excluded_program_ids else []
    excluded_urls = {normalize_program_url(item.official_url) for item in excluded_programs}
    excluded_identities = {
        (item.university.casefold().strip(), item.name.casefold().strip())
        for item in excluded_programs
    }
    canonical_targets = canonical_countries(countries)
    search_result = await search_official_program_candidates(
        countries=canonical_targets,
        fields=fields,
        degree_level=degree_level,
        target_university=target_university.strip(),
        max_qs_rank=max_qs_rank,
        excluded_urls=excluded_urls,
        max_candidates=max_candidates,
        school_offset=school_offset,
        include_trace=True,
    )
    if isinstance(search_result, tuple):
        raw_candidates, staged_discovery_trace = search_result
    else:
        raw_candidates, staged_discovery_trace = search_result, {}
    # The staged search has already fetched the official catalog, extracted its
    # real links and constrained the model to that allowlist. Running the old
    # two-level enrichment here repeated the same network traversal and was the
    # main source of avoidable request timeouts for multi-school searches.
    catalog_report = {
        "status": "completed",
        "mode": "staged_catalog_allowlist",
        "candidate_count": len(raw_candidates),
    }
    cached_programs = list((await session.scalars(select(Program))).all())

    semaphore = asyncio.Semaphore(5)
    identity_audits: Dict[str, Dict[str, Any]] = {}
    async def inspect_candidate(candidate: Dict[str, Any]):
        url = normalize_program_url(str(candidate.get("official_url") or ""))
        country_values = canonical_countries([str(candidate.get("country") or "")])
        country = country_values[0] if country_values else ""
        if not url or url in excluded_urls or not official_domain_allowed(url, country):
            return None
        if not target_university and canonical_targets and country not in canonical_targets:
            return None
        if target_university and not _school_matches(
            str(candidate.get("university") or ""), target_university
        ):
            return None
        if any(
            item.country == country
            and (
                _school_matches(item.university, str(candidate.get("university") or ""))
                or _school_matches(str(candidate.get("university") or ""), item.university)
            )
            and _program_names_match(item.name, str(candidate.get("name") or ""))
            for item in excluded_programs
        ):
            return None
        if _canonical_degree_level(str(candidate.get("degree") or "")) != degree_level:
            return None
        rank = _trusted_qs_rank(str(candidate.get("university") or ""))
        if rank is None:
            rank = candidate.get("qs_rank")
        if max_qs_rank is not None:
            if not isinstance(rank, int) or rank < 1 or rank > max_qs_rank:
                return None
        probe = Program(
            university=str(candidate.get("university") or "").strip(),
            name=str(candidate.get("name") or "").strip(),
            degree=str(candidate.get("degree") or degree_level).strip(),
            country=country,
            city=str(candidate.get("city") or "").strip(),
            field=str(candidate.get("field") or "").strip(),
            official_url=url,
        )
        if not probe.university or not probe.name or not program_matches_fields(probe, fields):
            return None
        cached_match = next(
            (
                item
                for item in cached_programs
                if item.id not in excluded_program_ids
                and item.country == country
                and _canonical_degree_level(item.degree) == degree_level
                and (
                    _school_matches(item.university, probe.university)
                    or _school_matches(probe.university, item.university)
                )
                and _program_names_match(item.name, probe.name)
                and program_matches_fields(item, fields)
            ),
            None,
        )
        urls_to_try = [url]
        if cached_match:
            cached_url = normalize_program_url(cached_match.official_url)
            if cached_url not in urls_to_try and cached_url not in excluded_urls:
                urls_to_try.append(cached_url)
        page = None
        for candidate_url in urls_to_try:
            try:
                async with semaphore:
                    fetched = await fetch_page(candidate_url)
                if _page_supports_identity(fetched, candidate):
                    try:
                        async with semaphore:
                            independent_audit = await verify_program_identity_independently(
                                fetched, candidate
                            )
                    except Exception as exc:
                        independent_audit = {
                            "verdict": "rejected",
                            "reason_code": "verifier_unavailable",
                            "message": type(exc).__name__,
                            "verification_model": (
                                settings.program_verification_model
                                or settings.dashscope_extraction_model
                            ),
                        }
                    identity_audits[normalize_program_url(fetched.url)] = independent_audit
                    if independent_audit.get("verdict") == "accepted":
                        page = fetched
                        break
            except Exception:
                continue
        if page is None:
            return None
        final_url = normalize_program_url(page.url)
        if final_url in excluded_urls or not official_domain_allowed(final_url, country):
            return None
        return candidate, page, probe, rank

    inspected = await asyncio.gather(*(inspect_candidate(item) for item in raw_candidates))
    failed_candidates = [
        candidate
        for candidate, result in zip(raw_candidates, inspected)
        if result is None
        and normalize_program_url(str(candidate.get("official_url") or ""))
        not in excluded_urls
        and (
            str(candidate.get("university") or "").casefold().strip(),
            str(candidate.get("name") or "").casefold().strip(),
        )
        not in excluded_identities
    ]
    if failed_candidates:
        repaired_candidates = await repair_official_program_urls(failed_candidates)
        repaired_results = await asyncio.gather(
            *(inspect_candidate(item) for item in repaired_candidates)
        )
        inspected.extend(repaired_results)
    discovered: List[Program] = []
    seen_urls: Set[str] = set()
    seen_identities: Set[tuple] = set()
    for result in inspected:
        if result is None:
            continue
        candidate, page, probe, rank = result
        final_url = normalize_program_url(page.url)
        identity = (probe.university.casefold().strip(), probe.name.casefold().strip())
        if final_url in seen_urls or identity in seen_identities:
            continue
        seen_urls.add(final_url)
        seen_identities.add(identity)
        existing = await session.scalar(
            select(Program).where(
                or_(
                    Program.official_url == final_url,
                    (Program.university == probe.university) & (Program.name == probe.name),
                )
            )
        )
        # A verified official URL identifies a concrete programme. Do not
        # merge it into another cached row merely because programme names share
        # generic words such as "AI", "Data" or "Science".
        program = existing or Program()
        program.university = probe.university
        program.name = probe.name
        program.degree = probe.degree
        program.country = probe.country
        program.city = probe.city
        program.field = probe.field
        program.qs_rank = rank if isinstance(rank, int) else None
        program.qs_ranking_year = 2026 if program.qs_rank is not None else None
        program.official_url = final_url
        program.catalog_url = str(candidate.get("_university_catalog_url") or "")
        program.faculty_catalog_url = str(candidate.get("_faculty_catalog_url") or "")
        program.summary = "由联网项目发现流程定位官方项目页；申请要求需继续通过官网核验。"
        program.active = True
        if existing is None:
            session.add(program)
        await session.flush()
        discovered.append(program)
    await session.commit()
    verified_candidate_count = len(discovered)
    discovered = discovered[:max_candidates]
    report = {
        **catalog_report,
        "staged_discovery": staged_discovery_trace,
        "identity_audits": [
            {"url": url, **audit} for url, audit in identity_audits.items()
        ],
        "requested_count": max_candidates,
        "school_offset": school_offset,
        "candidate_count": len(raw_candidates),
        "verified_candidate_count": verified_candidate_count,
        "verified_count": len(discovered),
        "schools": [],
    }
    return (discovered, report) if include_report else discovered
