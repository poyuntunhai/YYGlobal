import asyncio
import hashlib
import ipaddress
import socket
from dataclasses import dataclass, field
from typing import Dict, List
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup

from app.core.config import settings


class UnsafeUrlError(ValueError):
    pass


async def validate_public_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafeUrlError("只允许公开 HTTP(S) 地址")
    if parsed.username or parsed.password:
        raise UnsafeUrlError("URL 不允许包含认证信息")

    def resolve() -> list:
        return socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)

    try:
        addresses = await asyncio.to_thread(resolve)
    except socket.gaierror as exc:
        raise UnsafeUrlError("域名无法解析") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise UnsafeUrlError("不允许访问本机、内网或保留地址")


@dataclass
class FetchedPage:
    url: str
    title: str
    text: str
    content_hash: str
    related_links: List[Dict[str, str]]
    links: List[Dict[str, str]] = field(default_factory=list)


async def fetch_page(url: str) -> FetchedPage:
    await validate_public_http_url(url)
    headers = {"User-Agent": settings.official_fetch_user_agent}
    timeout = httpx.Timeout(settings.official_fetch_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()
        final_url = str(response.url)
        await validate_public_http_url(final_url)
    html = response.text
    extracted = trafilatura.extract(html, include_links=False, include_tables=True) or ""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else final_url
    if not extracted:
        extracted = soup.get_text("\n", strip=True)
    text = extracted[:200_000]
    related_links = []
    links = []
    seen = set()
    all_seen = set()
    keywords = (
        "admission",
        "application",
        "apply",
        "requirement",
        "deadline",
        "tuition",
        "fee",
        "cost",
        "financial",
        "how to apply",
    )
    origin = urlparse(final_url)
    origin_domain = ".".join((origin.hostname or "").lower().split(".")[-2:])
    origin_segments = [
        segment
        for segment in origin.path.lower().split("/")
        if segment
        and segment not in {"admission", "admissions", "program", "programs", "graduate", "masters"}
    ]
    program_markers = origin_segments[:2]
    for anchor in soup.select("a[href]"):
        label = anchor.get_text(" ", strip=True)
        candidate = urljoin(final_url, anchor.get("href", "")).split("#", 1)[0]
        parsed = urlparse(candidate)
        candidate_domain = ".".join((parsed.hostname or "").lower().split(".")[-2:])
        if (
            parsed.scheme in {"http", "https"}
            and candidate_domain == origin_domain
            and candidate != final_url
            and candidate not in all_seen
        ):
            all_seen.add(candidate)
            links.append({"url": candidate, "label": label[:300]})
        searchable = f"{label} {parsed.path} {parsed.query}".lower()
        same_program_path = not program_markers or any(
            marker in searchable for marker in program_markers
        )
        if (
            parsed.scheme in {"http", "https"}
            and candidate_domain == origin_domain
            and candidate != final_url
            and candidate not in seen
            and same_program_path
            and any(keyword in searchable for keyword in keywords)
        ):
            seen.add(candidate)
            related_links.append({"url": candidate, "label": label[:200]})
        if len(related_links) >= 20:
            break
    return FetchedPage(
        url=final_url,
        title=title[:300],
        text=text,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        related_links=related_links,
        links=links[:800],
    )
