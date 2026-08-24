import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

QS_RANKING_YEAR = 2026
# Ranking edition: https://www.topuniversities.com/world-university-rankings/2026
# Local table materialized from the public CSV mirror and normalized below:
# https://github.com/reshmaharidhas/Data-Analysis-of-QS-World-University-Rankings-2026
_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "qs_2026_top300.csv"

_COUNTRY_NAMES = {
    "United States of America": "United States",
    "Hong Kong SAR, China": "Hong Kong",
    "Macao SAR, China": "Macao",
    "China (Mainland)": "China",
    "Republic of Korea": "South Korea",
    "Russian Federation": "Russia",
}

_UNIVERSITY_NAMES = {
    # Correct the public table's typo and keep the canonical name used by
    # university pages and the recommendation workflow.
    "Nanyang Technical University, Singapore": "Nanyang Technological University",
}

_UNIVERSITY_ALIASES = {
    "mit": "Massachusetts Institute of Technology (MIT)",
    "caltech": "California Institute of Technology (Caltech)",
    "cmu": "Carnegie Mellon University",
    "ucl": "UCL (University College London)",
    "university college london": "UCL (University College London)",
    "kcl": "King's College London (KCL)",
    "lse": "London School of Economics and Political Science (LSE)",
    "nus": "National University of Singapore (NUS)",
    "national university of singapore": "National University of Singapore (NUS)",
    "ntu": "Nanyang Technological University",
    "nanyang technological university singapore": "Nanyang Technological University",
    "hku": "The University of Hong Kong",
    "cuhk": "The Chinese University of Hong Kong",
    "hkust": "The Hong Kong University of Science and Technology",
    "polyu": "The Hong Kong Polytechnic University",
    "cityu": "City University of Hong Kong",
    "hong kong polytechnic university": "The Hong Kong Polytechnic University",
    "university of illinois urbana champaign": "University of Illinois at Urbana-Champaign",
    "university of michigan": "University of Michigan-Ann Arbor",
    "university of california san diego": "University of California, San Diego (UCSD)",
    "ucsd": "University of California, San Diego (UCSD)",
    "new york university": "New York University (NYU)",
    "nyu": "New York University (NYU)",
    "university of massachusetts amherst": "University of Massachusetts, Amherst",
    "university of maryland": "University of Maryland, College Park",
    "university of california los angeles": "University of California, Los Angeles (UCLA)",
    "ucla": "University of California, Los Angeles (UCLA)",
    "university of california berkeley": "University of California, Berkeley (UCB)",
    "uc berkeley": "University of California, Berkeley (UCB)",
    "ucb": "University of California, Berkeley (UCB)",
}


def _name_key(value: str) -> str:
    value = re.sub(r"\([^)]*\)", " ", value.casefold())
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(token for token in value.split() if token != "the")


def university_name_key(value: str) -> str:
    """Return one stable identity key for official names and common aliases."""
    key = _name_key(value)
    canonical = _UNIVERSITY_ALIASES.get(key)
    return _name_key(canonical) if canonical else key


def same_university(left: str, right: str) -> bool:
    left_key = university_name_key(left)
    right_key = university_name_key(right)
    return bool(left_key and right_key and left_key == right_key)


@lru_cache(maxsize=1)
def qs_2026_top300() -> List[Dict[str, object]]:
    with _DATA_PATH.open(encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            university = _UNIVERSITY_NAMES.get(row["university"], row["university"])
            rows.append({
                "rank": int(row["rank"]),
                "university": university,
                "country": _COUNTRY_NAMES.get(row["country"], row["country"]),
                "city": row["city"],
            })
    return rows


@lru_cache(maxsize=1)
def qs_rank_map() -> Dict[str, int]:
    ranks = {str(item["university"]): int(item["rank"]) for item in qs_2026_top300()}
    for alias, canonical in _UNIVERSITY_ALIASES.items():
        if canonical in ranks:
            ranks[alias] = ranks[canonical]
    return ranks


def qs_rank_for_university(university: str) -> Optional[int]:
    requested = university_name_key(university)
    for item in qs_2026_top300():
        if requested == university_name_key(str(item["university"])):
            return int(item["rank"])
    return None


def ranked_university_options(
    countries: List[str],
    max_qs_rank: Optional[int],
) -> List[Dict[str, object]]:
    country_filter = set(countries)
    rank_limit = min(max_qs_rank or 300, 300)
    return [
        {
            "university": item["university"],
            "country": item["country"],
            "city": item["city"],
            "qs_rank": item["rank"],
        }
        for item in qs_2026_top300()
        if int(item["rank"]) <= rank_limit
        and (not country_filter or str(item["country"]) in country_filter)
    ]
