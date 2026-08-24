from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "YYGlobal API"
    app_env: str = "development"
    api_prefix: str = "/api"
    auth_enabled: bool = False
    local_owner_id: str = "local-admin"
    database_url: str = "sqlite+aiosqlite:///./data/yyglobal.db"
    upload_dir: Path = Path("./data/uploads")
    max_upload_mb: int = 20

    llm_provider: str = "auto"
    openai_api_key: str = ""
    llm_reasoning_model: str = "gpt-5.6-terra"
    llm_extraction_model: str = "gpt-5.6-luna"
    llm_default_reasoning_effort: str = "medium"
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_reasoning_model: str = "qwen-plus"
    dashscope_extraction_model: str = "qwen-vl-plus"
    llm_request_timeout_seconds: int = 60

    agent_max_steps: int = 8
    agent_max_tool_calls: int = 12
    agent_tool_timeout_seconds: int = 30
    agent_max_tool_retries: int = 2
    agent_context_token_budget: int = 40000
    agent_trace_retention_days: int = 30

    skills_dir: Path = Path(__file__).resolve().parents[1] / "skills"
    enabled_skills_csv: str = Field(
        default=(
            "applicant-profile,program-research,program-compare,shortlist-builder,"
            "cv-planner,ps-planner,application-timeline"
        ),
        alias="ENABLED_SKILLS",
    )

    mcp_enabled: bool = True
    mcp_config_file: Path = Path("./config/mcp_servers.json")
    official_fetch_timeout_seconds: int = 20
    official_fetch_user_agent: str = "YYGlobalBot/0.1 (+educational prototype)"
    program_discovery_model: str = ""
    program_verification_model: str = ""
    program_discovery_timeout_seconds: int = 90
    # The whole discovery tool also performs directory crawling and page
    # verification after the search-model request returns, so it must have a
    # larger budget than the model request itself.
    program_discovery_operation_timeout_seconds: int = 180
    cors_origins_csv: str = "http://localhost:3000"

    @property
    def enabled_skills(self) -> List[str]:
        return [item.strip() for item in self.enabled_skills_csv.split(",") if item.strip()]

    @property
    def cors_origins(self) -> List[str]:
        return [item.strip() for item in self.cors_origins_csv.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
