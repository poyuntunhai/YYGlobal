from dataclasses import dataclass
from typing import Any, Dict, List

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.business import get_program, search_programs_for_profile


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    read_only: bool = True


class DemoCatalogMCPAdapter:
    """Runnable read-only MCP-shaped adapter used to validate discovery and invocation."""

    name = "yyglobal-demo-catalog"
    transport = "in-process-demo-adapter"

    def list_tools(self) -> List[MCPTool]:
        return [
            MCPTool(
                name="catalog.search_programs",
                description="Search the local P0 program catalog",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            MCPTool(
                name="catalog.get_program",
                description="Get one program from the local P0 program catalog",
                input_schema={
                    "type": "object",
                    "properties": {"program_id": {"type": "string"}},
                    "required": ["program_id"],
                    "additionalProperties": False,
                },
            ),
        ]

    async def call_tool(self, session: AsyncSession, name: str, arguments: Dict[str, Any]) -> Any:
        if name == "catalog.search_programs":
            items = await search_programs_for_profile(
                session, query=arguments.get("query", ""), use_profile=False
            )
            return [
                {
                    "id": item.id,
                    "university": item.university,
                    "name": item.name,
                    "degree": item.degree,
                    "country": item.country,
                    "field": item.field,
                    "official_url": item.official_url,
                }
                for item in items[:20]
            ]
        if name == "catalog.get_program":
            item = await get_program(session, arguments["program_id"])
            if not item:
                return {"error": "program_not_found"}
            return {
                "id": item.id,
                "university": item.university,
                "name": item.name,
                "official_url": item.official_url,
            }
        raise KeyError(f"未知 MCP 工具：{name}")


demo_mcp = DemoCatalogMCPAdapter()
