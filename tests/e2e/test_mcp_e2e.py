"""MCP server over real stdio with the official MCP client (model-free config)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ..conftest import SKILLS, write_skill
from .test_hook_e2e import CONFIG

pytestmark = pytest.mark.e2e
mcp = pytest.importorskip("mcp")


def test_mcp_find_skills_and_explain(tmp_path: Path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    si, claude = tmp_path / "si", tmp_path / "claude"
    si.mkdir()
    for name, (desc, body) in SKILLS.items():
        write_skill(claude / "skills", name, desc, body)
    (si / "config.toml").write_text(CONFIG + "\n[sources]\nagents = false\nclaude_plugins = false\n", encoding="utf-8")
    env = {**os.environ, "SKILL_ISSUE_HOME": str(si), "CLAUDE_CONFIG_DIR": str(claude), "SKILL_ISSUE_MCP_WAIT": "60"}

    async def run() -> tuple[list[str], dict, dict, dict]:
        params = StdioServerParameters(command=sys.executable, args=["-m", "skillissue", "mcp"], env=env)
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            tools = [t.name for t in (await s.list_tools()).tools]
            hit = json.loads(
                (
                    await s.call_tool(
                        "find_skills", {"request": "merge two pdf files and pull out the tables", "cwd": str(tmp_path)}
                    )
                )
                .content[0]
                .text
            )
            none = json.loads(
                (await s.call_tool("find_skills", {"request": "thanks, that is all", "cwd": str(tmp_path)}))
                .content[0]
                .text
            )
            why = json.loads(
                (await s.call_tool("explain_route", {"request": "slow sql query", "cwd": str(tmp_path)}))
                .content[0]
                .text
            )
            return tools, hit, none, why

    try:
        tools, hit, none, why = asyncio.run(run())
    finally:
        subprocess.run([sys.executable, "-m", "skillissue", "daemon", "stop"], env=env, capture_output=True)
    assert set(tools) == {"find_skills", "explain_route"}
    assert [s["name"] for s in hit["skills"]] == ["pdf-tools"]
    assert hit["skills"][0]["skill_md"].endswith("SKILL.md")
    assert none["skills"] == []
    assert "sql-tuning" in why["summary"]
