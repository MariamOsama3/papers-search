# MCP_tools.py
import shutil
from langchain_mcp_adapters.client import MultiServerMCPClient

servers = {
    "arxiv_server": {
        "transport": "stdio",
        "command": shutil.which("uvx") or "uvx",
        # pin a version once you know which works; optionally set storage path
        "args": ["arxiv-mcp-server"],
    }
}
mcp_client = MultiServerMCPClient(servers)

# Only tools you explicitly allow ever reach the LLM
ALLOWED_LLM_TOOLS = {"read_paper"}   # adjust after printing real names

async def get_mcp_tool(name: str):
    tools = await mcp_client.get_tools()
    return next(t for t in tools if t.name == name)

async def get_llm_mcp_tools():
    tools = await mcp_client.get_tools()
    return [t for t in tools if t.name in ALLOWED_LLM_TOOLS]