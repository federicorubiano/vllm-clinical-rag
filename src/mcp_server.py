"""
mcp_server.py
-------------
MCP server that exposes the Clinical Knowledge API to Claude Desktop.
Once configured, you can ask Claude Desktop clinical questions and it
will query your local Merck Manual RAG instead of using training data.

Setup:
    ana mcp setup
    # Follow prompts, then add this server to claude_desktop_config.json

Manual config (add to ~/Library/Application Support/Claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "merck-manual-rag": {
          "command": "/path/to/envs/vllm-clinical-rag/bin/python",
          "args": ["-m", "src.mcp_server"],
          "cwd": "/path/to/vllm-clinical-rag",
          "env": {
            "PYTHONPATH": "/path/to/vllm-clinical-rag"
          }
        }
      }
    }

Get your paths:
    conda activate vllm-clinical-rag
    which python   # → use as "command"
    pwd            # → use as "cwd" and "PYTHONPATH"
"""

import asyncio
import logging
import os
import requests

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions

API_URL = os.getenv("API_URL", "http://localhost:8000")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

server = Server("merck-manual-rag")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="query_merck_manual",
            description=(
                "Answer clinical questions grounded in the Merck Manual Professional Edition. "
                "Use this when the user asks about medical symptoms, diagnoses, treatments, "
                "drug information, clinical protocols, or any healthcare topic. "
                "Returns a structured answer with citations and a medical disclaimer."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The clinical question to answer.",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum response length in tokens (default: 512).",
                        "default": 512,
                    },
                },
                "required": ["question"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name != "query_merck_manual":
        raise ValueError(f"Unknown tool: {name}")

    question   = arguments.get("question", "")
    max_tokens = arguments.get("max_tokens", 512)

    if not question.strip():
        return [types.TextContent(type="text", text="Question cannot be empty.")]

    try:
        resp = requests.post(
            f"{API_URL}/query",
            json={"question": question, "max_tokens": max_tokens},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "No answer returned.")
    except requests.exceptions.ConnectionError:
        answer = (
            "❌ Cannot reach the Clinical Knowledge API. "
            f"Make sure `uvicorn src.api:app` is running at {API_URL}."
        )
    except Exception as e:
        answer = f"❌ Error: {e}"

    return [types.TextContent(type="text", text=answer)]


async def main():
    log.info("Starting Merck Manual RAG MCP server...")
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="merck-manual-rag",
                server_version="2.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
