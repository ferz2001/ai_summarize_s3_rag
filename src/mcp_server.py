from fastmcp import FastMCP

from tools import (
    summarize_audio, 
    summarize_text, 
    summarize_video,
    search_summaries,
    summarize_text_and_save
)

mcp = FastMCP("My MCP Server")

mcp.tool(summarize_audio)
mcp.tool(summarize_video)
mcp.tool(summarize_text)
mcp.tool(search_summaries)
mcp.tool(summarize_text_and_save)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=9000)
