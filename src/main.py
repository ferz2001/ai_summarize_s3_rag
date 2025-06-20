import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from core.config import config

llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY, model="gpt-4o-mini")

client = MultiServerMCPClient(
    {
        "local_http_tools": {
            "url": "http://localhost:9000/mcp/",
            "transport": "streamable_http",
        }
    }
)

async def main():
    print("⏳ Подключение к MCP серверу и получение инструментов...")
    tools = await client.get_tools()
    print("✅ Инструменты получены:", [tool.name for tool in tools])

    agent = create_react_agent(llm, tools)

    prompt = "Сделай краткую выжимку из файла `audio38s.ogg`."
    print(f"\n▶️  Отправка запроса агенту: '{prompt}'")

    response = await agent.ainvoke({"messages": [("user", prompt)]})

    print("\n--- Ответ агента ---")
    final_response = response["messages"][-1]
    if hasattr(final_response, "content"):
        print(final_response.content)
    print("--------------------")


if __name__ == "__main__":
    asyncio.run(main())
