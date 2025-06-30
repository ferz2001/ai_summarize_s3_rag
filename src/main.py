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

    # # Создаем выжимку и сохраняем в Qdrant
    # prompt = "Сделай краткую выжимку из файла `video5m.mp4` и сохрани её в Qdrant."
    # print(f"\n▶️  Отправка запроса агенту: '{prompt}'")

    # response = await agent.ainvoke({"messages": [("user", prompt)]})

    # print("\n--- Ответ агента ---")
    # final_response = response["messages"][-1]
    # if hasattr(final_response, "content"):
    #     print(final_response.content)
    # print("--------------------")
    
    # Демонстрируем поиск
    print("\n🔍 Демонстрация поиска в Qdrant...")
    search_prompt = "Найди информацию про платформу техниум чтобы схожесть была больше 0.1"
    print(f"\n▶️  Поиск: '{search_prompt}'")
    
    search_response = await agent.ainvoke({"messages": [("user", search_prompt)]})
    
    print("\n--- Результаты поиска ---")
    search_final = search_response["messages"][-1]
    if hasattr(search_final, "content"):
        print(search_final.content)
    print("-------------------------")


if __name__ == "__main__":
    asyncio.run(main())
