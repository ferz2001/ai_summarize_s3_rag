#!/usr/bin/env python3
"""
Скрипт для создания выжимок из файлов и сохранения их в Qdrant.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from core.config import config
from qdrant_manager import QdrantManager
from tools import summarize_audio, summarize_video, search_summaries


async def create_summary_with_agent(file_path: str):
    """Создает выжимку с помощью MCP агента."""
    print("⏳ Подключение к MCP серверу...")
    
    llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY, model="gpt-4o-mini")
    
    client = MultiServerMCPClient(
        {
            "local_http_tools": {
                "url": "http://localhost:9000/mcp/",
                "transport": "streamable_http",
            }
        }
    )
    
    tools = await client.get_tools()
    print("✅ Инструменты получены:", [tool.name for tool in tools])

    agent = create_react_agent(llm, tools)

    prompt = f"Сделай краткую выжимку из файла `{file_path}` и сохрани её в Qdrant."
    print(f"\n▶️  Отправка запроса агенту: '{prompt}'")

    response = await agent.ainvoke({"messages": [("user", prompt)]})

    print("\n--- Ответ агента ---")
    final_response = response["messages"][-1]
    if hasattr(final_response, "content"):
        print(final_response.content)
    print("--------------------")


def create_summary_direct(file_path: str):
    """Создает выжимку напрямую без MCP агента."""
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        print(f"❌ Файл {file_path} не найден")
        return
    
    file_extension = file_path_obj.suffix.lower()
    
    try:
        if file_extension in ['.mp3', '.wav', '.ogg', '.flac', '.m4a']:
            print(f"🎵 Обрабатываю аудиофайл: {file_path}")
            summary = summarize_audio(file_path)
        elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
            print(f"🎬 Обрабатываю видеофайл: {file_path}")
            summary = summarize_video(file_path)
        else:
            print(f"❌ Неподдерживаемый формат файла: {file_extension}")
            return
        
        print(f"\n✅ Выжимка создана и сохранена в Qdrant!")
        print(f"📄 Длина выжимки: {len(summary)} символов")
        
    except Exception as e:
        print(f"❌ Ошибка при обработке файла: {e}")


def search_in_qdrant(query: str, limit: int = 5, min_score: float = 0.3):
    """Ищет выжимки в Qdrant."""
    print(f"🔍 Поиск по запросу: '{query}' (мин. схожесть: {min_score})")
    results = search_summaries(query, limit, min_score)
    print(results)


def list_all_summaries():
    """Показывает все сохраненные выжимки."""
    try:
        qdrant_manager = QdrantManager()
        
        # Получаем все точки из коллекции
        results = qdrant_manager.client.scroll(
            collection_name=qdrant_manager.collection_name,
            limit=100,
            with_payload=True
        )
        
        points = results[0]
        
        if not points:
            print("📭 В базе данных нет сохраненных выжимок")
            return
        
        # Группируем точки по session_id для объединения чанков
        session_groups = {}
        
        for point in points:
            payload = point.payload
            session_id = payload.get('session_id', point.id)
            
            if session_id not in session_groups:
                session_groups[session_id] = {
                    'file_name': payload.get('file_name', 'Неизвестно'),
                    'file_type': payload.get('file_type', 'unknown'), 
                    'created_at': payload.get('created_at', 'Неизвестно')[:19],
                    'summary_length': payload.get('summary_length', 0),
                    'chunks': []
                }
            
            session_groups[session_id]['chunks'].append({
                'id': point.id,
                'chunk_index': payload.get('chunk_index', 0),
                'is_chunk': payload.get('is_chunk', False)
            })
        
        print(f"📚 Всего сохранено выжимок: {len(session_groups)} файлов ({len(points)} записей)\n")
        
        for i, (session_id, group) in enumerate(session_groups.items(), 1):
            chunks_count = len(group['chunks'])
            is_chunked = any(chunk['is_chunk'] for chunk in group['chunks'])
            
            print(f"{i}. 📁 {group['file_name']} ({group['file_type']})")
            print(f"   📅 Создана: {group['created_at']}")
            print(f"   📏 Длина: {group['summary_length']} символов")
            
            if is_chunked:
                print(f"   🧩 Чанков: {chunks_count}")
            
            print(f"   🆔 Session ID: {session_id}")
            print()
        
    except Exception as e:
        print(f"❌ Ошибка при получении списка: {e}")


def main():
    parser = argparse.ArgumentParser(description='Создание выжимок и работа с Qdrant')
    
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')
    
    # Команда для создания выжимки
    create_parser = subparsers.add_parser('create', help='Создать выжимку из файла')
    create_parser.add_argument('file_path', help='Путь к файлу')
    create_parser.add_argument('--agent', action='store_true', 
                             help='Использовать MCP агента (по умолчанию - прямой вызов)')
    
    # Команда для поиска
    search_parser = subparsers.add_parser('search', help='Поиск выжимок')
    search_parser.add_argument('query', help='Поисковый запрос')
    search_parser.add_argument('--limit', type=int, default=5, 
                             help='Максимальное количество результатов')
    search_parser.add_argument('--min-score', type=float, default=0.3,
                             help='Минимальный порог схожести (0.0-1.0)')
    
    # Команда для просмотра всех выжимок
    subparsers.add_parser('list', help='Показать все сохраненные выжимки')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'create':
        if args.agent:
            asyncio.run(create_summary_with_agent(args.file_path))
        else:
            create_summary_direct(args.file_path)
    
    elif args.command == 'search':
        search_in_qdrant(args.query, args.limit, args.min_score)
    
    elif args.command == 'list':
        list_all_summaries()


if __name__ == "__main__":
    main() 