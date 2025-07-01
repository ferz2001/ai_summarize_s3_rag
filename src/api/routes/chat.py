from fastapi import APIRouter, HTTPException

from schemas.common import ChatRequest, RAGResponse
from services.rag_service import rag_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=RAGResponse)
async def chat(request: ChatRequest):
    """
    Основной чат с RAG - ищет релевантные документы и генерирует ответ.
    """
    try:
        result = await rag_service.query(
            question=request.message,
            k=request.max_results or 5,
            score_threshold=request.min_score or 0.3
        )
        
        return RAGResponse(
            answer=result["answer"],
            sources=result["sources"],
            context_used=result["context_used"],
            documents_found=result["documents_found"],
            query=request.message
        )
        
    except Exception as e:
        print(f"❌ Ошибка RAG чата: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка чата: {str(e)}") 