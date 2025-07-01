from fastapi import APIRouter, HTTPException

from schemas.search import SearchRequest, SearchResponse
from services.summary_service import summary_service

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/", response_model=SearchResponse, summary="Поиск выжимок")
async def search_summaries_endpoint(request: SearchRequest):
    """Ищет выжимки по запросу."""
    try:
        results = await summary_service.search_summaries(
            query=request.query,
            limit=request.limit,
            min_score=request.min_score
        )
        
        return SearchResponse(
            results=results,
            total=len(results),
            query=request.query,
            min_score=request.min_score
        )
        
    except Exception as e:
        print(f"❌ Ошибка при поиске: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка поиска: {str(e)}") 