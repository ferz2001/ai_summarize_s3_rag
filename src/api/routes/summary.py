from fastapi import APIRouter, HTTPException, UploadFile, File, Query, BackgroundTasks

from schemas.summary import TextSummaryRequest, SummaryResponse, ListSummariesResponse
from services.summary_service import summary_service
from services.file_service import file_service

router = APIRouter(prefix="/summarize", tags=["summaries"])


@router.post("/text", response_model=SummaryResponse, summary="Создать выжимку из текста")
async def summarize_text_endpoint(request: TextSummaryRequest):
    """Создает выжимку из переданного текста."""
    try:
        return await summary_service.create_text_summary(
            text=request.text,
            file_path=request.file_path
        )
    except Exception as e:
        print(f"❌ Ошибка при создании выжимки из текста: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка создания выжимки: {str(e)}")


@router.post("/audio", response_model=SummaryResponse, summary="Создать выжимку из аудиофайла")
async def summarize_audio_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Аудиофайл для обработки"),
    language: str = Query(default="ru", description="Язык аудио"),
    speed_multiplier: float = Query(default=2.0, description="Множитель скорости обработки (2.0 = в 2 раза быстрее)", ge=0.5, le=4.0)
):
    """Создает выжимку из загруженного аудиофайла."""
    try:
        # Проверяем тип файла
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Загруженный файл не является аудиофайлом")
        
        # Сохраняем файл в постоянную папку
        saved_path = await file_service.save_uploaded_file(file, file_type="audio")
        
        # Планируем очистку только если это временные файлы
        background_tasks.add_task(file_service.cleanup_temp_files)
        
        return await summary_service.create_audio_summary(
            file_path=saved_path,
            file_name=file.filename,
            language=language,
            speed_multiplier=speed_multiplier
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Ошибка при обработке аудио: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки аудио: {str(e)}")


@router.post("/video", response_model=SummaryResponse, summary="Создать выжимку из видеофайла")
async def summarize_video_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Видеофайл для обработки"),
    language: str = Query(default="ru", description="Язык аудио в видео"),
    speed_multiplier: float = Query(default=2.0, description="Множитель скорости обработки (2.0 = в 2 раза быстрее)", ge=0.5, le=4.0)
):
    """Создает выжимку из загруженного видеофайла."""
    try:
        # Проверяем тип файла
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="Загруженный файл не является видеофайлом")
        
        # Сохраняем видеофайл в постоянную папку
        saved_path = await file_service.save_uploaded_file(file, file_type="video")
        
        # Планируем очистку только временных файлов
        background_tasks.add_task(file_service.cleanup_temp_files)
        
        return await summary_service.create_video_summary(
            video_path=saved_path,
            file_name=file.filename,
            language=language,
            speed_multiplier=speed_multiplier
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Ошибка при обработке видео: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки видео: {str(e)}")


@router.get("/", response_model=ListSummariesResponse, summary="Список всех выжимок")
async def list_summaries_endpoint():
    """Возвращает список всех сохраненных выжимок."""
    try:
        summaries = await summary_service.get_all_summaries()
        
        return ListSummariesResponse(
            summaries=summaries,
            total_files=len(summaries),
            total_records=len(summaries)  # Упрощенно, можно добавить подсчет записей
        )
        
    except Exception as e:
        print(f"❌ Ошибка при получении списка: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения списка: {str(e)}") 