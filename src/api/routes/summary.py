from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pathlib import Path
import tempfile

from schemas.common import SummaryRequest, SummaryResponse
from services.summary_service import summary_service

router = APIRouter(prefix="/summarize", tags=["summarize"])


@router.post("/text", response_model=SummaryResponse)
async def summarize_text(request: SummaryRequest):
    """Создаёт суммаризацию текста."""
    if not request.text:
        raise HTTPException(status_code=400, detail="Текст не предоставлен")
    
    try:
        result = await summary_service.summarize_text(
            text=request.text,
            file_path=request.file_path or "manual_input.txt"
        )
        return SummaryResponse(**result)
    
    except Exception as e:
        print(f"❌ Ошибка суммаризации текста: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка суммаризации: {str(e)}")


@router.post("/audio", response_model=SummaryResponse)
async def summarize_audio(
    file: UploadFile = File(...),
    speed_multiplier: float = Form(2.0)
):
    """Создаёт суммаризацию аудио файла."""
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Файл должен быть аудио")
    
    try:
        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            result = await summary_service.summarize_audio(
                file_path=temp_path,
                speed_multiplier=speed_multiplier
            )
            # Обновляем file_path на оригинальное имя файла
            result["file_path"] = file.filename or "audio_file"
            return SummaryResponse(**result)
        
        finally:
            # Очищаем временный файл
            Path(temp_path).unlink(missing_ok=True)
    
    except Exception as e:
        print(f"❌ Ошибка суммаризации аудио: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка суммаризации аудио: {str(e)}")


@router.post("/video", response_model=SummaryResponse)
async def summarize_video(
    file: UploadFile = File(...),
    speed_multiplier: float = Form(2.0)
):
    """Создаёт суммаризацию видео файла."""
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Файл должен быть видео")
    
    try:
        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            result = await summary_service.summarize_video(
                file_path=temp_path,
                speed_multiplier=speed_multiplier
            )
            # Обновляем file_path на оригинальное имя файла
            result["file_path"] = file.filename or "video_file"
            return SummaryResponse(**result)
        
        finally:
            # Очищаем временный файл
            Path(temp_path).unlink(missing_ok=True)
    
    except Exception as e:
        print(f"❌ Ошибка суммаризации видео: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка суммаризации видео: {str(e)}") 