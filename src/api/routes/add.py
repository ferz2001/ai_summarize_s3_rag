from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pathlib import Path
import tempfile
import os

from schemas.common import AddTextRequest, AddDocumentResponse
from services.summary_service import summary_service
from services.rag_service import rag_service

router = APIRouter(prefix="/add", tags=["add"])


@router.post("/text", response_model=AddDocumentResponse)
async def add_text_summary(request: AddTextRequest):
    """
    Создаёт выжимку из текста и добавляет её в RAG систему.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Текст не предоставлен")
    
    try:
        # Создаём выжимку
        summary_result = await summary_service.summarize_text(
            text=request.text,
            file_path=request.file_path or "manual_text_input.txt"
        )
        
        # Добавляем выжимку в RAG
        rag_result = await rag_service.add_summary_to_rag(
            summary_text=summary_result["summary"],
            original_text=request.text,
            file_path=summary_result["file_path"],
            file_type="text_summary"
        )
        
        return AddDocumentResponse(
            success=True,
            message=f"Выжимка текста создана и добавлена в RAG",
            summary=summary_result["summary"],
            file_path=summary_result["file_path"],
            summary_length=summary_result["summary_length"],
            original_length=summary_result["original_length"],
            chunks_count=rag_result["chunks_count"],
            document_ids=rag_result["document_ids"],
            status=rag_result["status"]
        )
        
    except Exception as e:
        print(f"❌ Ошибка добавления текстовой выжимки: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


@router.post("/audio", response_model=AddDocumentResponse)
async def add_audio_summary(
    file: UploadFile = File(...),
    speed_multiplier: float = Form(2.0)
):
    """
    Создаёт выжимку из аудио файла и добавляет её в RAG систему.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не предоставлен")
    
    # Проверяем формат файла
    allowed_formats = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_formats:
        raise HTTPException(
            status_code=400, 
            detail=f"Неподдерживаемый формат аудио: {file_ext}. Поддерживаются: {', '.join(allowed_formats)}"
        )
    
    # Сохраняем файл во временную папку
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Создаём выжимку из аудио
        summary_result = await summary_service.summarize_audio(
            file_path=temp_path,
            speed_multiplier=speed_multiplier
        )
        
        # Добавляем выжимку в RAG
        rag_result = await rag_service.add_summary_to_rag(
            summary_text=summary_result["summary"],
            original_text=summary_result["transcription"],
            file_path=file.filename,
            file_type="audio_summary",
            metadata={
                "transcription": summary_result["transcription"],
                "speed_multiplier": speed_multiplier
            }
        )
        
        return AddDocumentResponse(
            success=True,
            message=f"Выжимка аудио создана и добавлена в RAG",
            summary=summary_result["summary"],
            file_path=file.filename,
            summary_length=summary_result["summary_length"],
            original_length=len(summary_result["transcription"]),
            chunks_count=rag_result["chunks_count"],
            document_ids=rag_result["document_ids"],
            status=rag_result["status"]
        )
        
    except Exception as e:
        print(f"❌ Ошибка добавления аудио выжимки: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки аудио: {str(e)}")
    
    finally:
        # Удаляем временный файл
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@router.post("/video", response_model=AddDocumentResponse)
async def add_video_summary(
    file: UploadFile = File(...),
    speed_multiplier: float = Form(2.0)
):
    """
    Создаёт выжимку из видео файла и добавляет её в RAG систему.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не предоставлен")
    
    # Проверяем формат файла
    allowed_formats = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_formats:
        raise HTTPException(
            status_code=400, 
            detail=f"Неподдерживаемый формат видео: {file_ext}. Поддерживаются: {', '.join(allowed_formats)}"
        )
    
    # Сохраняем файл во временную папку
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Создаём выжимку из видео
        summary_result = await summary_service.summarize_video(
            file_path=temp_path,
            speed_multiplier=speed_multiplier
        )
        
        # Добавляем выжимку в RAG
        rag_result = await rag_service.add_summary_to_rag(
            summary_text=summary_result["summary"],
            original_text=summary_result["transcription"],
            file_path=file.filename,
            file_type="video_summary",
            metadata={
                "transcription": summary_result["transcription"],
                "speed_multiplier": speed_multiplier
            }
        )
        
        return AddDocumentResponse(
            success=True,
            message=f"Выжимка видео создана и добавлена в RAG",
            summary=summary_result["summary"],
            file_path=file.filename,
            summary_length=summary_result["summary_length"],
            original_length=len(summary_result["transcription"]),
            chunks_count=rag_result["chunks_count"],
            document_ids=rag_result["document_ids"],
            status=rag_result["status"]
        )
        
    except Exception as e:
        print(f"❌ Ошибка добавления видео выжимки: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки видео: {str(e)}")
    
    finally:
        # Удаляем временный файл
        if os.path.exists(temp_path):
            os.unlink(temp_path) 