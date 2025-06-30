from faster_whisper import WhisperModel

model = WhisperModel("turbo", device="cpu", compute_type="int8") 

# Транскрипция файла
segments, info = model.transcribe("audio38s.ogg", language='ru')

# Вывод результата
for segment in segments:
    print(f"[{segment.start:.2f}s → {segment.end:.2f}s] {segment.text}")