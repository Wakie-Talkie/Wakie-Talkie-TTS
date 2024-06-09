from fastapi import FastAPI, UploadFile, File, Response
from pydantic import BaseModel
import torch
from TTS.api import TTS
import numpy as np
import os
import shutil


app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

class TextToSpeechRequest(BaseModel):
    voice: str
    language: str
    text: str

@app.post("/upload-ai-voice/")
async def UploadAiVoice(file: UploadFile = File(...)):
    upload_dir = "my/ai_voice"
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"complete-msg": "upload ai-voice complete"}

@app.post("/tts/")
async def TtsAiVoice(request: TextToSpeechRequest):
    text = request.text
    language = request.language
    voice = request.voice

    file_path = f"my/output/{voice}_{language}.mp3"
    if os.path.exists(file_path):
        os.remove(file_path)

    tts.tts_to_file(text=text, speaker_wav=f"my/ai_voice/{voice}.mp3", language=language,file_path=file_path)
    with open(file_path, 'rb') as file:
        audio_data = file.read()
    return Response(content=audio_data, media_type='audio/mpeg')

if __name__=="__main__":
    import uvicorn
    uvicorn.run("test_voice_trans:app", host="0.0.0.0", port=5001)