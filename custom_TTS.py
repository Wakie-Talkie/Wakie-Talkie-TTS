from fastapi import FastAPI, UploadFile, File, Response, HTTPException
from pydantic import BaseModel
import torch
from TTS.api import TTS
import numpy as np
import os
import shutil
import mimetypes
import time

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

class FindFileRequest(BaseModel):
    name: str

CUSTOM_MIME_TYPE_MAPPING = {
    'audio/wave': '.wav',
    'audio/x-wav': '.wav',
    'audio/x-pn-wav': '.wav'
}

@app.post("/upload-ai-voice/")
async def UploadAiVoice(file: UploadFile = File(...)):
    upload_dir = "my/ai_voice/"
    os.makedirs(os.path.dirname(upload_dir), exist_ok=True)
    extension = mimetypes.guess_extension(file.content_type)
    if not extension:
        # Check custom MIME type mapping
        extension = CUSTOM_MIME_TYPE_MAPPING.get(file.content_type)
    if not extension:
        return {"error": "Unsupported file type"}
    os.makedirs(os.path.dirname(upload_dir), exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename + extension)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"complete-msg": "upload ai-voice complete"}

def find_file_by_name(file_name: str) -> str:
    upload_dir = "my/ai_voice/"
    matching_files = []
    for file in os.listdir(upload_dir):
        if os.path.splitext(file)[0] == file_name:
            matching_files.append(os.path.join(upload_dir, file))
    if len(matching_files) == 0:
        print("no file!\n")
        return "no_file_found"
    print(matching_files[0])
    return matching_files[0]

@app.post("/tts/")
async def TtsAiVoice(request: TextToSpeechRequest):
    text = request.text
    language = request.language
    voice = request.voice

    print(f"voice {voice} language {language} text {text}")
    start_time = time.time()
    file_path = f"my/output/{voice}_{language}.wav"
    if os.path.exists(file_path):
        os.remove(file_path)

    os.makedirs(os.path.dirname("my/output/"), exist_ok=True)
    file_path = os.path.join("my/output/", f"{voice}_{language}.wav")

    end_time = time.time()
    elasped_time = end_time - start_time
    print(f"make output file path : {elasped_time}")

    start_time = time.time()
    speaker_audio_path = find_file_by_name(voice)
    print(speaker_audio_path)
    print(language)
    end_time = time.time()
    elasped_time = end_time - start_time
    print(f"find voice file path : {elasped_time}")
    if language == "zh":
        language = "zh-cn"
    start_time = time.time()
    tts.tts_to_file(text=text, speaker_wav=speaker_audio_path, language=language, speed=1.5,file_path=file_path)
    with open(file_path, 'rb') as file:
        audio_data = file.read()

    end_time = time.time()
    elasped_time = end_time - start_time
    print(f"generate tts audio data : {elasped_time}")
    # 'application/octet-stream'
    # print(file.content_type)
    return Response(content=audio_data, media_type='audio/wav')

if __name__=="__main__":
    import uvicorn
    uvicorn.run("test_voice_trans:app", host="0.0.0.0", port=5001)