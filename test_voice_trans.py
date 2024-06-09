from fastapi import FastAPI, UploadFile, File
import torch
import os
import shutil

app = FastAPI()

@app.post("/upload-ai-voice/")
async def uploadAiVoice(file: UploadFile = File(...)):
    upload_dir = "my/ai_voice"
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename, "filepath": file_path}

if __name__=="__main__":
    import uvicorn
    uvivorn.run(app, host="0.0.0.0", port=5001)