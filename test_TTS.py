import torch
from TTS.api import TTS
from openai import OpenAI
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Use your API key to authenticate
# OPENAI_API_KEY = 'key'
#
# client = OpenAI(
#     # This is the default and can be omitted
#     api_key=OPENAI_API_KEY,
# )
#
# with open("my/content/hobby_que.mp3", "rb") as audio_file:
#     transcription = client.audio.transcriptions.create(file=audio_file,model="whisper-1")
#
# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "you are an expert on movies."},
#         {"role": "system", "content": "you are a American girl who's 23 and lives in New York. Plus you are a Department of Performing Arts & Media Arts. student."},
#         {"role": "system", "content": "you are always bright, talkative, and busy"},
#         {"role": "user", "content": "What do you think about the movie 'Parasite' "},
#     ],
# )
# print(response.choices[0].message.content)

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
# wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
# Text to speech to a file
text = "Oh, I absolutely loved 'Parasite'!"
# wav = tts.tts(text=text, speaker_wav="my/ai_voice/eunhwa.mp3", language="en")
# print(type(wav))
# sentences = sent_tokenize(text))
# print(sentences)
# i = 0
# for sentence in sentences:
#     tts.tts_to_file(text=sentence, speaker_wav="my/cloning/example_en.wav", language="en", file_path=f"my/output/output_{str(i)}.mp3")
#     i = i+1