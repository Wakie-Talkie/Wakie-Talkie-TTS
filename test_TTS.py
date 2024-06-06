import torch
from TTS.api import TTS
from openai import OpenAI

# Use your API key to authenticate
OPENAI_API_KEY = 'key'

client = OpenAI(
    # This is the default and can be omitted
    api_key=OPENAI_API_KEY,
)

with open("my/content/hobby_que.mp3", "rb") as audio_file:
    transcription = client.audio.transcriptions.create(file=audio_file,model="whisper-1")

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "you are an expert on movies."},
        {"role": "system", "content": "you are a American girl who's 23 and lives in New York. Plus you are a Department of Performing Arts & Media Arts. student."},
        {"role": "system", "content": "you are always bright, talkative, and busy"},
        {"role": "user", "content": "What do you think about the movie 'Parasite' "},
    ],
)
print(response.choices[0].message.content)

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
# wav = tts.tts(text="안녕하세요!", speaker_wav="TTS/my/cloning/kor.wav", language="ko")
# Text to speech to a file
tts.tts_to_file(text=response.choices[0].message.content, speaker_wav="my/cloning/example_en.wav", language="en", file_path="my/output/output_en_neutral.wav")
# tts.tts_to_file(text="Hi i'm a fourth grade in chung-ang universe. a whole new world", speaker_wav="my/cloning/example_en.wav", language="en", file_path="my/output/output_en_happy.wav",emotion="Happy")
# tts.tts_to_file(text="Hi i'm a fourth grade in chung-ang universe. a whole new world", speaker_wav="my/cloning/example_en.wav", language="en", file_path="my/output/output_en_sad.wav",emotion="Sad")
# tts.tts_to_file(text="Hi i'm a fourth grade in chung-ang universe. a whole new world", speaker_wav="my/cloning/example_en.wav", language="en", file_path="my/output/output_en_angry.wav",emotion="Angry")
# tts.tts_to_file(text="Hi i'm a fourth grade in chung-ang universe. a whole new world", speaker_wav="my/cloning/example_en.wav", language="en", file_path="my/output/output_en_surprise.wav",emotion="Surprise")


