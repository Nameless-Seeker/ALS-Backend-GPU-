from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
from backend.apis.predict_asl_voice import router as predict_voice_router

app = FastAPI(
    title="ALS Speech Recognition API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Transcription", "X-TTS-Available"]
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "whisper-small-torgo",
    }


app.include_router(predict_voice_router)
