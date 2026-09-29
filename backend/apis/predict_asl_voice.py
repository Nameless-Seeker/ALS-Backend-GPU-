from fastapi import APIRouter, UploadFile, File

from backend.apis.audio_process import process_audio, text_to_speech, TTSRequest

router  =APIRouter(tags=["predict_voice"])

@router.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
):
    predicted_text = await process_audio(audio)

    predicted_audio = await text_to_speech(TTSRequest(text = predicted_text))

    return predicted_audio
