import os
import tempfile
from dotenv import load_dotenv
import httpx
from fastapi import UploadFile,Response, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from backend.ml.whisper import WhisperTranscriber

# Load once when the server starts
transcriber = WhisperTranscriber()

async def process_audio(
    audio: UploadFile,
) -> str:

    temp_path = None

    try:

        suffix = os.path.splitext(
            audio.filename or ".wav"
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_file.write(
                await audio.read()
            )

            temp_path = temp_file.name

        # Actual ML inference
        text = transcriber.transcribe(
            temp_path
        )

        return text

    finally:

        if (
            temp_path
            and os.path.exists(temp_path)
        ):
            os.remove(temp_path)

load_dotenv()

PIPER_URL = os.getenv("PIPER_URL")
PIPER_HEALTH_URL = os.getenv("PIPER_HEALTH_URL")

#The text to speech conversion by piper
class TTSRequest(BaseModel):
    text: str

async def text_to_speech(request: TTSRequest):

    # --------------------------------------------------
    # 1. Check Piper configuration
    # --------------------------------------------------

    if not PIPER_URL or not PIPER_HEALTH_URL:

        return Response(
            content=b"",
            media_type="audio/wav",
            headers={
                "X-Transcription": request.text,
                "X-TTS-Available": "false",
            },
        )

    # --------------------------------------------------
    # 2. Check if Piper is alive
    # --------------------------------------------------

    try:

        async with httpx.AsyncClient() as client:

            health_response = await client.get(
                PIPER_HEALTH_URL,
                timeout=2.0,
            )

            if health_response.status_code != 200:

                return Response(
                    content=b"",
                    media_type="audio/wav",
                    headers={
                        "X-Transcription": request.text,
                        "X-TTS-Available": "false",
                    },
                )

    except httpx.RequestError:

        # Piper container is not available
        return Response(
            content=b"",
            media_type="audio/wav",
            headers={
                "X-Transcription": request.text,
                "X-TTS-Available": "false",
            },
        )

    # --------------------------------------------------
    # 3. Piper is alive → stream the generated audio
    # --------------------------------------------------

    timeout = httpx.Timeout(
        connect=10.0,
        read=120.0,
        write=30.0,
        pool=30.0,
    )

    client = httpx.AsyncClient(timeout=timeout)

    try:

        stream = client.stream(
            "POST",
            PIPER_URL,
            json={
                "text": request.text,
            },
        )

        response = await stream.__aenter__()

        if response.status_code != 200:

            await stream.__aexit__(None, None, None)
            await client.aclose()

            return Response(
                content=b"",
                media_type="audio/wav",
                headers={
                    "X-Transcription": request.text,
                    "X-TTS-Available": "false",
                },
            )

        async def audio_stream():

            try:

                async for chunk in response.aiter_bytes():
                    yield chunk

            finally:

                await stream.__aexit__(
                    None,
                    None,
                    None,
                )

                await client.aclose()

        return StreamingResponse(
            audio_stream(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline",
                "X-Transcription": request.text,
                "X-TTS-Available": "true",
            },
        )

    except httpx.RequestError:

        await client.aclose()

        return Response(
            content=b"",
            media_type="audio/wav",
            headers={
                "X-Transcription": request.text,
                "X-TTS-Available": "false",
            },
        )