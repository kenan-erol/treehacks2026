"""
Cosmos Reason 2 - OpenAI-Compatible API Server
Serves the model via a simple REST API that works with VLM WebUIs.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import uuid
import time
import base64
import tempfile
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# --- Config ---
MODEL_NAME = os.environ.get("MODEL_NAME", "nvidia/Cosmos-Reason2-2B")
PORT = int(os.environ.get("PORT", "8000"))
HOST = os.environ.get("HOST", "0.0.0.0")

# --- Load model at startup ---
print(f"Loading {MODEL_NAME}...")
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("Model loaded!")

# --- FastAPI app ---
app = FastAPI(title="Cosmos Reason 2 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- OpenAI-compatible schemas ---
class ChatMessage(BaseModel):
    role: str
    content: str | list

class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    temperature: float = 0.6
    max_tokens: int = 2048
    stream: bool = False

class ChatChoice(BaseModel):
    index: int = 0
    message: dict
    finish_reason: str = "stop"

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "nvidia"

class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]

# --- Endpoints ---
@app.get("/v1/models")
async def list_models():
    return ModelList(data=[ModelInfo(id=MODEL_NAME)])

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    temp_files = []

    try:
        # Build messages for the processor
        messages = []
        video_paths = []

        for msg in request.messages:
            if isinstance(msg.content, str):
                messages.append({"role": msg.role, "content": msg.content})
            elif isinstance(msg.content, list):
                content_parts = []
                for part in msg.content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            content_parts.append({"type": "text", "text": part["text"]})

                        elif part.get("type") == "video_url":
                            video_url = part.get("video_url", {}).get("url", "")

                            if video_url.startswith("data:"):
                                # Base64 encoded video
                                header, b64data = video_url.split(",", 1)
                                ext = ".mp4"
                                if "webm" in header:
                                    ext = ".webm"
                                tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                                tmp.write(base64.b64decode(b64data))
                                tmp.close()
                                temp_files.append(tmp.name)
                                video_paths.append(tmp.name)
                                content_parts.append({
                                    "type": "video",
                                    "video": f"file://{tmp.name}",
                                    "fps": 4
                                })

                            elif video_url.startswith("file://"):
                                # Local file path
                                path = video_url.replace("file://", "")
                                video_paths.append(path)
                                content_parts.append({
                                    "type": "video",
                                    "video": video_url,
                                    "fps": 4
                                })

                            else:
                                # Treat as file path
                                video_paths.append(video_url)
                                content_parts.append({
                                    "type": "video",
                                    "video": f"file://{video_url}",
                                    "fps": 4
                                })

                        elif part.get("type") == "image_url":
                            image_url = part.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:"):
                                header, b64data = image_url.split(",", 1)
                                ext = ".jpg"
                                if "png" in header:
                                    ext = ".png"
                                tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                                tmp.write(base64.b64decode(b64data))
                                tmp.close()
                                temp_files.append(tmp.name)
                                content_parts.append({
                                    "type": "image",
                                    "image": f"file://{tmp.name}"
                                })
                            else:
                                content_parts.append({
                                    "type": "image",
                                    "image": image_url
                                })
                    else:
                        content_parts.append({"type": "text", "text": str(part)})

                messages.append({"role": msg.role, "content": content_parts})

        # Process with model
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if video_paths:
            inputs = processor(text=[text], videos=video_paths, return_tensors="pt").to(model.device)
        else:
            inputs = processor(text=[text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature if request.temperature > 0 else None,
                do_sample=request.temperature > 0,
            )

        response_text = processor.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[ChatChoice(
                message={"role": "assistant", "content": response_text}
            )]
        )

    finally:
        # Clean up temp files
        for f in temp_files:
            try:
                os.unlink(f)
            except OSError:
                pass


# --- Upload endpoint for WebUIs that upload files directly ---
@app.post("/v1/files/upload")
async def upload_video(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(
        suffix=os.path.splitext(file.filename)[1],
        delete=False,
        dir="/tmp"
    )
    content = await file.read()
    tmp.write(content)
    tmp.close()
    return {"file_path": tmp.name, "filename": file.filename}


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "gpu": torch.cuda.is_available()}


if __name__ == "__main__":
    print(f"Starting server on {HOST}:{PORT}")
    print(f"API endpoint: http://{HOST}:{PORT}/v1/chat/completions")
    print(f"Models list:  http://{HOST}:{PORT}/v1/models")
    uvicorn.run(app, host=HOST, port=PORT)