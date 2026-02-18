"""
Tryplicity Inference Server
OpenAI-compatible API for the trained model.
Run after training to serve the model.

Usage: python inference/serve.py --checkpoint ./checkpoints/final_h100_pretrain.pt --port 8080
"""

import sys
import os
import time
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from model.architecture import create_full_model, create_tiny_model
from tokenizers import Tokenizer


# ── Request/Response models ──

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = 0.8
    max_tokens: int = 512
    top_k: int = 50
    top_p: float = 0.9
    stream: bool = False

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str = "tryplicity-3b"
    choices: list[ChatChoice]

class SimpleRequest(BaseModel):
    message: str

class SimpleResponse(BaseModel):
    id: str
    text: str
    sources: list[str]
    timestamp: str


# ── Globals ──
app = FastAPI(title="Tryplicity Inference")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = None
tokenizer = None
device = None
model_name = "tryplicity-3b"


def load_model(checkpoint_path: str, model_size: str = "full"):
    """Load the trained model from checkpoint."""
    global model, tokenizer, device

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create model
    if model_size == "tiny":
        model = create_tiny_model()
    else:
        model = create_full_model()

    # Load checkpoint
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        # Handle DDP wrapped checkpoints
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.replace("module.", "")] = v
        model.load_state_dict(cleaned, strict=False)
        print(f"  Loaded (step {ckpt.get('global_step', '?')}, "
              f"tokens {ckpt.get('total_tokens', '?'):,}, "
              f"loss {ckpt.get('best_loss', '?')})")
    else:
        print(f"WARNING: No checkpoint at {checkpoint_path}, using random weights!")

    model = model.to(device)
    model.eval()
    params = model.count_parameters()
    print(f"  Model params: {params['total']:,}")

    if device == "cuda":
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory: {mem:.2f} GB")


def generate_text(prompt: str, max_tokens: int = 512, temperature: float = 0.8,
                  top_k: int = 50, top_p: float = 0.9) -> str:
    """Generate text from a prompt."""
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], device=device)

    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    output = tokenizer.decode(generated[0].tolist())
    # Remove the prompt from the output
    if output.startswith(prompt):
        output = output[len(prompt):]
    return output.strip()


def format_chat_prompt(messages: list[ChatMessage]) -> str:
    """Format chat messages into a prompt string."""
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"System: {msg.content}\n")
        elif msg.role == "user":
            parts.append(f"User: {msg.content}\n")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {msg.content}\n")
    parts.append("Assistant:")
    return "\n".join(parts)


# ── API Endpoints ──

@app.get("/health")
async def health():
    return {"status": "ok", "model": model_name, "device": str(device)}


@app.post("/api/chat")
async def simple_chat(req: SimpleRequest):
    """Simple chat endpoint matching the Tryplicity frontend format."""
    prompt = f"User: {req.message}\nAssistant:"
    text = generate_text(prompt, max_tokens=256, temperature=0.8)

    return SimpleResponse(
        id=f"tri-{int(time.time()*1000)}",
        text=text,
        sources=[],
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


@app.post("/v1/chat/completions")
async def openai_chat(req: ChatRequest):
    """OpenAI-compatible chat completions endpoint."""
    prompt = format_chat_prompt(req.messages)
    text = generate_text(
        prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
    )

    return ChatResponse(
        id=f"chatcmpl-{int(time.time()*1000)}",
        created=int(time.time()),
        model=model_name,
        choices=[ChatChoice(
            index=0,
            message=ChatMessage(role="assistant", content=text),
            finish_reason="stop",
        )],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tryplicity Inference Server")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/final_h100_pretrain.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer/tryplicity.json",
                        help="Path to tokenizer")
    parser.add_argument("--model-size", type=str, default="full", choices=["tiny", "full"],
                        help="Model size to load")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port to serve on")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(f"Tokenizer loaded: {tokenizer.get_vocab_size()} tokens")

    # Load model
    load_model(args.checkpoint, args.model_size)

    print(f"\nStarting server on {args.host}:{args.port}")
    print(f"  Simple API:  POST /api/chat")
    print(f"  OpenAI API:  POST /v1/chat/completions")
    print(f"  Health:      GET  /health")

    uvicorn.run(app, host=args.host, port=args.port)
