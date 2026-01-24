# mypy: disable-error-code="misc,untyped-decorator,union-attr,no-any-return,arg-type"
"""LLM chat proxy using LiteLLM for universal provider support.

This module provides a streaming chat endpoint that proxies requests to any LLM
provider (OpenAI, Anthropic, Ollama, Groq, etc.) via LiteLLM. This allows the
Artifacta UI to integrate LLM chat without requiring separate API clients for
each provider.

Why LiteLLM:
    - Unified interface for 100+ LLM providers
    - Auto-detects provider from model name
    - Handles provider-specific authentication
    - Normalizes response format across providers
    - Supports streaming for all providers
    - Open source and actively maintained

Supported Providers (examples):
    - OpenAI: gpt-4o, gpt-4o-mini, gpt-3.5-turbo
    - Anthropic: claude-3-5-sonnet-20241022, claude-3-opus
    - Ollama: ollama/llama2, ollama/mistral
    - Groq: groq/mixtral-8x7b-32768
    - Many more: Cohere, Replicate, HuggingFace, etc.

Model Name Format:
    LiteLLM uses model name to determine provider:
    - "gpt-4o" -> OpenAI
    - "claude-3-5-sonnet" -> Anthropic
    - "ollama/llama2" -> Ollama (local)
    - "groq/mixtral" -> Groq

    Prefix (ollama/, groq/) explicitly specifies provider.
    No prefix means OpenAI or Anthropic based on model name pattern.

API Key Handling:
    Client sends optional API key in request:
    1. If provided, set appropriate environment variable:
       - OPENAI_API_KEY for OpenAI models
       - ANTHROPIC_API_KEY for Anthropic models
       - GROQ_API_KEY for Groq models
    2. LiteLLM reads from environment variables
    3. Local models (Ollama) don't require API keys

    Security consideration:
        - API keys stored temporarily in environment (process lifetime)
        - Not persisted to disk
        - Each request can use different key (multi-user support)

Streaming Response Format:
    Server-Sent Events (SSE) protocol:
    - Content-Type: text/event-stream
    - Data format: "data: {json}\\n\\n"
    - Chunks contain delta (incremental content)
    - Client reconstructs full response from deltas

    Example stream:
        data: {"choices": [{"delta": {"content": "Hello"}}]}

        data: {"choices": [{"delta": {"content": " world"}}]}

        data: {"choices": [{"delta": {"content": "!"}}]}


    Why streaming:
        - Progressive rendering in UI (better UX)
        - Lower perceived latency (first token arrives faster)
        - Works for long responses (minutes of generation)
        - Standard for modern LLM interfaces

System Message Handling:
    Optional system_message field prepended to messages:
    - If provided, inserted as first message with role="system"
    - Helps set context/instructions for LLM
    - Example: "You are a helpful data science assistant"

Error Handling:
    - LLM API errors (rate limits, auth failures, timeouts)
    - Network errors (connection failures)
    - Invalid model names
    - All errors returned as SSE data events with "error" field
    - Logged to server logs for debugging

Async Generator Pattern:
    generate() is an async generator:
    - Yields chunks as they arrive from LLM API
    - Allows FastAPI to stream response back to client
    - Efficient (no buffering, low memory usage)
    - Handles backpressure automatically

Integration with UI:
    Frontend uses EventSource or fetch with ReadableStream:
    1. POST to /api/chat/stream with messages array
    2. Read SSE stream incrementally
    3. Parse JSON from each "data:" line
    4. Extract delta.content and append to display
    5. Continue until stream closes
"""

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


# Request models
class ChatRequest(BaseModel):
    """Chat request model."""

    model: str  # e.g. "gpt-4o-mini", "claude-3-5-sonnet-20241022", "ollama/llama2"
    api_key: Optional[str] = None  # Optional for local models
    messages: List[Dict[str, str]]
    system_message: Optional[str] = None


@router.post("/stream")
async def chat_stream(request: ChatRequest) -> Any:
    """Proxy streaming chat requests to any LLM provider via LiteLLM.

    Supports 100+ providers: OpenAI, Anthropic, Ollama, Groq, etc.
    Provider is auto-detected from model name format.
    """
    import litellm

    # Set API key in environment if provided (LiteLLM reads from env)
    if request.api_key:
        import os

        # LiteLLM auto-detects which env var to use based on model name
        # e.g. gpt-4 → OPENAI_API_KEY, claude → ANTHROPIC_API_KEY
        model_lower = request.model.lower()
        if "gpt" in model_lower or "openai" in model_lower:
            os.environ["OPENAI_API_KEY"] = request.api_key
        elif "claude" in model_lower or "anthropic" in model_lower:
            os.environ["ANTHROPIC_API_KEY"] = request.api_key
        elif "groq" in model_lower:
            os.environ["GROQ_API_KEY"] = request.api_key
        # Ollama and other local models don't need API keys

    async def generate() -> AsyncGenerator[str, None]:
        try:
            # Build messages with system message if provided
            messages = request.messages.copy()
            if request.system_message:
                messages = [{"role": "system", "content": request.system_message}] + messages

            # LiteLLM handles all provider-specific formatting
            response = await litellm.acompletion(
                model=request.model, messages=messages, stream=True, timeout=60.0
            )

            # Stream responses in unified format
            async for chunk in response:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        # Return in OpenAI SSE format (standard for streaming)
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': delta.content}}]})}\n\n"

        except Exception as e:
            logger.error(f"LLM streaming error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
