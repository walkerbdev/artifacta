# mypy: disable-error-code="misc,untyped-decorator,union-attr,no-any-return,arg-type"
"""LLM Chat proxy endpoints using LiteLLM."""

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
