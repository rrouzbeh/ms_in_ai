"""Grounded OpenAI chat client for NASA mission questions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI


SYSTEM_PROMPT = """You are NASA Mission Intelligence, a careful archive research assistant
specializing in Apollo 11, Apollo 13, and Space Shuttle Challenger (STS-51L).

Rules:
1. Ground mission-specific factual claims only in the retrieved NASA archive context supplied
   with the current user message. Do not use memory to fill gaps.
2. Cite evidence inline using the exact source labels, for example [Source 1].
3. If the excerpts do not support an answer, say what cannot be determined from the retrieved
   context and suggest a narrower follow-up. Never invent events, quotations, times, or causes.
4. Distinguish explicitly documented facts from reasonable interpretation.
5. Be concise but sufficiently detailed for a mission operations specialist.
6. Conversation history may clarify the user's intent, but it is not evidence. Only the current
   retrieved context is evidence.
"""


def _clean_history(history: List[Dict[str, Any]], max_messages: int = 10) -> List[Dict[str, str]]:
    """Keep a bounded sequence of valid user/assistant turns."""
    cleaned: List[Dict[str, str]] = []
    for message in history[-max_messages:]:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content.strip()})
    return cleaned


def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict[str, Any]],
    model: str = "gpt-4.1-mini",
    *,
    client: Optional[Any] = None,
    max_history_messages: int = 10,
) -> str:
    """Generate a source-cited answer from query, retrieved context, and chat history."""
    if not openai_key and client is None:
        raise ValueError("An OpenAI API key is required")
    if not user_message or not user_message.strip():
        raise ValueError("user_message must not be empty")
    if not isinstance(context, str):
        raise ValueError("context must be a string")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(_clean_history(conversation_history, max_history_messages))
    current_prompt = (
        f"Question:\n{user_message.strip()}\n\n"
        f"Retrieved context:\n{context.strip() or '[No relevant context was retrieved.]'}"
    )
    messages.append({"role": "user", "content": current_prompt})

    sdk_client = client or OpenAI(api_key=openai_key)
    response = sdk_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=800,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("The model returned an empty response")
    return content.strip()
