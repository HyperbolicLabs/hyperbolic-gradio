import os
from typing import Any, Callable, Dict, Iterable, List, Optional

import gradio as gr
from openai import OpenAI, OpenAIError

__version__ = "0.0.5"

# Default Hyperbolic inference endpoint. Overridable per interface via the
# `base_url` argument of `registry`, which was previously accepted and ignored.
DEFAULT_BASE_URL = "https://api.hyperbolic.xyz/v1"

# Generation defaults. These were hardcoded inside the request call, which
# silently truncated every answer at 1024 tokens with no way to change it.
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024


def _normalize_history(history: Optional[Iterable[Any]]) -> List[Dict[str, str]]:
    """Convert a Gradio chat history into OpenAI `messages` entries.

    Both history formats are accepted:
      * `type="messages"` (the Gradio 5 default): a list of
        `{"role": ..., "content": ...}` dicts.
      * `type="tuples"` (legacy): a list of `(user, assistant)` pairs.

    Entries whose content is empty or `None` are dropped. Gradio emits a pair
    with `assistant=None` while a response is still streaming, and forwarding
    `content: None` makes the OpenAI-compatible endpoint reject the request.
    """
    if not history:
        return []

    messages: List[Dict[str, str]] = []
    for entry in history:
        if isinstance(entry, dict):
            role = entry.get("role")
            content = entry.get("content")
            # Non-text content (files, images) has no textual representation to
            # replay, so it is skipped rather than stringified.
            if role in ("user", "assistant", "system") and isinstance(content, str) and content:
                messages.append({"role": role, "content": content})
            continue

        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            user_msg, assistant_msg = entry
            if isinstance(user_msg, str) and user_msg:
                messages.append({"role": "user", "content": user_msg})
            if isinstance(assistant_msg, str) and assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})

    return messages


def get_fn(
    model_name: str,
    preprocess: Callable,
    postprocess: Callable,
    api_key: str,
    base_url: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
):
    """Build the streaming chat function passed to `gr.ChatInterface`.

    The `OpenAI` client is created once per interface instead of once per
    message: it owns a connection pool, so rebuilding it on every turn forced a
    fresh TLS handshake and leaked sockets under load.
    """
    client = OpenAI(api_key=api_key, base_url=base_url or DEFAULT_BASE_URL)

    def fn(message, history):
        inputs = preprocess(message, history)
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=inputs["messages"],
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            response_text = ""
            for chunk in completion:
                # Some providers emit chunks with an empty `choices` list (for
                # example the final usage-only chunk). Indexing [0] blindly ended
                # the stream with an IndexError instead of finishing cleanly.
                if not getattr(chunk, "choices", None):
                    continue
                delta = getattr(chunk.choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta else None
                if not content:
                    continue
                response_text += content
                yield postprocess(response_text)
        except OpenAIError as e:
            # Surface a readable message in the UI. Raising the raw exception
            # rendered a full traceback in the browser, which can include request
            # metadata; `gr.Error` shows only this text.
            raise gr.Error(f"Hyperbolic API request failed: {e}") from e

    return fn


def get_interface_args(pipeline):
    if pipeline == "chat":
        inputs = None
        outputs = None

        def preprocess(message, history):
            messages = _normalize_history(history)
            if isinstance(message, dict):
                # Multimodal textbox payload: {"text": ..., "files": [...]}.
                message = message.get("text", "")
            messages.append({"role": "user", "content": message or ""})
            return {"messages": messages}

        def postprocess(x):
            """Identity: the model already returns display-ready text."""
            return x
    else:
        # Add other pipeline types when they will be needed
        raise ValueError(f"Unsupported pipeline type: {pipeline}")
    return inputs, outputs, preprocess, postprocess


def get_pipeline(model_name):
    # Determine the pipeline type based on the model name
    # For simplicity, assuming all models are chat models at the moment
    return "chat"


def registry(
    name: str,
    token: str | None = None,
    base_url: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    **kwargs,
):
    """
    Create a Gradio Interface for a model on Hyperbolic.

    Parameters:
        - name (str): The name of the model.
        - token (str, optional): The Hyperbolic API key. If not provided, will look for HYPERBOLIC_API_KEY env variable.
        - base_url (str, optional): The base URL for the Hyperbolic API. Defaults to
          https://api.hyperbolic.xyz/v1. This argument was previously ignored.
        - temperature (float, optional): Sampling temperature for generation.
        - max_tokens (int, optional): Maximum number of tokens to generate.
        - **kwargs: Forwarded to the underlying Gradio interface.
    """
    api_key = token or os.environ.get("HYPERBOLIC_API_KEY")
    if not api_key:
        raise ValueError("API key is not set. Please provide a token or set HYPERBOLIC_API_KEY environment variable.")

    pipeline = get_pipeline(name)
    inputs, outputs, preprocess, postprocess = get_interface_args(pipeline)
    fn = get_fn(
        name,
        preprocess,
        postprocess,
        api_key,
        base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if pipeline == "chat":
        interface = gr.ChatInterface(fn=fn, **kwargs)
    else:
        interface = gr.Interface(fn=fn, inputs=inputs, outputs=outputs, **kwargs)

    return interface
