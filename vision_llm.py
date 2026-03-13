from __future__ import annotations

import base64
import json
import os
import urllib.request


def ask_vision_model(image_jpeg: bytes, prompt: str) -> dict:
    """
    Send one frame to an OpenAI-compatible vision endpoint.
    Set OPENAI_API_KEY in your environment before calling.
    """
    api_key = os.environ["OPENAI_API_KEY"]
    b64 = base64.b64encode(image_jpeg).decode("utf-8")

    payload = {
        "model": "gpt-4.1-mini",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
                ],
            }
        ],
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))
