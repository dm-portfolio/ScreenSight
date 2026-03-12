from __future__ import annotations

import base64
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from ai_vision import analyze_frame, build_assistant_text, decode_jpeg, to_payload
from ai_vision import analyze_frame, decode_jpeg, to_payload

app = FastAPI(title="ScreenSight AI")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(Path("static") / "index.html")


@app.websocket("/ws")
async def ws_analyze(websocket: WebSocket) -> None:
    await websocket.accept()
    prev_frame = None
    latest_stats = None

    try:
        while True:
            message = await websocket.receive_json()
            msg_type = message.get("type")

            if msg_type == "user_text":
                text = str(message.get("text", "")).strip()
                if latest_stats is None:
                    assistant_text = "Share your screen first so I can describe what I see."
                else:
                    assistant_text = build_assistant_text(latest_stats, user_text=text)
                await websocket.send_json({"type": "assistant_text", "text": assistant_text, "ts": time.time()})
                continue

            if msg_type != "frame":
                await websocket.send_json({"type": "ack", "received": msg_type})
            if message.get("type") != "frame":
                await websocket.send_json({"type": "ack", "received": message.get("type")})
                continue

            b64_data = message.get("jpeg_base64")
            if not isinstance(b64_data, str):
                await websocket.send_json({"type": "error", "error": "jpeg_base64 missing"})
                continue

            jpeg = base64.b64decode(b64_data)
            frame = decode_jpeg(jpeg)
            stats = analyze_frame(
                frame,
                prev_frame_bgr=prev_frame,
                enable_ocr=bool(message.get("enable_ocr", False)),
            )
            latest_stats = stats
            prev_frame = frame

            await websocket.send_json(
                {
                    "type": "analysis",
                    "ts": time.time(),
                    "frame": {
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                    },
                    "analysis": to_payload(stats),
                    "assistant_text": build_assistant_text(stats),
                }
            )
    except WebSocketDisconnect:
        return
