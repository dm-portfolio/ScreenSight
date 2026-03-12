# ScreenSight AI (Codespaces, no `mss`)

This project provides a **screen-seeing AI pipeline for GitHub Codespaces** without using `mss`.

Instead of desktop capture libraries, it uses the browser's native `getDisplayMedia()` API:

1. You open a local web page in your Codespace.
2. You choose exactly which screen/window/tab to share.
3. The page streams frames to a Python backend.
4. The backend runs AI-style analysis on each frame.

This works well in Codespaces because the browser is already your UI surface.

## Why this avoids `mss`

- Capture is performed in-browser with `navigator.mediaDevices.getDisplayMedia()`.
- Backend only receives JPEG frame bytes over a WebSocket.
- No `mss` package or OS framebuffer scraping is used.

## Features

- User-selectable screen/window/tab capture.
- Adjustable FPS and JPEG quality.
- Real-time frame analysis endpoint over WebSocket.
- Built-in analysis:
  - brightness estimate
  - edge density (scene complexity proxy)
  - motion score versus previous frame
  - optional OCR (`pytesseract`) if installed

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open forwarded port `8000` in your browser and click **Start sharing**.

## API behavior

- Browser sends JSON messages to `ws://<host>/ws`.
- Message types:
  - `frame`: base64 JPEG + metadata
  - `control`: start/stop and settings updates
- Server responds with JSON containing analysis for each frame.

## Notes for Codespaces

- `getDisplayMedia` requires HTTPS or localhost context. Forwarded Codespaces URLs satisfy this.
- Screen chooser is controlled by the browser for security; apps cannot bypass it.
- For better performance, keep FPS between 1–8 unless you have strong CPU headroom.

## Optional OpenAI vision integration

You can route sampled frames to a multimodal model for higher-level reasoning.
A helper stub is included in `vision_llm.py` for sending a frame to an OpenAI-compatible endpoint.

