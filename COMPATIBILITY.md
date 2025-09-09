# ComfyUI Kokoro (Remote‑Only) — Compatibility & “Missing Nodes” Guide

This plugin is a **remote‑only fork** of `stavsap/comfyui-kokoro`. It registers the **same node keys** (`KokoroSpeaker`, `KokoroSpeakerCombiner`, `KokoroGenerator`) so existing workflows load **without edits**.

## ✅ Simple Rule

**Install only one Kokoro plugin at a time — either the original _or_ this remote‑only fork. Do not install both.**

If both are present, whichever registers last will take over those node keys, leading to confusion.

## How Saved Workflows Resolve

- A workflow that references `KokoroGenerator`/`KokoroSpeaker`/`KokoroSpeakerCombiner` will load and run with **this fork** as long as **only this fork** is installed.
- If **both** plugins are installed, resolution is undefined (last one loaded wins). Avoid this by keeping only one installed.

## ComfyUI Manager: “Install Missing Nodes”

If a workflow references `KokoroGenerator` and you have **no** Kokoro plugin installed, ComfyUI Manager may suggest the **original** repo when you click **Install Missing Nodes**.

If you prefer this remote‑only fork instead:
1. Open **ComfyUI Manager**.
2. Choose **Install via Git URL**.
3. Paste:
   ```
   https://github.com/bigsk1/comfyui-kokoro-remote.git
   ```
4. Restart ComfyUI.

Your workflow will then resolve to this fork (same node keys).

## Switching from Original → Remote‑Only Fork

1. Remove the original plugin folder from `ComfyUI/custom_nodes/`.
2. Install this repo:
   - Manager → **Install via Git URL** → `https://github.com/bigsk1/comfyui-kokoro-remote.git`, or
   - Manually:
     ```bash
     cd ComfyUI/custom_nodes
     git clone https://github.com/bigsk1/comfyui-kokoro-remote.git
     cd comfyui-kokoro-remote
     pip install -r requirements.txt
     ```
3. Restart ComfyUI.

## Local vs Remote Usage

### Local Kokoro (no auth)
Works out of the box. The node’s default `base_url` is `http://localhost:8880`. The plugin also handles whether `/v1` is included.

### Remote Kokoro (with auth)
Create a `.env` **inside this plugin folder** and restart ComfyUI:

```env
KOKORO_BASE_URL=https://kokoro-tts.yourdomain.com
# Optional auth (choose Basic OR Bearer):
# KOKORO_USERNAME=admin
# KOKORO_PASSWORD=secret
# KOKORO_BEARER=token

# Optional tuning:
KOKORO_TIMEOUT=60
KOKORO_SAMPLE_RATE=44100
```

> **Tip:** Changes to `.env` are loaded at import time. After editing `.env`, restart ComfyUI. If you set `base_url` directly in the node UI, no restart is required.

## Notes on Mixed Voices & Backward Compatibility

- Mixed voices are supported (e.g., `af_sky+af_bella`). The **Kokoro Speaker Combiner** node lets you combine two voices; this fork defaults to supporting weights when provided.
- Old workflows that used the original node names will work with this fork (same registry keys), as long as **only one plugin** is installed.

## TL;DR

- Keep **one** Kokoro plugin installed to avoid collisions.
- This fork uses the **same node names**, so your existing graphs “just work.”
- Manager may suggest the original on “Missing Nodes”; you can instead install this repo by URL.
- For remote servers, set `.env` in this plugin folder; for local servers, the default `http://localhost:8880` works without `.env`.
