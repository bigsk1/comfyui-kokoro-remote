# ComfyUI Kokoro (Remote-Only)

Remote-only Kokoro TTS nodes for ComfyUI that call a hosted Kokoro server over HTTP.
No ONNX downloads. Nodes/UX: **Kokoro Speaker**, **Kokoro Speaker Combiner**, **Kokoro Generator**.

## Features
- Text → AUDIO tensor (44.1 kHz default) for **PreviewAudio** / **SaveAudio (.flac + metadata)** / lip-sync / Infinite Talk chains
- Supports Basic Auth (Traefik) or Bearer
- Voice mixing: `speaker_a + speaker_b`
- Languages and speed forwarded to the API (ignored safely if unsupported)

<img width="950" height="440" alt="kokoro-remote" src="https://github.com/user-attachments/assets/162da51a-5775-4a24-ab94-54082b6b27d6" />


## Install (Manual)
1. Clone into `ComfyUI/custom_nodes/comfyui-kokoro-remote`
2. `pip install -r requirements.txt`
3. Create `.env` in the repo:


## ComfyUI Manager

Open ComfyUI Manager in your ComfyUI UI.

Click Install via Git URL.

Paste the repo url:

https://github.com/bigsk1/comfyui-kokoro-remote.git


Install → Restart ComfyUI.

### Kokoro Remote Authentication

.env in root of comfyui-kokoro-remote

 ```env
# Your Kokoro base (with or without /v1 suffix)
KOKORO_BASE_URL=https://tts.example.com or http://localhost:8880
# Basic auth (Traefik) OR bearer
KOKORO_USERNAME=admin
KOKORO_PASSWORD=secret
# KOKORO_BEARER=your_token

# Optional
KOKORO_TIMEOUT=60
KOKORO_SAMPLE_RATE=44100
```

If you have a locally running Kokoro server with no auth just enter the url in the node.


<img width="950" height="480" alt="kokoro-remote2" src="https://github.com/user-attachments/assets/f7739415-6bf7-4033-b7bd-229963516cb6" />


## Compatibility & “Missing Nodes” Guide

[Compatibility & “Missing Nodes” Guide](COMPATIBILITY.md)
    
## License

- [This repo](LICENSE)
- kokoro-onnx: MIT
- kokoro model: Apache 2.0

## Credits

- [stavsap Forked Repo](https://github.com/stavsap/comfyui-kokoro)
- [Kokoro TTS Engine](https://huggingface.co/hexgrad/Kokoro-82M)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI-BS_Kokoro-onnx](https://github.com/Burgstall-labs/ComfyUI-BS_Kokoro-onnx)
- [ComfyUI-KokoroTTS](https://github.com/benjiyaya/ComfyUI-KokoroTTS)
