# ComfyUI Kokoro (Remote-Only)

Remote-only Kokoro TTS nodes for ComfyUI that call a hosted Kokoro server over HTTP.
No ONNX downloads. Same nodes/UX as the original: **Kokoro Speaker**, **Kokoro Speaker Combiner**, **Kokoro Generator**.

## Features
- Text → AUDIO tensor (44.1 kHz default) for **PreviewAudio** / **SaveAudio (.flac + metadata)** / lip-sync / Infinite Talk chains
- Supports Basic Auth (Traefik) or Bearer
- Voice mixing: `speaker_a + speaker_b` (optionally pass weights if your server supports it)
- Languages and speed forwarded to the API (ignored safely if unsupported)

<img width="1425" height="604" alt="image" src="https://github.com/user-attachments/assets/162da51a-5775-4a24-ab94-54082b6b27d6" />


## Install
1. Clone into `ComfyUI/custom_nodes/comfyui-kokoro-remote`
2. `pip install -r requirements.txt`
3. Create `.env` in the repo:
   ```env
   KOKORO_BASE_URL=https://tts.example.com
   KOKORO_USERNAME=admin
   KOKORO_PASSWORD=secret
   KOKORO_TIMEOUT=60
   KOKORO_SAMPLE_RATE=44100


## License

- [This repo](LICENSE)
- kokoro-onnx: MIT
- kokoro model: Apache 2.0

## Credits

- [Kokoro TTS Engine](https://huggingface.co/hexgrad/Kokoro-82M)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI-BS_Kokoro-onnx](https://github.com/Burgstall-labs/ComfyUI-BS_Kokoro-onnx)
- [ComfyUI-KokoroTTS](https://github.com/benjiyaya/ComfyUI-KokoroTTS)
