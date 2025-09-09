# KokoroStringGate.py
import os, wave, numpy as np
import folder_paths

def _ensure_silence(path: str, sr: int = 44100, seconds: float = 0.2):
    if os.path.exists(path):
        return path
    n = max(1, int(sr * seconds))
    pcm16 = (np.zeros(n, dtype=np.float32) * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm16.tobytes())
    return path

class StringGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": ""}),         # incoming file path
                "enabled": ("BOOLEAN", {"default": False}),   # switch
            },
            "optional": {
                "fallback": ("STRING", {"default": ""}),      # optional: use this path when disabled
                "sr": ("INT", {"default": 44100}),            # silence sample rate
                "silence_seconds": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 5.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "go"
    CATEGORY = "kokoro"  # keep with your group

    def go(self, text, enabled=False, fallback="", sr=44100, silence_seconds=0.2):
        if enabled and isinstance(text, str) and text.strip():
            return (text,)
        if isinstance(fallback, str) and fallback.strip():
            return (fallback,)
        # create/use a tiny silence file so VHS never errors
        outdir = folder_paths.get_output_directory()
        silence_path = os.path.join(outdir, "_kokoro_silence.wav")
        return (_ensure_silence(silence_path, sr=sr, seconds=silence_seconds),)

NODE_CLASS_MAPPINGS = {"StringGate": StringGate}
NODE_DISPLAY_NAME_MAPPINGS = {"StringGate": "String Gate"}
