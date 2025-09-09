# Remote-only Kokoro (HTTP)
# - No local ONNX, no downloads
# - Reads creds and base URL from .env (KOKORO_BASE_URL, KOKORO_USERNAME/PASSWORD or KOKORO_BEARER)

import os, io, json, wave
import numpy as np
import torch
import requests
from requests.auth import HTTPBasicAuth

# .env loader: first try CWD (ComfyUI root), then plugin-local .env
try:
    from dotenv import load_dotenv
    load_dotenv()  # pick up a root-level .env if present
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))  # plugin-local .env
except Exception:
    pass


# ----- constants kept from original for UI compatibility -----
supported_languages_display = [
    "English", "English (British)", "French", "Japanese", "Hindi",
    "Mandarin Chinese", "Spanish", "Brazilian Portuguese", "Italian"
]
supported_languages = {
    supported_languages_display[0]: "en-us",
    supported_languages_display[1]: "en-gb",
    supported_languages_display[2]: "fr-fr",
    supported_languages_display[3]: "ja",
    supported_languages_display[4]: "hi",
    supported_languages_display[5]: "cmn",
    supported_languages_display[6]: "es",
    supported_languages_display[7]: "pt-br",
    supported_languages_display[8]: "it",
}

supported_voices = [
    # American Female
    "af_heart","af_alloy","af_aoede","af_bella","af_jessica","af_kore","af_nicole","af_nova","af_river","af_sarah","af_sky",
    # American Male
    "am_adam","am_echo","am_eric","am_fenrir","am_liam","am_michael","am_onyx","am_puck","am_santa",
    # British Female
    "bf_alice","bf_emma","bf_isabella","bf_lily",
    # British Male
    "bm_daniel","bm_fable","bm_george","bm_lewis",
    # Japanese Female
    "jf_alpha","jf_gongitsune","jf_nezumi","jf_tebukuro",
    # Japanese Male
    "jm_kumo",
    # Chinese (Z?) + misc
    "zf_xiaobei","zf_xiaoni","zf_xiaoxiao","zf_xiaoyi","zm_yunjian","zm_yunxi","zm_yunxia","zm_yunyang",
    "ef_dora","em_alex","em_santa","ff_siwis","hf_alpha","hf_beta","hm_omega","hm_psi","if_sara","im_nicola","pf_dora","pm_alex","pm_santa",
]

# ----- helpers -----
def _resolve_url(base: str) -> str:
    b = (base or "").rstrip("/")
    if b.endswith("/v1"):
        return b + "/audio/speech"
    return b + "/v1/audio/speech"

def _auth_headers():
    user = os.getenv("KOKORO_USERNAME", "")
    pwd  = os.getenv("KOKORO_PASSWORD", "")
    bearer = os.getenv("KOKORO_BEARER", "")
    headers = {"Accept": "*/*", "Content-Type": "application/json"}
    auth = None
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    elif user or pwd:
        auth = HTTPBasicAuth(user, pwd)
    return headers, auth

def _post_tts(base_url: str, text: str, voice_str: str, speed: float, lang_code: str, timeout_s: int) -> io.BytesIO:
    url = _resolve_url(base_url)
    payload = {
        "model": "kokoro",
        "voice": voice_str,       # "af_sky" or "af_sky+af_bella"
        "input": text,
        "format": "wav",          # WAV for easy decode
        "response_format": "wav",
        # The following are supported by some servers; safe to include if ignored
        "speed": float(speed),
        "lang": lang_code or "en-us",
    }
    headers, auth = _auth_headers()
    r = requests.post(url, data=json.dumps(payload), headers=headers, auth=auth, timeout=timeout_s)
    r.raise_for_status()
    return io.BytesIO(r.content)

def _wav_to_audio(bio: io.BytesIO, target_sr: int):
    with wave.open(bio, "rb") as w:
        ch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        n  = w.getnframes()
        pcm = w.readframes(n)

    # int PCM -> float32 [-1,1]
    if sw == 1:
        arr = np.frombuffer(pcm, dtype=np.int8).astype(np.float32) / 127.0
    elif sw == 2:
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
    elif sw == 3:
        a = np.frombuffer(pcm, dtype=np.uint8).reshape(-1,3)
        b = (a[:,2].astype(np.int32) << 16) | (a[:,1].astype(np.int32) << 8) | a[:,0].astype(np.int32)
        b = np.where(b & 0x800000, b - 0x1000000, b).astype(np.int32)
        arr = b.astype(np.float32) / (2**23)
    elif sw == 4:
        arr = np.frombuffer(pcm, dtype=np.int32).astype(np.float32) / 2147483647.0
    else:
        raise ValueError(f"Unsupported sample width: {sw}")

    if ch > 1:
        arr = arr.reshape(-1, ch).T.mean(axis=0, keepdims=True)  # mono
    else:
        arr = arr.reshape(1, -1)

    cur_sr = sr
    if target_sr and target_sr > 0 and target_sr != cur_sr:
        t_old = np.linspace(0, 1, arr.shape[1], endpoint=False)
        t_new = np.linspace(0, 1, int(arr.shape[1] * (target_sr / cur_sr)), endpoint=False)
        arr = np.interp(t_new, t_old, arr[0]).reshape(1, -1)
        cur_sr = target_sr

    audio_tensor = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)  # [B=1, C=1, T]
    return {"waveform": audio_tensor, "sample_rate": int(cur_sr)}


class KokoroSpeaker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "speaker_name": (supported_voices, {"default": "af_sarah"}),
            },
        }
    RETURN_TYPES = ("KOKORO_SPEAKER",)
    RETURN_NAMES = ("speaker",)
    FUNCTION = "select"
    CATEGORY = "kokoro"

    def select(self, speaker_name):
        # remote-only: return a descriptor the generator can consume
        return ({"voice": str(speaker_name)},)

    @classmethod
    def IS_CHANGED(cls, speaker_name):
        return hash(speaker_name)

class KokoroSpeakerCombiner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "speaker_a": ("KOKORO_SPEAKER", ),
                "speaker_b": ("KOKORO_SPEAKER", ),
                "weight": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.05}),
            },
        }
    RETURN_TYPES = ("KOKORO_SPEAKER",)
    RETURN_NAMES = ("speaker",)
    FUNCTION = "combine"
    CATEGORY = "kokoro"

    def combine(self, speaker_a, speaker_b, weight):
        va = speaker_a.get("voice") if isinstance(speaker_a, dict) else None
        vb = speaker_b.get("voice") if isinstance(speaker_b, dict) else None
        if not va or not vb:
            # fall back: pass through first
            return ({"voice": va or vb or "af_sarah"},)

        # Most Kokoro HTTP servers accept "a+b" (50/50). Weighted mixing is server-dependent.
        # We preserve the weight in the object for servers that support it via custom fields.
        return ({"voice": f"{va}+{vb}", "weights": [float(weight), float(1.0 - weight)]},)

    @classmethod
    def IS_CHANGED(cls, speaker_a, speaker_b, weight):
        return hash((speaker_a.get("voice"), speaker_b.get("voice"), weight))

class KokoroGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "I am a synthesized robot"}),
                "speaker": ("KOKORO_SPEAKER", ),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05}),
                "lang": (supported_languages_display, {"default": "English"}),
            },
            "optional": {
                "base_url": ("STRING", {"default": os.getenv("KOKORO_BASE_URL", "http://localhost:8880")}),
                "timeout_s": ("INT", {"default": int(os.getenv("KOKORO_TIMEOUT", "60"))}),
                "target_sample_rate": ("INT", {"default": int(os.getenv("KOKORO_SAMPLE_RATE", "44100"))}),
                # If your server supports explicit mixing weights, set KOKORO_SUPPORTS_WEIGHTS=1
                "supports_weights": ("BOOLEAN", {"default": os.getenv("KOKORO_SUPPORTS_WEIGHTS","0") == "1"}),
            },
        }
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "kokoro"

    def generate(self, text, speaker, speed, lang,
                 base_url=None, timeout_s=60, target_sample_rate=44100, supports_weights=False):
        voice = "af_sarah"
        weights = None
        if isinstance(speaker, dict):
            voice = speaker.get("voice", voice)
            weights = speaker.get("weights")

        # Convert display -> code just like original
        lang_code = supported_languages.get(lang) or "en-us"

        # Build voice string; optionally include weights if your API supports it
        voice_str = voice
        payload_weights = None
        if supports_weights and isinstance(weights, list) and len(weights) == 2:
            payload_weights = [float(weights[0]), float(weights[1])]

        # Make request
        url_base = base_url or os.getenv("KOKORO_BASE_URL", "")
        bio = _post_tts(url_base, text, voice_str, speed, lang_code, timeout_s)

        # Decode + shape to Comfy AUDIO
        audio = _wav_to_audio(bio, target_sr=target_sample_rate)
        return (audio,)

NODE_CLASS_MAPPINGS = {
    "KokoroGenerator": KokoroGenerator,
    "KokoroSpeaker": KokoroSpeaker,
    "KokoroSpeakerCombiner": KokoroSpeakerCombiner,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KokoroGenerator": "Kokoro Generator",
    "KokoroSpeaker": "Kokoro Speaker",
    "KokoroSpeakerCombiner": "Kokoro Speaker Combiner",
}
