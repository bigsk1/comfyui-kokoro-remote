import os, time, wave
import numpy as np
import torch
import folder_paths  # provided by ComfyUI

class SaveAudioPathWAV:
    DESCRIPTION = (
        "Write AUDIO to .wav and return its path.\n"
        "To save ONLY the audio without running the rest of your workflow: "
        "right-click this node → 'Queue selected output nodes'."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "kokoro"}),
            },
            "optional": {"subfolder": ("STRING", {"default": ""})}
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "audio_path")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "kokoro"


    def save(self, audio, filename_prefix, subfolder=""):
        outdir = folder_paths.get_output_directory()
        if subfolder:
            outdir = os.path.join(outdir, subfolder)
        os.makedirs(outdir, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        fpath = os.path.join(outdir, f"{filename_prefix}_{ts}.wav")

        wav = audio["waveform"]; sr = int(audio["sample_rate"])
        if isinstance(wav, torch.Tensor): arr = wav[0, 0].detach().cpu().numpy()
        else: arr = np.asarray(wav, dtype=np.float32)

        arr = np.clip(arr, -1.0, 1.0)
        pcm16 = (arr * 32767.0).astype(np.int16)

        with wave.open(fpath, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm16.tobytes())
        return (audio, fpath)

NODE_CLASS_MAPPINGS = {"SaveAudioPathWAV": SaveAudioPathWAV}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveAudioPathWAV": "SaveAudioPath (WAV)"}
