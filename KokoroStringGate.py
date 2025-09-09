# KokoroStringGate.py
# A tiny gate for STRINGs (e.g., an audio file path). When enabled=False, passes fallback/blank.

class StringGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": ""}),
                "enabled": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # Optional: if disabled, output this value instead of blank
                "fallback": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "go"
    CATEGORY = "kokoro" 

    def go(self, text, enabled=False, fallback=""):
        return (text if enabled else fallback,)

NODE_CLASS_MAPPINGS = {"StringGate": StringGate}
NODE_DISPLAY_NAME_MAPPINGS = {"StringGate": "String Gate"}
