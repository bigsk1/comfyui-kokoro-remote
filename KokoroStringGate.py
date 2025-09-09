class StringGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"default": ""}),
                             "enabled": ("BOOLEAN", {"default": False})}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "go"
    CATEGORY = "kokoro"
    def go(self, text, enabled=False):
        return (text if enabled else "",)

NODE_CLASS_MAPPINGS = {"StringGate": StringGate}
NODE_DISPLAY_NAME_MAPPINGS = {"StringGate": "String Gate"}
