import os, json, requests
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

base = os.getenv("KOKORO_BASE_URL", "http://localhost:8880").rstrip("/")
url = base + ("/audio/speech" if base.endswith("/v1") else "/v1/audio/speech")

payload = {
    "model": "kokoro",
    "voice": "af_sky",
    "input": "This is a remote Kokoro test from ComfyUI.",
    "response_format": "wav",
    "format": "wav",
}

headers = {"Accept": "*/*", "Content-Type": "application/json"}
auth = None
bearer = os.getenv("KOKORO_BEARER","")
if bearer:
    headers["Authorization"] = f"Bearer {bearer}"
else:
    from requests.auth import HTTPBasicAuth
    user = os.getenv("KOKORO_USERNAME",""); pwd = os.getenv("KOKORO_PASSWORD","")
    if user or pwd:
        auth = HTTPBasicAuth(user, pwd)

r = requests.post(url, data=json.dumps(payload), headers=headers, auth=auth, timeout=int(os.getenv("KOKORO_TIMEOUT","60")))
r.raise_for_status()
open("tts_test.wav","wb").write(r.content)
print("Wrote tts_test.wav")
