"""Modal Studio: the free cloud GPU video engine behind Creative Studio in Syok AI.

Not a standalone app: it has no screen of its own. Syok's Creative Studio sends
it video jobs through one password-protected address. Each "room" is a GPU model
that only wakes up when a job arrives, so unused rooms cost nothing.

Rooms today:
  wan  - text to video, or picture to video (Wan 2.2 TI2V 5B, Apache-2.0)

Deploy:  modal deploy app.py
"""
import base64
import hmac
import io
import os

import modal

APP_NAME = "modal-studio"
WAN_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_GPU = ["L40S", "A100-40GB"]   # first choice, then backup if the first is busy
SCALEDOWN_SECONDS = 180            # GPU sleeps after 3 idle minutes to save your free credit
MAX_SECONDS = 5                    # longest clip a request may ask for
CACHE = "/cache"                   # Modal Volume that keeps the model files between runs

app = modal.App(APP_NAME)
cache = modal.Volume.from_name("modal-studio-cache", create_if_missing=True)

wan_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.8.0",
        "transformers",
        "diffusers==0.43.0",
        "accelerate",
        "sentencepiece",
        "ftfy",
        "imageio",
        "imageio-ffmpeg",
        "pillow",
        "numpy",
        "huggingface_hub",
    )
    .env({"HF_HOME": CACHE})
)

gateway_image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi[standard]")

# width x height, all multiples of 32 (Wan 5B needs that)
SIZES = {
    "fast": {"16:9": (832, 480), "9:16": (480, 832), "1:1": (640, 640)},
    "hd": {"16:9": (1280, 704), "9:16": (704, 1280), "1:1": (960, 960)},
}
STEPS = {"fast": 30, "hd": 40}
FPS = 24
NEGATIVE = (
    "bright tones, overexposed, static, blurred details, subtitles, worst quality, "
    "low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, "
    "poorly drawn face, fused fingers, still picture, messy background, walking backwards"
)


@app.cls(
    image=wan_image,
    gpu=WAN_GPU,
    volumes={CACHE: cache},
    scaledown_window=SCALEDOWN_SECONDS,
    timeout=30 * 60,
    max_containers=1,  # never run two GPUs at once: keeps your bill predictable
)
class Wan:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline

        vae = AutoencoderKLWan.from_pretrained(WAN_MODEL, subfolder="vae", torch_dtype=torch.float32)
        self.t2v = WanPipeline.from_pretrained(WAN_MODEL, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
        # The picture pipeline re-uses the big parts already in memory.
        self.i2v = WanImageToVideoPipeline.from_pretrained(
            WAN_MODEL,
            vae=vae,
            transformer=self.t2v.transformer,
            text_encoder=self.t2v.text_encoder,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        cache.commit()

    @modal.method()
    def make(self, prompt: str, image_b64: str | None, seconds: int, ratio: str, quality: str) -> bytes:
        import imageio
        import numpy as np
        import torch
        from PIL import Image, ImageOps

        width, height = SIZES[quality][ratio]
        frames = 4 * round(FPS * seconds / 4) + 1  # Wan needs 4k+1 frames
        gen = torch.Generator("cuda").manual_seed(int.from_bytes(os.urandom(4), "big"))
        common = dict(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            height=height,
            width=width,
            num_frames=frames,
            guidance_scale=5.0,
            num_inference_steps=STEPS[quality],
            generator=gen,
        )
        if image_b64:
            img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
            img = ImageOps.fit(img, (width, height), Image.LANCZOS)  # crop to the chosen shape
            out = self.i2v(image=img, **common).frames[0]
        else:
            out = self.t2v(**common).frames[0]

        path = "/tmp/out.mp4"
        arr = [np.asarray((f * 255).clip(0, 255).astype("uint8")) if f.dtype != "uint8" else f for f in out]
        with imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=1) as w:
            for f in arr:
                w.append_data(f)
        with open(path, "rb") as fh:
            return fh.read()


@app.function(
    image=gateway_image,
    secrets=[modal.Secret.from_name("modal-studio-key")],  # holds MODAL_STUDIO_KEY, your password
    scaledown_window=120,
)
@modal.asgi_app()
def gateway():
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel

    api = FastAPI(title="Modal Studio")
    secret = os.environ["MODAL_STUDIO_KEY"]

    def check(authorization: str | None):
        got = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(got.encode(), secret.encode()):
            raise HTTPException(401, "wrong password")

    class VideoJob(BaseModel):
        prompt: str
        image: str | None = None  # a picture as a data: URL or plain base64 (optional)
        seconds: int = 5
        ratio: str = "16:9"
        quality: str = "fast"

    @api.get("/v1/status")
    def status(authorization: str | None = Header(default=None)):
        check(authorization)
        return {"ok": True, "rooms": ["wan"], "max_seconds": MAX_SECONDS}

    @api.post("/v1/video")
    def start(job: VideoJob, authorization: str | None = Header(default=None)):
        check(authorization)
        prompt = job.prompt.strip()
        if not 3 <= len(prompt) <= 2000:
            raise HTTPException(400, "prompt must be 3 to 2000 characters")
        if job.ratio not in SIZES["fast"] or job.quality not in SIZES:
            raise HTTPException(400, "ratio must be 16:9, 9:16 or 1:1; quality fast or hd")
        seconds = max(1, min(MAX_SECONDS, job.seconds))
        image = job.image
        if image:
            image = image.split(",", 1)[1] if image.startswith("data:") else image
            if len(image) > 14_000_000:
                raise HTTPException(400, "picture too large (max about 10 MB)")
        call = Wan().make.spawn(prompt, image, seconds, job.ratio, job.quality)
        return {"id": call.object_id, "status": "running"}

    @api.get("/v1/video/{job_id}")
    def poll(job_id: str, authorization: str | None = Header(default=None)):
        check(authorization)
        call = modal.FunctionCall.from_id(job_id)
        try:
            data = call.get(timeout=0)
        except TimeoutError:
            return {"id": job_id, "status": "running"}
        except Exception as exc:  # the GPU job itself failed
            return {"id": job_id, "status": "failed", "error": str(exc)[:300]}
        return {
            "id": job_id,
            "status": "done",
            "video": "data:video/mp4;base64," + base64.b64encode(data).decode(),
        }

    return api
