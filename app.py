"""Modal Studio: the free cloud GPU video engine behind Creative Studio in Syok AI.

Not a standalone app: it has no screen of its own. Syok's Creative Studio sends
it video jobs through one password-protected address. Each "room" is a GPU model
that only wakes up when a job arrives, so unused rooms cost nothing.

Rooms today:
  wan  - text to video, or picture to video (Wan 2.2 TI2V 5B, Apache-2.0)
  ltx  - LTX-2 (19B distilled): text, start picture, end picture, video to video, with sound

Deploy:        modal deploy app.py
First time:    modal run app.py::fetch_ltx    (downloads LTX-2 on a cheap CPU, not the GPU)
"""
import base64
import hmac
import io
import os

import modal

APP_NAME = "modal-studio"
WAN_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_GPU = ["L40S", "A100-40GB"]   # first choice, then backup if the first is busy
LTX_MODEL = "rootonchair/LTX-2-19b-distilled"   # public, not gated; the text encoder is inside the repo
LTX_GPU = ["A100-80GB", "H100"]    # the 19B model is big: needs an 80 GB card (sizing still to be measured)
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

ltx_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.8.0", "transformers", "diffusers==0.43.0", "accelerate", "sentencepiece",
        "ftfy", "av", "imageio", "imageio-ffmpeg", "pillow", "numpy", "huggingface_hub",
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


# LTX-2 sizes: multiples of 32; frame counts must be 8k+1
LTX_SIZES = {
    "fast": {"16:9": (768, 448), "9:16": (448, 768), "1:1": (640, 640)},
    "hd": {"16:9": (1024, 576), "9:16": (576, 1024), "1:1": (832, 832)},
}


@app.function(
    image=ltx_image,
    volumes={CACHE: cache},
    timeout=60 * 60,
    cpu=2,
)
def fetch_ltx():
    """Run once: download LTX-2 into the cache volume on a cheap CPU box, so the
    expensive GPU is not billed while tens of gigabytes download."""
    from huggingface_hub import snapshot_download

    snapshot_download(LTX_MODEL, max_workers=8)
    cache.commit()
    return "LTX-2 downloaded"


@app.cls(
    image=ltx_image,
    gpu=LTX_GPU,
    volumes={CACHE: cache},
    scaledown_window=SCALEDOWN_SECONDS,
    timeout=40 * 60,
    max_containers=1,  # one GPU at a time (the two rooms never run together either)
)
class Ltx:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import LTX2ConditionPipeline

        self.pipe = LTX2ConditionPipeline.from_pretrained(LTX_MODEL, torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload(device="cuda")
        self.pipe.vae.enable_tiling()
        cache.commit()

    @modal.method()
    def make(self, prompt: str, image_b64: str | None, last_b64: str | None,
             video_b64: str | None, seconds: int, ratio: str, quality: str) -> bytes:
        import torch
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
        from diffusers.utils import encode_video, load_video
        from PIL import Image, ImageOps

        width, height = LTX_SIZES[quality][ratio]
        frames = 8 * round(FPS * seconds / 8) + 1

        def pic(b64):
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            return ImageOps.fit(img, (width, height), Image.LANCZOS)

        conditions = []
        if video_b64:  # video to video: the source clip guides the new one
            with open("/tmp/in.mp4", "wb") as fh:
                fh.write(base64.b64decode(video_b64))
            conditions.append(LTX2VideoCondition(frames=load_video("/tmp/in.mp4")[:frames], index=0, strength=1.0))
        elif image_b64:  # start picture
            conditions.append(LTX2VideoCondition(frames=pic(image_b64), index=0, strength=1.0))
        if last_b64:  # end picture
            conditions.append(LTX2VideoCondition(frames=pic(last_b64), index=-1, strength=1.0))

        video, audio = self.pipe(
            conditions=conditions or None,  # None = plain text to video (supported by the pipeline)
            prompt=prompt,
            width=width,
            height=height,
            num_frames=frames,
            frame_rate=float(FPS),
            sigmas=DISTILLED_SIGMA_VALUES,
            guidance_scale=1.0,
            generator=torch.Generator("cuda").manual_seed(int.from_bytes(os.urandom(4), "big")),
            output_type="np",
            return_dict=False,
        )
        path = "/tmp/ltx.mp4"
        encode_video(
            video[0], fps=FPS, audio=audio[0].float().cpu(),
            audio_sample_rate=self.pipe.vocoder.config.output_sampling_rate, output_path=path,
        )
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
        room: str = "wan"          # "wan" (fast, picture optional) or "ltx" (start/end frame, video in, sound)
        image: str | None = None  # a picture as a data: URL or plain base64 (optional); LTX: the start picture
        last_image: str | None = None  # LTX only: the end picture
        video: str | None = None  # LTX only: a source clip (data: URL) for video to video
        seconds: int = 5
        ratio: str = "16:9"
        quality: str = "fast"

    def raw(x, limit):
        if not x:
            return None
        x = x.split(",", 1)[1] if x.startswith("data:") else x
        if len(x) > limit:
            raise HTTPException(400, "file too large")
        return x

    @api.get("/v1/status")
    def status(authorization: str | None = Header(default=None)):
        check(authorization)
        return {"ok": True, "rooms": ["wan", "ltx"], "max_seconds": MAX_SECONDS}

    @api.post("/v1/video")
    def start(job: VideoJob, authorization: str | None = Header(default=None)):
        check(authorization)
        prompt = job.prompt.strip()
        if not 3 <= len(prompt) <= 2000:
            raise HTTPException(400, "prompt must be 3 to 2000 characters")
        if job.ratio not in SIZES["fast"] or job.quality not in SIZES:
            raise HTTPException(400, "ratio must be 16:9, 9:16 or 1:1; quality fast or hd")
        if job.room not in ("wan", "ltx"):
            raise HTTPException(400, "room must be wan or ltx")
        seconds = max(1, min(MAX_SECONDS, job.seconds))
        image = raw(job.image, 14_000_000)  # about 10 MB of picture

        if job.room == "ltx":
            call = Ltx().make.spawn(
                prompt, image, raw(job.last_image, 14_000_000), raw(job.video, 52_000_000),
                seconds, job.ratio, job.quality,
            )
            return {"id": call.object_id, "status": "running", "room": "ltx"}
        if job.last_image or job.video:
            raise HTTPException(400, "end picture and video-in need room ltx")
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
