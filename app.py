"""Modal Studio: the free cloud GPU video engine behind Creative Studio in Syok AI.

Not a standalone app: it has no screen of its own. Syok's Creative Studio sends
it video jobs through one password-protected address. Each "room" is a GPU model
that only wakes up when a job arrives, so unused rooms cost nothing.

Rooms today:
  wan  - text to video, or picture to video (Wan 2.2 TI2V 5B, Apache-2.0)
  ltx  - LTX-2 (19B distilled): text, start picture, end picture, video to video, with sound
  lipsync - LongCat-Video-Avatar 1.5 (MIT): one photo + a voice recording -> a talking video, lips in sync
  wanhd - Wan 2.2 A14B, the best-looking room: text or picture to video, slow and costly but sharp,
          honours a "do not show" (negative) prompt. 5 s clip, 16 frames per second.

Deploy:        modal deploy app.py
First time:    modal run app.py::fetch_ltx    (downloads LTX-2 on a cheap CPU, not the GPU)
               modal run app.py::fetch_wanhd  (same, for the two Wan 2.2 A14B models)
               modal run --detach app.py::fetch_longcat  (the lip-sync model; --detach keeps it running if you close the terminal)
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
WANHD_T2V = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
WANHD_I2V = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
WANHD_GPU = ["H100", "A100-80GB"]  # two 14B experts: 80 GB card + CPU offload
LONGCAT_BASE = "meituan-longcat/LongCat-Video"            # only its tokenizer, text encoder and VAE are used
LONGCAT_AVATAR = "meituan-longcat/LongCat-Video-Avatar-1.5"
LONGCAT_GPU = ["H100", "A100-80GB"]

# Head swap (BFS on LTX-2.3): the official Lightricks IC-LoRA pipeline plus Alissonerdx's BFS head-swap LoRA.
HEADSWAP_CKPT = ("Lightricks/LTX-2.3", "ltx-2.3-22b-distilled-1.1.safetensors")
HEADSWAP_UP = ("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
HEADSWAP_GEMMA = "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized"   # public copy of Google's Gemma text encoder
HEADSWAP_LORA = ("Alissonerdx/BFS-Best-Face-Swap-Video", "ltx-2.3/head_swap_v3_rank_adaptive_fro_098.safetensors")
HEADSWAP_GPU = ["H100", "A100-80GB"]
HEADSWAP_MAX_SECONDS = 8
HEADSWAP_DIR = "/cache/headswap"
LONGCAT_MAX_SECONDS = 15           # longest talking clip one request may make
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

# LongCat needs Python 3.10, torch 2.6 + CUDA 12.4 and FlashAttention 2 (a ready-made wheel, so no long compile).
longcat_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("ffmpeg", "git", "libsndfile1")
    .pip_install("torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
        "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    )
    .run_commands("git clone --depth 1 https://github.com/meituan-longcat/LongCat-Video /opt/longcat")
    .run_commands("cd /opt/longcat && sed -i '/^torch==/d;/^flash-attn/d;/^streamlit/d' requirements.txt "
                  "&& sed -i '/^libsndfile1/d;/^tritonserverclient/d' requirements_avatar.txt "
                  "&& pip install -r requirements.txt && pip install -r requirements_avatar.txt huggingface_hub")
    .env({"HF_HOME": CACHE})
)

headswap_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0", "pillow", "numpy", "huggingface_hub", "av")
    .run_commands("git clone --depth 1 https://github.com/Lightricks/LTX-2 /opt/ltx2",
                  "cd /opt/ltx2 && pip install -e packages/ltx-core -e packages/ltx-pipelines")
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
    def make(self, prompt: str, image_b64: str | None, seconds: int, ratio: str, quality: str,
             negative: str = "") -> bytes:
        import imageio
        import numpy as np
        import torch
        from PIL import Image, ImageOps

        width, height = SIZES[quality][ratio]
        frames = 4 * round(FPS * seconds / 4) + 1  # Wan needs 4k+1 frames
        gen = torch.Generator("cuda").manual_seed(int.from_bytes(os.urandom(4), "big"))
        common = dict(
            prompt=prompt,
            negative_prompt=_join_negative(NEGATIVE, negative),
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


# Wan 2.2 A14B: sizes are multiples of 16, 81 frames at 16 fps is the model's native clip
WANHD_SIZES = {
    "fast": {"16:9": (832, 480), "9:16": (480, 832), "1:1": (640, 640)},
    "hd": {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (960, 960)},
}
WANHD_FPS = 16
# The negative prompt Wan's own makers publish (it is meant to be Chinese).
WANHD_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def _join_negative(base, extra):
    extra = (extra or "").strip()[:600]
    return f"{base}, {extra}" if extra else base


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
             video_b64: str | None, seconds: int, ratio: str, quality: str,
             negative: str = "") -> bytes:  # negative is accepted but unused: the distilled model has no CFG
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
    image=wan_image,
    volumes={CACHE: cache},
    timeout=60 * 60,
    cpu=2,
)
def fetch_wanhd():
    """Run once: download both Wan 2.2 A14B models (about 120 GB) on a cheap CPU box."""
    from huggingface_hub import snapshot_download

    for repo in (WANHD_I2V, WANHD_T2V):
        snapshot_download(repo, max_workers=8)
        cache.commit()
    return "Wan 2.2 A14B downloaded"


def _save_hd(frames, path="/tmp/hd.mp4"):
    from diffusers.utils import export_to_video

    export_to_video(frames, path, fps=WANHD_FPS)
    with open(path, "rb") as fh:
        return fh.read()


@app.cls(
    image=wan_image,
    gpu=WANHD_GPU,
    volumes={CACHE: cache},
    scaledown_window=SCALEDOWN_SECONDS,
    timeout=60 * 60,
    memory=122880,   # MiB of computer memory: the two experts wait here while the other works
    max_containers=1,
)
class WanHdPicture:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

        vae = AutoencoderKLWan.from_pretrained(WANHD_I2V, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanImageToVideoPipeline.from_pretrained(WANHD_I2V, vae=vae, torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload(device="cuda")
        cache.commit()

    @modal.method()
    def make(self, prompt: str, image_b64: str, seconds: int, ratio: str, quality: str,
             negative: str = "") -> bytes:
        import torch
        from PIL import Image, ImageOps

        width, height = WANHD_SIZES[quality][ratio]
        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        img = ImageOps.fit(img, (width, height), Image.LANCZOS)
        out = self.pipe(
            image=img, prompt=prompt, negative_prompt=_join_negative(WANHD_NEGATIVE, negative),
            height=height, width=width, num_frames=WANHD_FPS * seconds + 1,
            guidance_scale=3.5, num_inference_steps=40,
            generator=torch.Generator("cuda").manual_seed(int.from_bytes(os.urandom(4), "big")),
        ).frames[0]
        return _save_hd(out)


@app.cls(
    image=wan_image,
    gpu=WANHD_GPU,
    volumes={CACHE: cache},
    scaledown_window=SCALEDOWN_SECONDS,
    timeout=60 * 60,
    memory=122880,
    max_containers=1,
)
class WanHdText:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import AutoencoderKLWan, WanPipeline

        vae = AutoencoderKLWan.from_pretrained(WANHD_T2V, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanPipeline.from_pretrained(WANHD_T2V, vae=vae, torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload(device="cuda")
        cache.commit()

    @modal.method()
    def make(self, prompt: str, seconds: int, ratio: str, quality: str, negative: str = "") -> bytes:
        import torch

        width, height = WANHD_SIZES[quality][ratio]
        out = self.pipe(
            prompt=prompt, negative_prompt=_join_negative(WANHD_NEGATIVE, negative),
            height=height, width=width, num_frames=WANHD_FPS * seconds + 1,
            guidance_scale=4.0, guidance_scale_2=3.0, num_inference_steps=40,
            generator=torch.Generator("cuda").manual_seed(int.from_bytes(os.urandom(4), "big")),
        ).frames[0]
        return _save_hd(out)


@app.function(
    image=longcat_image,
    volumes={CACHE: cache},
    timeout=60 * 60,
    cpu=2,
)
def fetch_longcat():
    """Run once (with --detach): download the lip-sync weights into the cache volume on a cheap CPU box."""
    from huggingface_hub import snapshot_download

    snapshot_download(LONGCAT_BASE, local_dir=f"{CACHE}/longcat/LongCat-Video", max_workers=8,
                      allow_patterns=["tokenizer/*", "text_encoder/*", "vae/*", "scheduler/*", "*.json"])
    snapshot_download(LONGCAT_AVATAR, local_dir=f"{CACHE}/longcat/LongCat-Video-Avatar-1.5", max_workers=8)
    cache.commit()
    return "LongCat lip-sync weights downloaded"


@app.cls(
    image=longcat_image,
    gpu=LONGCAT_GPU,
    volumes={CACHE: cache},
    scaledown_window=SCALEDOWN_SECONDS,
    timeout=40 * 60,
    max_containers=1,
)
class Lipsync:
    @modal.method()
    def make(self, prompt: str, image_b64: str, audio_b64: str, resolution: str = "480p") -> bytes:
        import glob
        import json
        import math
        import shutil
        import subprocess

        import librosa

        work = "/tmp/lipsync"
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        with open(f"{work}/face.png", "wb") as fh:
            fh.write(base64.b64decode(image_b64))
        with open(f"{work}/voice.bin", "wb") as fh:
            fh.write(base64.b64decode(audio_b64))
        subprocess.run(["ffmpeg", "-y", "-i", f"{work}/voice.bin", "-ac", "1", "-ar", "16000", f"{work}/voice.wav"],
                       check=True, capture_output=True)
        seconds = min(librosa.get_duration(path=f"{work}/voice.wav"), LONGCAT_MAX_SECONDS)
        # one segment is 93 frames (3.7 s at 25 fps); every extra segment adds 80 new frames (3.2 s)
        segments = 1 + max(0, math.ceil((seconds - 93 / 25) / (80 / 25)))
        with open(f"{work}/job.json", "w") as fh:
            json.dump({"prompt": prompt, "cond_image": f"{work}/face.png",
                       "cond_audio": {"person1": f"{work}/voice.wav"}}, fh)
        cmd = ["torchrun", "--nproc_per_node=1", "run_demo_avatar_single_audio_to_video.py",
               "--context_parallel_size=1", f"--checkpoint_dir={CACHE}/longcat/LongCat-Video-Avatar-1.5",
               "--stage_1=ai2v", f"--input_json={work}/job.json", f"--output_dir={work}/out",
               f"--resolution={resolution}", f"--num_segments={segments}",
               "--use_distill", "--model_type", "avatar-v1.5", "--use_int8"]
        run = subprocess.run(cmd, cwd="/opt/longcat", capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError("lip-sync failed: " + (run.stderr or run.stdout)[-600:])
        outs = [f for f in glob.glob(f"{work}/out/*.mp4") if "-temp" not in f and "-crop" not in f]
        if not outs:
            raise RuntimeError("lip-sync made no video: " + (run.stdout or "")[-400:])
        final = max(outs, key=os.path.getmtime)
        with open(final, "rb") as fh:
            return fh.read()


@app.function(
    image=headswap_image,
    volumes={CACHE: cache},
    timeout=60 * 60,
    cpu=2,
)
def fetch_headswap():
    """Run once (with --detach): download the head-swap weights (about 60 GB) into the cache volume on a cheap CPU box."""
    from huggingface_hub import hf_hub_download, snapshot_download

    for repo, name in (HEADSWAP_CKPT, HEADSWAP_UP, HEADSWAP_LORA):
        hf_hub_download(repo, name, local_dir=HEADSWAP_DIR)
    snapshot_download(HEADSWAP_GEMMA, local_dir=f"{HEADSWAP_DIR}/gemma", max_workers=8,
                      allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model"])
    cache.commit()
    return "head-swap weights downloaded"


@app.cls(
    image=headswap_image,
    gpu=HEADSWAP_GPU,
    volumes={CACHE: cache},
    scaledown_window=SCALEDOWN_SECONDS,
    timeout=40 * 60,
    max_containers=1,
)
class Headswap:
    @modal.method()
    def make(self, prompt: str, video_b64: str, image_b64: str, seconds: int = 5, quality: str = "fast") -> bytes:
        import shutil
        import subprocess

        import numpy as np
        from PIL import Image

        work = "/tmp/headswap"
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        with open(f"{work}/in.bin", "wb") as fh:
            fh.write(base64.b64decode(video_b64))
        face = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")

        def run(cmd):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError((r.stderr or r.stdout)[-600:])
            return r.stdout

        info = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                    "-of", "csv=p=0", f"{work}/in.bin"]).strip().split(",")
        sw, sh = int(info[0]), int(info[1])
        has_audio = bool(run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                              "stream=index", "-of", "csv=p=0", f"{work}/in.bin"]).strip())
        # the clip is shrunk so its long side is 768 (1024 for hd); the model wants sides divisible by 64
        long_side = 1024 if quality == "hd" else 768
        k = long_side / max(sw, sh)
        cw, ch = max(64, round(sw * k / 64) * 64), max(64, round(sh * k / 64) * 64)
        seconds = max(1, min(HEADSWAP_MAX_SECONDS, seconds))
        frames = 8 * int(FPS * seconds // 8) + 1
        # the guide video: the clip on the right, a green side strip on the left holding the new face on every frame
        strip = round(cw * 0.25 / 8) * 8
        area_w, area_h = cw - strip, ch
        fit = min(area_w / cw, area_h / ch)
        vw, vh = max(2, round(cw * fit / 2) * 2), max(2, round(ch * fit / 2) * 2)
        vx, vy = strip, (ch - vh) // 2
        pad = 12
        fw, fh = face.size
        fk = min((strip - 2 * pad) / fw, (ch - 2 * pad) / fh)
        face = face.resize((max(1, round(fw * fk)), max(1, round(fh * fk))), Image.LANCZOS)
        base = Image.new("RGB", (cw, ch), (0, 0, 0))
        base.paste(Image.new("RGB", (strip, ch), (0, 255, 0)), (0, 0))
        base.paste(face, ((strip - face.size[0]) // 2, (ch - face.size[1]) // 2))
        base_arr = np.array(base)
        reader = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", f"{work}/in.bin", "-t", str(seconds), "-r", str(FPS),
             "-vf", f"scale={vw}:{vh}:flags=lanczos", "-frames:v", str(frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE)
        writer = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{cw}x{ch}", "-r", str(FPS),
             "-i", "-", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", f"{work}/guide.mp4"],
            stdin=subprocess.PIPE)
        count = 0
        fsize = vw * vh * 3
        last = None
        while True:
            buf = reader.stdout.read(fsize)
            if len(buf) < fsize:
                break
            fr = base_arr.copy()
            fr[vy:vy + vh, vx:vx + vw] = np.frombuffer(buf, np.uint8).reshape(vh, vw, 3)
            writer.stdin.write(fr.tobytes())
            last = fr
            count += 1
        # pad a short clip by holding its last frame, so the frame count fits the model (8 x n + 1)
        while last is not None and count < frames:
            writer.stdin.write(last.tobytes())
            count += 1
        writer.stdin.close()
        writer.wait()
        reader.wait()
        if count < 9:
            raise RuntimeError("the clip is too short (needs about half a second)")
        frames = 8 * ((count - 1) // 8) + 1

        if not prompt.lstrip().lower().startswith("head_swap"):
            prompt = ("head_swap:\nFACE:\n" + prompt.strip() +
                      "\n\nACTION:\nA person performs the same movements as in the video.")
        # BFS ships its LoRA with ComfyUI names (lora_down / lora_up); the official pipeline reads lora_A / lora_B.
        # Convert once and keep the result on the volume. The small bias and norm "diff" tensors are not LoRA factors and are left out.
        lora = f"{HEADSWAP_DIR}/bfs_v3_peft.safetensors"
        if not os.path.exists(lora):
            from safetensors.torch import load_file, save_file
            src = load_file(f"{HEADSWAP_DIR}/{HEADSWAP_LORA[1]}")
            dst = {k.replace(".lora_down.weight", ".lora_A.weight").replace(".lora_up.weight", ".lora_B.weight"): v.contiguous()
                   for k, v in src.items() if ".lora_down.weight" in k or ".lora_up.weight" in k}
            save_file(dst, lora + ".tmp")
            os.replace(lora + ".tmp", lora)
            cache.commit()
        out = f"{work}/out.mp4"
        cmd = ["python", "-m", "ltx_pipelines.ic_lora",
               "--distilled-checkpoint-path", f"{HEADSWAP_DIR}/{HEADSWAP_CKPT[1]}",
               "--gemma-root", f"{HEADSWAP_DIR}/gemma",
               "--spatial-upsampler-path", f"{HEADSWAP_DIR}/{HEADSWAP_UP[1]}",
               "--lora", lora, "1.0",
               "--video-conditioning", f"{work}/guide.mp4", "1.0",
               "--prompt", prompt, "--height", str(ch), "--width", str(cw),
               "--num-frames", str(frames), "--frame-rate", str(FPS),
               "--seed", str(int.from_bytes(os.urandom(3), "big")),
               "--quantization", "fp8-cast", "--output-path", out]
        r = subprocess.run(cmd, cwd="/opt/ltx2", capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            keep = [ln for ln in ((r.stderr or "") + "\n" + (r.stdout or "")).splitlines()
                    if ln.strip() and "nanobind" not in ln and "leaked" not in ln and not ln.startswith(" - ")]
            raise RuntimeError(f"head swap failed (exit {r.returncode}): " + "\n".join(keep)[-1500:])
        # keep only the video area (the green strip is cut off), bring it back to the clip's own shape and sound
        final = f"{work}/final.mp4"
        crop = f"crop={vw}:{vh}:{vx}:{vy},scale={round(sw * k / 2) * 2}:{round(sh * k / 2) * 2}:flags=lanczos"
        fc = ["ffmpeg", "-y", "-v", "error", "-i", out]
        if has_audio:
            fc += ["-i", f"{work}/in.bin", "-map", "0:v", "-map", "1:a?", "-shortest"]
        fc += ["-vf", crop, "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac", final]
        run(fc)
        with open(final, "rb") as fh:
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
        audio: str | None = None  # room lipsync only: the voice recording (data: URL), 15 s are used at most
        negative_prompt: str | None = None  # "do not show" list (Wan rooms; LTX distilled ignores it)

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
        return {"ok": True, "rooms": ["wan", "ltx", "wanhd", "lipsync", "headswap"], "max_seconds": MAX_SECONDS}

    @api.post("/v1/video")
    def start(job: VideoJob, authorization: str | None = Header(default=None)):
        check(authorization)
        prompt = job.prompt.strip()
        if not 3 <= len(prompt) <= 2000:
            raise HTTPException(400, "prompt must be 3 to 2000 characters")
        if job.ratio not in SIZES["fast"] or job.quality not in SIZES:
            raise HTTPException(400, "ratio must be 16:9, 9:16 or 1:1; quality fast or hd")
        if job.room not in ("wan", "ltx", "wanhd", "lipsync", "headswap"):
            raise HTTPException(400, "room must be wan, ltx, wanhd, lipsync or headswap")
        seconds = max(1, min(MAX_SECONDS, job.seconds))
        image = raw(job.image, 14_000_000)  # about 10 MB of picture
        negative = (job.negative_prompt or "").strip()[:600]

        if job.room == "lipsync":
            audio = raw(job.audio, 20_000_000)  # about 15 MB of sound
            if not image or not audio:
                raise HTTPException(400, "lipsync needs a picture of the face (image) and a voice recording (audio)")
            call = Lipsync().make.spawn(prompt, image, audio, "720p" if job.quality == "hd" else "480p")
            return {"id": call.object_id, "status": "running", "room": "lipsync"}

        if job.room == "headswap":
            video = raw(job.video, 52_000_000)
            if not image or not video:
                raise HTTPException(400, "headswap needs a clip (video) and a photo of the new face (image)")
            call = Headswap().make.spawn(prompt, video, image, seconds, job.quality)
            return {"id": call.object_id, "status": "running", "room": "headswap"}

        if job.room == "wanhd":
            if job.last_image or job.video:
                raise HTTPException(400, "end picture and video-in need room ltx")
            if image:
                call = WanHdPicture().make.spawn(prompt, image, seconds, job.ratio, job.quality, negative)
            else:
                call = WanHdText().make.spawn(prompt, seconds, job.ratio, job.quality, negative)
            return {"id": call.object_id, "status": "running", "room": "wanhd"}

        if job.room == "ltx":
            call = Ltx().make.spawn(
                prompt, image, raw(job.last_image, 14_000_000), raw(job.video, 52_000_000),
                seconds, job.ratio, job.quality, negative,
            )
            return {"id": call.object_id, "status": "running", "room": "ltx"}
        if job.last_image or job.video:
            raise HTTPException(400, "end picture and video-in need room ltx")
        call = Wan().make.spawn(prompt, image, seconds, job.ratio, job.quality, negative)
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
