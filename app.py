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
  headswap - BFS head swap on LTX-2.3: a clip + one face photo per person (up to 4, matched to the heads left to right).
          Heads are tracked and swapped one by one at a fixed size, then fitted back to the original head's size and position.

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
    .pip_install("torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0", "pillow", "numpy", "huggingface_hub", "av",
                 "opencv-python-headless")
    .run_commands("python -c \"import urllib.request as u; u.urlretrieve('https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx', '/opt/yunet.onnx')\"",
                  "test $(stat -c %s /opt/yunet.onnx) -gt 100000",
                  "git clone --depth 1 https://github.com/Lightricks/LTX-2 /opt/ltx2",
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


# ---------------------------------------------------------------------------------------------
# Head swap helpers (CPU only). The clip is never swapped as a whole: every head is found and tracked, cut out as a
# head-sized square, swapped there (BFS works best when the head is a fixed size in the picture), then moved and scaled
# so the new head sits exactly where the old one was, and blended back into the original frames. Several people in one
# scene are swapped one after another, left to right.
# ---------------------------------------------------------------------------------------------
HS_CROPK = 3.0     # a head crop is 3 face-heights wide: head, hair and the top of the shoulders
HS_REFK = 1.7      # the reference photo is cut to 1.7 face-heights: hair to chin
HS_MAX_PEOPLE = 4
HS_YUNET = "/opt/yunet.onnx"
HS_YUNET_URL = "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx"


def hs_faces(frame_bgr, detector=None):
    """Every face in one frame as (x, y, w, h, score), in pixels."""
    import cv2

    h, w = frame_bgr.shape[:2]
    det = detector or cv2.FaceDetectorYN.create(HS_YUNET, "", (w, h), 0.5, 0.3, 5000)
    det.setInputSize((w, h))
    _, rows = det.detect(frame_bgr)
    # (x, y, w, h, score, landmarks): landmarks are the two eyes, the nose tip and the two mouth corners as 10 numbers
    return [] if rows is None else [(float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[-1]), [float(v) for v in r[4:14]]) for r in rows]


def _hs_smooth(a, k=9):
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    if len(a) < 3:
        return a
    k = min(k, len(a) // 2 * 2 + 1)
    pad = np.pad(a, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def hs_tracks(dets, n):
    """Follow each face through the clip. dets[i] = faces of frame i. Returns one dict per person, left to right:
    "box" = (n, 4) smooth (centre x, centre y, width, height) used to cut the head crop, "lm" = (n, 10) lightly smoothed face landmarks.
    Gaps are filled in, tiny false hits are dropped."""
    import numpy as np

    tracks = []
    for i, faces in enumerate(dets):
        pairs = []
        for ti, t in enumerate(tracks):
            if i - t["last_i"] > 10:
                continue
            lx, ly, lw, lh = t["last"]
            for fi, f in enumerate(faces):
                fx, fy = f[0] + f[2] / 2, f[1] + f[3] / 2
                dist = ((fx - lx) ** 2 + (fy - ly) ** 2) ** 0.5
                if dist < 0.8 * max(lh, f[3]):
                    pairs.append((dist, ti, fi))
        pairs.sort()
        taken_t, taken_f = set(), set()
        for _, ti, fi in pairs:
            if ti in taken_t or fi in taken_f:
                continue
            taken_t.add(ti)
            taken_f.add(fi)
            f = faces[fi]
            box = (f[0] + f[2] / 2, f[1] + f[3] / 2, f[2], f[3])
            tracks[ti]["boxes"][i] = box + tuple(f[5])
            tracks[ti]["last"], tracks[ti]["last_i"] = box, i
        for fi, f in enumerate(faces):
            if fi not in taken_f and f[4] >= 0.6:
                box = (f[0] + f[2] / 2, f[1] + f[3] / 2, f[2], f[3])
                tracks.append({"boxes": {i: box + tuple(f[5])}, "last": box, "last_i": i})
    keep = [t for t in tracks if len(t["boxes"]) >= max(4, int(0.2 * n))]
    out = []
    xs = np.arange(n)
    for t in keep:
        idx = np.array(sorted(t["boxes"]))
        arr = np.array([t["boxes"][i] for i in idx])
        cx = _hs_smooth(np.interp(xs, idx, arr[:, 0]))
        cy = _hs_smooth(np.interp(xs, idx, arr[:, 1]))
        h = np.exp(_hs_smooth(np.interp(xs, idx, np.log(arr[:, 3]))))
        ratio = float(np.median(arr[:, 2] / arr[:, 3]))
        lm = np.stack([_hs_smooth(np.interp(xs, idx, arr[:, 4 + j]), 5) for j in range(10)], axis=1)
        out.append({"box": np.stack([cx, cy, h * ratio, h], axis=1), "lm": lm})
    out.sort(key=lambda t: float(np.median(t["box"][:, 0])))
    return out


def hs_crop_matrix(box, R):
    """Affine matrix (source -> R x R head crop) that puts the face at a fixed size in the middle of the crop."""
    import numpy as np

    cx, cy, _, fh = box
    r = R / (HS_CROPK * fh)
    cy0 = cy - 0.12 * fh      # head centre sits a little above the middle of the face box
    return np.array([[r, 0, R / 2 - r * cx], [0, r, R / 2 - r * cy0]], dtype=np.float64)


def hs_reference(photo_bgr, side, detector=None):
    """The new face: cut the head out of the photo (hair to chin) and return a side x side picture (BGR)."""
    import cv2
    import numpy as np

    h, w = photo_bgr.shape[:2]
    faces = hs_faces(photo_bgr, detector)
    if faces:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])[:4]
        cx, cy, s = x + fw / 2, y + fh / 2 - 0.1 * fh, HS_REFK * fh
    else:   # no face found: use the middle of the photo
        cx, cy, s = w / 2, h / 2, min(w, h)
    r = side / s
    m = np.array([[r, 0, side / 2 - r * cx], [0, r, side / 2 - r * cy]], dtype=np.float64)
    return cv2.warpAffine(photo_bgr, m, (side, side), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def hs_gen_landmarks(gen_frames_bgr, R, detector=None):
    """Where are the face landmarks of the (single) swapped face in each generated crop? Gaps are filled in. (n, 10) or None."""
    import numpy as np

    n = len(gen_frames_bgr)
    pts = {}
    for i, fr in enumerate(gen_frames_bgr):
        faces = hs_faces(fr, detector)
        if faces:
            f = min(faces, key=lambda f: (f[0] + f[2] / 2 - R / 2) ** 2 + (f[1] + f[3] / 2 - R / 2) ** 2)
            pts[i] = f[5]
    if len(pts) < max(3, n // 5):
        return None
    idx = np.array(sorted(pts))
    arr = np.array([pts[i] for i in idx])
    xs = np.arange(n)
    return np.stack([_hs_smooth(np.interp(xs, idx, arr[:, j]), 5) for j in range(10)], axis=1)


def hs_mask(R):
    """Soft egg-shaped blend mask (just the head: face, hair and chin, never the shoulders or the background) in head-crop coordinates (1 = new head, 0 = original)."""
    import cv2
    import numpy as np

    fhc = R / HS_CROPK
    m = np.zeros((R, R), np.float32)
    cv2.ellipse(m, (R // 2, R // 2 - int(0.12 * fhc)), (int(0.68 * fhc), int(0.92 * fhc)), 0, 0, 360, 1.0, -1)
    s = max(3, int(0.2 * fhc)) | 1
    return np.clip(cv2.GaussianBlur(m, (s * 2 + 1, s * 2 + 1), s / 2.0), 0, 1)


def hs_blend(frames, track, gen_crops, gen_lm, R, mask):
    """Put the swapped head of one person back into every frame (in place). frames: list of BGR uint8 arrays, track: that
    person's track from hs_tracks, gen_crops: the generated R x R BGR crops, gen_lm: face landmarks found in those crops (or None).
    The new head is turned, scaled and moved so its eyes, nose and mouth land on the old ones, then blended in with a soft mask."""
    import cv2
    import numpy as np

    n = len(frames)
    H, W = frames[0].shape[:2]
    fhc = R / HS_CROPK
    boxes, lms = track["box"], track["lm"]
    mats = [hs_crop_matrix(boxes[i], R) for i in range(n)]
    aligns = None
    if gen_lm is not None:   # similarity transform (scale, turn, shift) per frame, then smoothed over time
        p = np.zeros((n, 4))
        for i in range(n):
            src = (mats[i][:, :2] @ lms[i].reshape(5, 2).T + mats[i][:, 2:3]).T   # old landmarks in head-crop coordinates
            m, _ = cv2.estimateAffinePartial2D(gen_lm[i].reshape(5, 2).astype(np.float32), src.astype(np.float32), method=cv2.LMEDS)
            if m is None:
                m = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
            sc = float(np.hypot(m[0, 0], m[1, 0]))
            if not 0.5 <= sc <= 1.8 or abs(np.degrees(np.arctan2(m[1, 0], m[0, 0]))) > 35:
                m = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
            p[i] = (m[0, 0], m[1, 0], m[0, 2], m[1, 2])
        p = np.stack([_hs_smooth(p[:, j], 5) for j in range(4)], axis=1)
        aligns = [np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64) for a, b, tx, ty in p]
    offs = np.zeros((n, 3))
    jobs = []
    for i in range(n):
        inv = cv2.invertAffineTransform(mats[i])
        g = gen_crops[i]
        mk = mask
        if aligns is not None:
            g = cv2.warpAffine(g, aligns[i], (R, R), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            ok = cv2.warpAffine(np.ones((R, R), np.float32), aligns[i], (R, R), flags=cv2.INTER_LINEAR, borderValue=0)
            ok = cv2.GaussianBlur(cv2.erode(ok, np.ones((7, 7), np.uint8)), (15, 15), 0)   # never blend pixels that were stretched in from outside the picture
            mk = mask * ok
        gw = cv2.warpAffine(g, inv, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        mw = cv2.warpAffine(mk, inv, (W, H), flags=cv2.INTER_LINEAR)
        ring = (mw > 0.08) & (mw < 0.5)
        if ring.sum() > 400:   # light correction so the head's tones follow the scene (kept small)
            offs[i] = frames[i][ring].mean(0) - gw[ring].mean(0)
        jobs.append((gw, mw))
    for j in range(3):
        offs[:, j] = _hs_smooth(offs[:, j], 9)
    offs = np.clip(offs * 0.6, -14, 14)
    for i, (gw, mw) in enumerate(jobs):
        gf = np.clip(gw.astype(np.float32) + offs[i], 0, 255)
        a = mw[..., None]
        frames[i][:] = np.clip(frames[i].astype(np.float32) * (1 - a) + gf * a, 0, 255).astype(np.uint8)


@app.cls(
    image=headswap_image,
    gpu=HEADSWAP_GPU,
    volumes={CACHE: cache},
    scaledown_window=SCALEDOWN_SECONDS,
    timeout=60 * 60,
    max_containers=1,
)
class Headswap:
    def _bfs(self, work, guide, prompt, cw, ch, frames, out):
        """One BFS run: the guide video (new face on the left strip, head crop on the right) -> the swapped video."""
        import subprocess

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
        if os.path.exists(out):
            os.remove(out)
        cmd = ["python", "-m", "ltx_pipelines.ic_lora",
               "--distilled-checkpoint-path", f"{HEADSWAP_DIR}/{HEADSWAP_CKPT[1]}",
               "--gemma-root", f"{HEADSWAP_DIR}/gemma",
               "--spatial-upsampler-path", f"{HEADSWAP_DIR}/{HEADSWAP_UP[1]}",
               "--lora", lora, "1.0",
               "--video-conditioning", guide, "1.0",
               "--prompt", prompt, "--height", str(ch), "--width", str(cw),
               "--num-frames", str(frames), "--frame-rate", str(FPS),
               "--seed", str(int.from_bytes(os.urandom(3), "big")),
               "--quantization", "fp8-cast", "--output-path", out]
        r = subprocess.run(cmd, cwd="/opt/ltx2", capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            keep = [ln for ln in ((r.stderr or "") + "\n" + (r.stdout or "")).splitlines()
                    if ln.strip() and "nanobind" not in ln and "leaked" not in ln and not ln.startswith(" - ")]
            raise RuntimeError(f"head swap failed (exit {r.returncode}): " + "\n".join(keep)[-1500:])

    @modal.method()
    def make(self, prompt: str, video_b64: str, images_b64: list, seconds: int = 5, quality: str = "fast") -> bytes:
        import shutil
        import subprocess

        import cv2
        import numpy as np

        work = "/tmp/headswap"
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)
        with open(f"{work}/in.bin", "wb") as fh:
            fh.write(base64.b64decode(video_b64))
        photos = []
        for b in (images_b64 or [])[:HS_MAX_PEOPLE]:
            arr = np.frombuffer(base64.b64decode(b), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("one of the face photos could not be read (use a JPG or PNG)")
            photos.append(img)
        if not photos:
            raise RuntimeError("add at least one photo of the new face")

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
        # the scene is worked on at most 1280 pixels on its long side; the head crops are what the model sees
        k = min(1.0, 1280 / max(sw, sh))
        W, H = max(2, round(sw * k / 2) * 2), max(2, round(sh * k / 2) * 2)
        seconds = max(1, min(HEADSWAP_MAX_SECONDS, seconds))
        want = 8 * int(FPS * seconds // 8) + 1
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", f"{work}/in.bin", "-t", str(seconds), "-r", str(FPS),
             "-vf", f"scale={W}:{H}:flags=lanczos", "-frames:v", str(want), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            capture_output=True).stdout
        fsz = W * H * 3
        frames = [np.frombuffer(raw[i * fsz:(i + 1) * fsz], np.uint8).reshape(H, W, 3).copy() for i in range(len(raw) // fsz)]
        if len(frames) < 9:
            raise RuntimeError("the clip is too short (needs about half a second)")
        while (len(frames) - 1) % 8:   # the model wants 8 x n + 1 frames: hold the last frame a moment
            frames.append(frames[-1].copy())
        n = len(frames)

        det = cv2.FaceDetectorYN.create(HS_YUNET, "", (W, H), 0.5, 0.3, 5000)
        dets = [hs_faces(f, det) for f in frames]
        tracks = hs_tracks(dets, n)
        if not tracks:
            raise RuntimeError("no face found in the clip. Use a clip where the face is clearly visible (not tiny, not turned away).")
        if len(photos) > len(tracks):
            raise RuntimeError(f"found {len(tracks)} {'person' if len(tracks) == 1 else 'people'} in the clip but you gave {len(photos)} photos. "
                               "Remove the extra photos, or use a clip where every head is clearly visible.")
        if not prompt.lstrip().lower().startswith("head_swap"):
            prompt = ("head_swap:\nFACE:\n" + prompt.strip() +
                      "\n\nACTION:\nA person performs the same movements as in the video.")
        R = 768 if quality == "hd" else 576
        strip = R // 3                      # the green side strip is a quarter of the canvas
        cw, ch = strip + R, R
        mask = hs_mask(R)
        source = [f.copy() for f in frames] if len(photos) > 1 else frames
        for who, photo in enumerate(photos):   # people are numbered left to right
            track = tracks[who]
            tbox = track["box"]
            ref = hs_reference(photo, strip - 24, det)
            canvas = np.zeros((ch, cw, 3), np.uint8)
            canvas[:, :strip] = (0, 255, 0)
            y0 = (ch - ref.shape[0]) // 2
            canvas[y0:y0 + ref.shape[0], 12:12 + ref.shape[1]] = ref
            writer = subprocess.Popen(
                ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{cw}x{ch}", "-r", str(FPS),
                 "-i", "-", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", f"{work}/guide{who}.mp4"],
                stdin=subprocess.PIPE)
            for i in range(n):
                m = hs_crop_matrix(tbox[i], R)
                canvas[:, strip:] = cv2.warpAffine(source[i], m, (R, R), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                writer.stdin.write(canvas.tobytes())
            writer.stdin.close()
            writer.wait()
            self._bfs(work, f"{work}/guide{who}.mp4", prompt, cw, ch, n, f"{work}/out{who}.mp4")
            gen = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{work}/out{who}.mp4", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                                 capture_output=True).stdout
            gsz = cw * ch * 3
            crops = [np.frombuffer(gen[i * gsz:(i + 1) * gsz], np.uint8).reshape(ch, cw, 3)[:, strip:].copy() for i in range(len(gen) // gsz)]
            if len(crops) < n:
                raise RuntimeError("the model returned a shorter video than expected")
            glm = hs_gen_landmarks(crops[:n], R, det)
            hs_blend(frames, track, crops[:n], glm, R, mask)

        raw_out = f"{work}/raw.mp4"
        enc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
             "-i", "-", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", raw_out], stdin=subprocess.PIPE)
        for f in frames:
            enc.stdin.write(f.tobytes())
        enc.stdin.close()
        enc.wait()
        final = f"{work}/final.mp4"
        fc = ["ffmpeg", "-y", "-v", "error", "-i", raw_out]
        if has_audio:
            fc += ["-i", f"{work}/in.bin", "-map", "0:v", "-map", "1:a?", "-shortest"]
        fc += ["-c:v", "copy", "-c:a", "aac", final]
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
        images: list[str] | None = None  # room headswap only: one new-face photo per person in the clip, left to right
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
        seconds = max(1, min(HEADSWAP_MAX_SECONDS if job.room == "headswap" else MAX_SECONDS, job.seconds))
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
            photos = [p for p in (raw(x, 14_000_000) for x in (job.images or [])[:HS_MAX_PEOPLE]) if p] or ([image] if image else [])
            if not photos or not video:
                raise HTTPException(400, "headswap needs a clip (video) and a photo of the new face (image, or images for several people)")
            call = Headswap().make.spawn(prompt, video, photos, seconds, job.quality)
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
