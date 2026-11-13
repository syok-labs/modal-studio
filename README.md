# Modal Studio: the free cloud GPU video engine for Creative Studio

Modal Studio is not a standalone app and has no screen of its own. It is the video engine that serves **Creative Studio in Syok AI**. You deploy it once to your own Modal account, then connect it in Syok. Creative Studio sends it video jobs and shows the finished clips.

Modal's Starter plan includes about US$30 of free compute a month. Set a spending limit in Modal's billing settings first.

Today it has these rooms (one address and one password for all of them):

| Room | What it makes | Model |
|---|---|---|
| `wan` | Text to video, or picture to video (5 seconds at most) | Wan 2.2 TI2V 5B (Apache-2.0) |
| `ltx` | Start and end picture, video to video, with sound | LTX-2 19B distilled |
| `wanhd` | Higher quality text or picture to video | Wan 2.2 A14B |
| `lipsync` | A face photo plus a voice recording gives a talking video | LongCat-Video-Avatar 1.5 |
| `headswap` | A clip plus a photo of a new face gives the same clip with the head swapped (3 to 5 seconds) | BFS head-swap LoRA on LTX-2.3 (LTX-2 community license) |

Only use footage and photos of people who have agreed to it.

One-time model downloads (run once each, they go to your Modal volume on a cheap CPU box): `modal run --detach app.py::fetch_ltx`, `::fetch_wanhd`, `::fetch_longcat`, `::fetch_headswap` (about 60 GB).

## Deploy (4 commands)

1. `pip install modal`
2. `modal token new` (sign in in the browser tab it opens)
3. `modal secret create modal-studio-key MODAL_STUDIO_KEY=make-up-a-long-password`
4. `modal deploy app.py`

The last command prints an address ending in `.modal.run`. In Syok choose "Free cloud GPU (Modal)" for video, type that address (without https://) and your password, then press Test.

## Good to know

- The GPU sleeps after 3 idle minutes. The first clip after a sleep waits 2 to 5 minutes while the model loads (the very first time it also downloads about 25 GB).
- A 3 second clip in "fast" quality (480p) took about 48 seconds on an L40S GPU in our test. Cost is shown in your Modal dashboard. Only one GPU runs at a time, so your bill cannot jump.
- Only you can use it: every request needs your password.

## Direct API (for developers)

`POST /v1/video` with header `Authorization: Bearer <your password>` and JSON `{"prompt": "...", "room": "wan", "image": "<optional data: URL>", "video": "<data: URL, ltx and headswap>", "audio": "<data: URL, lipsync>", "seconds": 5, "ratio": "16:9", "quality": "fast"}` returns `{"id": "..."}`. Then `GET /v1/video/<id>` until `status` is `done` (the clip comes back as a `data:video/mp4;base64,...` string). `GET /v1/status` lists the rooms.
