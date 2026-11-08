# Avatar-Forever on Apple Silicon (MLX)

Long talking-head videos from one portrait and an audio track, rendered locally on a Mac. This adds
[Avatar-Forever](https://github.com/leeruibin/avatarforever) to [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx).
Avatar-Forever is an LTX-2.3 22B model fine-tuned for avatars that stay stable for minutes. Upstream it runs only on
CUDA GPUs.

The model generates the whole frame, not just the mouth, so head movement, gestures and posture follow the speech.

## Results we measured (Apple M5 Ultra, 256 GB)

| | |
|---|---|
| Speed | 60 s of 768x512 video in about 12 min (10 s in about 1 min 45 s, including model load) |
| Peak memory | 29-41 GB |
| Identity drift over 60 s | none measurable (face match to the portrait: 0.95 at the start, 0.94 at the end) |
| Lip timing vs a real recording of the same audio | 0-80 ms over 60 s, with the audio-split fix below |

## What this fork adds

| File | What |
|---|---|
| `packages/ltx-core-mlx/.../transformer/model.py` | the gated first-frame channel condition that Avatar-Forever adds after `patchify_proj` |
| `packages/ltx-pipelines-mlx/.../ar_avatar_forever.py` | the chunked generation loop: 4 latent frames per chunk; one history chunk plus the first chunk as a fixed anchor; relative positions, so the length is unbounded |
| `scripts/af_convert.py` | converts the original PyTorch checkpoint into an MLX int8 pack, using MLX only |
| `scripts/run_avatar_forever.py` | command-line runner |
| `examples/talking-head/prep_look.py` | crops a portrait to the video frame without cutting off the top of the head |
| `tests/test_ar_avatar_forever.py` | unit tests for the chunk and audio scheduling |

**One fix over upstream:** upstream splits the audio across chunks in proportion to latent frames. But the first
latent frame covers 1 video frame and every later one covers 8, so the mouth trails the speech, by 80 ms at first and
160 ms after a minute. This port splits the audio at each chunk's true video time by default, which keeps lip timing
within 0-80 ms. Set `AF_AUDIO_ALIGN=upstream` to reproduce the original behaviour.

## Setup

Requirements: an Apple Silicon Mac, `uv` and `ffmpeg`. Tested only on a 256 GB machine. Rendering peaked at 41 GB, so
64 GB Macs may work for rendering (untested); the conversion step holds the 46 GB checkpoint in memory.

```bash
git clone https://github.com/nexaiadvisors/avatar-forever-mlx && cd avatar-forever-mlx
uv sync
# 1. The stock LTX-2.3 MLX pack (VAE, audio VAE, vocoder, text connector); see the main README for the download.
# 2. The Avatar-Forever weights (46 GB)
HF_HUB_DISABLE_XET=1 uv run hf download LetsThink/AvatarForever avatarforever-ltx-2.3-22b.safetensors --local-dir ~/models/avatarforever
# 3. Convert to an MLX int8 pack (holds the 46 GB checkpoint in memory)
uv run python scripts/af_convert.py ~/models/avatarforever/avatarforever-ltx-2.3-22b.safetensors ~/models/ltx-2.3-mlx-q8 ~/models/avatarforever-mlx-q8
```

The converter prints `extra keys` (the 3 `video_channel_condition_*` tensors) and `missing keys: [] 0`. Any other
result means the checkpoint does not match.

## Make a video

```bash
# Optional: crop your portrait to 768x512 with room above the head (exits 1 if the photo is too tight)
uv run python examples/talking-head/prep_look.py portrait.png look-768x512.jpg 768 512

# Frames must be 8n+1 at 25 fps, and the audio must be at least that long (pad it with 1 s of silence)
uv run python scripts/run_avatar_forever.py --model ~/models/avatarforever-mlx-q8 \
  --image look-768x512.jpg --audio speech.wav --out talk.mp4 --frames 1497 \
  --prompt "A man in a dark blazer talks to the camera with energy and warm smiles, natural head movements and clear lip movements. Static camera, medium shot, full head in frame, photorealistic."
```

Tips from our tests:
- Describe the person and setting first, then the delivery you want. The prompt changes how much he moves.
- The voice carries the emotion: a flat recording gives a flat performance. An expressive TTS with emotion tags
  (we used Fish Audio's Drama 3 locally through `mlx-audio`) gives a livelier result.
- Likeness can never exceed the portrait's: start from a photo that really looks like the person.
- For 1080p, upscale afterwards. RelayVSR 2x (Wan-based generative upscaler) added real hair and skin detail in about 3 min per 45 s.

## Licenses (read before you use or share this)

- The ltx-2-mlx base code is MIT (`LICENSE`, © dgrauet).
- The Avatar-Forever port (the files above marked NOTICE) and the converted weights are Derivatives of LTX-2. They are
  distributed under the **LTX-2 Community License** (`LICENSE-LTX-2`). It is free for individuals and for companies
  under \$10M annual revenue, and its use-based restrictions (Attachment A) apply to you. That includes no deceptive
  impersonation, and no use in a product that competes with Lightricks' commercial offerings without a separate
  license.
- Only make videos of people who agreed, and label them as AI-generated wherever you publish them.

This is not legal advice. Read both license files.

## Credits

Avatar-Forever by its authors ([paper](https://arxiv.org/abs/2608.12107), [code](https://github.com/leeruibin/avatarforever)).
LTX-2 by Lightricks. ltx-2-mlx by dgrauet. MLX port and the audio-split fix by [NexAI Advisors](https://nexaiadvisors.com).
