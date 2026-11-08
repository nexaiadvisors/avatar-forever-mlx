"""Avatar-Forever: long-horizon audio-to-video on an LTX-2.3 Avatar-Forever checkpoint, chunk by chunk.

MLX port of the inference path of https://github.com/leeruibin/avatarforever (LTX-2 Community License):
``inference.py`` defaults + ``ltx_pipelines/utils/autoregressive.py:autoregressive_euler_denoising_loop`` +
``ARA2VidDistilledPipeline`` one-stage mode. What it reproduces (upstream defaults):

- one stage at full resolution (768x512, 25 fps), deterministic Euler, sigmas (1.0, 0.98125, 0.909375, 0.421875, 0.0);
- the video is generated ``chunk`` latent frames at a time (4). Each chunk is denoised through the whole sigma
  schedule while it attends to: the first chunk (the "sink"), ``history`` previous chunk(s) (1), and itself.
  Those context chunks are already clean (denoise mask 0);
- relative positions: the selected tokens take the positions of the first tokens of the clip, so RoPE never grows
  and the length is unbounded;
- the first-frame channel condition: the reference image's latent, repeated on every latent frame, is added after
  ``patchify_proj`` through the checkpoint's gated projection, on the current chunk's tokens only;
- the input audio is a frozen conditioning stream (mask 0), split into contiguous ranges aligned with the chunks.

Not ported: the experimental history feature cache (off by default upstream), the first-frame prefix condition and
RoPE temporal rebasing (both off by default).
NOTICE: this file is a modified derivative of Avatar-Forever (https://github.com/leeruibin/avatarforever), which is
distributed under the LTX-2 Community License (LICENSE-LTX-2 in this repository, including its use-based restrictions
in Attachment A). Ported to MLX and modified by NexAI Advisors in 2026; changes from upstream: MLX rewrite, and the
audio split follows pixel time by default (see pixel_time_audio_ranges).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import (
    compute_audio_positions,
    compute_audio_token_count,
    compute_video_positions,
)
from ltx_pipelines_mlx.a2vid_distilled import A2VidDistilledPipeline
from ltx_pipelines_mlx.a2vid_two_stage import decode_with_source_audio, encode_source_audio
from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings
from ltx_pipelines_mlx.utils.blocks import snap_num_frames
from ltx_pipelines_mlx.utils.helpers import create_noised_state
from ltx_pipelines_mlx.utils.samplers import denoise_loop
from ltx_pipelines_mlx.utils.types import ImageConditioningInput

AVATAR_FOREVER_SIGMAS = [1.0, 0.98125, 0.909375, 0.421875, 0.0]


def chunk_ranges(total: int, size: int) -> list[tuple[int, int]]:
    """Non-overlapping progressive ranges (upstream ``build_progressive_chunk_ranges``)."""
    return [(s, min(s + size, total)) for s in range(0, total, size)]


def aligned_audio_ranges(video_ranges: list[tuple[int, int]], total_frames: int, total_audio: int) -> list[tuple[int, int]]:
    """Contiguous audio ranges that follow video progress (upstream ``build_aligned_audio_chunk_ranges``)."""
    out, start = [], 0
    for i, (_s, e) in enumerate(video_ranges):
        end = total_audio if i == len(video_ranges) - 1 else math.ceil(e * total_audio / total_frames)
        if end <= start:
            end = min(total_audio, start + 1)
        out.append((start, min(total_audio, end)))
        start = min(total_audio, end)
    return out


def pixel_time_audio_ranges(video_ranges: list[tuple[int, int]], num_frames: int, total_audio: int) -> list[tuple[int, int]]:
    """Split the audio at each chunk's true pixel-time boundary (this port's default).

    Latent frame 0 covers 1 pixel frame and every later latent frame covers 8, so upstream's proportional split
    (``aligned_audio_ranges``) gives chunk 0 about 0.28 s too much audio. Measured on a 60 s clip, the mouth then
    trails the speech by 80 ms, growing to 160 ms; with this split it stays within 0-80 ms.
    """
    def px(e: int) -> int:
        return 0 if e == 0 else 1 + 8 * (e - 1)

    bounds = [round(px(e) / num_frames * total_audio) for _s, e in video_ranges]
    bounds[-1] = total_audio
    return [(0 if i == 0 else bounds[i - 1], bounds[i]) for i in range(len(video_ranges))]


def selected_chunks(k: int, history: int | None, sink_first: bool) -> list[int]:
    """Chunks visible while denoising chunk ``k`` (upstream ``select_inference_chunk_indices``)."""
    start = 0 if history is None else max(0, k - history)
    sel = list(range(start, k + 1))
    if sink_first and k > 0 and 0 not in sel:
        sel = [0, *sel]
    return sel


class AvatarForeverPipeline(A2VidDistilledPipeline):
    """``ltx-2-mlx`` driver for an Avatar-Forever checkpoint (see module docstring)."""

    def generate_and_save(  # type: ignore[override]
        self,
        prompt: str,
        output_path: str,
        audio_path: str | Path | None = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 161,
        *,
        frame_rate: float = 25.0,
        seed: int = 42,
        image: str | None = None,
        chunk: int = 4,
        history: int | None = 1,
        sink_first: bool = True,
        audio_start_time: float = 0.0,
        sigmas: list[float] | None = None,
        **_ignored,
    ) -> str:
        if audio_path is None:
            raise ValueError("audio_path is required")
        if image is None:
            raise ValueError("Avatar-Forever needs the reference image (--image)")
        sigmas = sigmas or AVATAR_FOREVER_SIGMAS
        num_frames = snap_num_frames(num_frames)

        self._source_audio_tokens = encode_source_audio(
            self, audio_path, num_frames=num_frames, frame_rate=frame_rate,
            start_time=audio_start_time, max_duration=num_frames / frame_rate,
        )
        try:
            video_latent = self._generate_ar(prompt, height, width, num_frames, frame_rate, seed, image,
                                             chunk, history, sink_first, sigmas)
        finally:
            self._source_audio_tokens = None

        if self.low_memory:
            self.dit = None
            self.prompt_encoder.free()
            self.image_conditioner.free()
            self.upsampler = None
            self._loaded = False
            aggressive_cleanup()
        self._load_decoders()
        decode_with_source_audio(self, video_latent, output_path, audio_path=audio_path,
                                 start_time=audio_start_time, num_frames=num_frames, frame_rate=frame_rate)
        return output_path

    def _generate_ar(self, prompt, height, width, num_frames, frame_rate, seed, image, chunk, history, sink_first,
                     sigmas) -> mx.array:
        self._load_text_encoder()
        video_embeds, audio_embeds = self._encode_text(prompt)
        mx.eval(video_embeds, audio_embeds)
        if self.low_memory:
            self.prompt_encoder.free()
            aggressive_cleanup()
        self.load()
        dit = self.dit
        if not hasattr(dit, "video_channel_condition_proj"):
            raise ValueError("This checkpoint has no video_channel_condition_proj: not an Avatar-Forever pack")

        F, H, W = compute_video_latent_shape(num_frames, height, width)
        tpf = H * W
        n_video = F * tpf
        audio_T = compute_audio_token_count(num_frames, frame_rate=frame_rate)
        video_positions = compute_video_positions(F, H, W, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_T)

        images = self.image_conditioner.resolve_crf([ImageConditioningInput.at(image, 0, 1.0)])
        conds = combined_image_conditionings(images, enc_h=H * 32, enc_w=W * 32, spatial_dims=(F, H, W),
                                             video_encoder=self.vae_encoder, frame_rate=frame_rate)
        vs = create_noised_state(base_shape=(1, n_video, 128), conditionings=conds, spatial_dims=(F, H, W),
                                 positions=video_positions, seed=seed, sigma=1.0, initial_latent=None,
                                 legacy_scalar_blend=True)
        if vs.latent.shape[1] != n_video:
            raise ValueError(f"image conditioning appended tokens ({vs.latent.shape[1]} vs {n_video}); expected a replace")
        audio = self._frozen_audio_state((1, audio_T, 128), audio_positions, (F, H, W), seed + 1)

        # First-frame channel condition: the conditioned frame-0 latent (the image), repeated on every latent frame.
        first = vs.clean_latent[:, :tpf].astype(mx.float32)
        cc_full = mx.tile(first, (1, F, 1))

        latent = vs.latent
        vranges = chunk_ranges(F, chunk)
        if os.environ.get("AF_AUDIO_ALIGN", "pixel") == "pixel":
            aranges = pixel_time_audio_ranges(vranges, num_frames, audio_T)
        else:  # upstream behaviour, kept only to reproduce the growing lip lag
            aranges = aligned_audio_ranges(vranges, F, audio_T)
        model = X0Model(dit)
        for k, ((f0, f1), (a0, a1)) in enumerate(zip(vranges, aranges, strict=True)):
            sel = selected_chunks(k, history, sink_first)
            v_idx, a_idx, v_mask, a_mask, cc_parts = [], [], [], [], []
            cur_off = 0
            off = 0
            for j in sel:
                s, e = vranges[j][0] * tpf, vranges[j][1] * tpf
                v_idx.append(mx.arange(s, e))
                a_idx.append(mx.arange(aranges[j][0], aranges[j][1]))
                if j == k:
                    v_mask.append(vs.denoise_mask[:, s:e])
                    cc_parts.append(cc_full[:, s:e])
                    cur_off = off
                else:
                    v_mask.append(mx.zeros_like(vs.denoise_mask[:, s:e]))
                    cc_parts.append(mx.zeros_like(cc_full[:, s:e]))
                off += e - s
            vi, ai = mx.concatenate(v_idx), mx.concatenate(a_idx)
            nv, na = int(vi.shape[0]), int(ai.shape[0])
            sub_latent = latent[:, vi]
            sub_v = LatentState(latent=sub_latent, clean_latent=sub_latent,
                                denoise_mask=mx.concatenate(v_mask, axis=1),
                                positions=video_positions[:, :nv])  # relative positions
            sub_a_lat = audio.latent[:, ai]
            sub_a = LatentState(latent=sub_a_lat, clean_latent=sub_a_lat,
                                denoise_mask=mx.zeros((1, na, 1), dtype=audio.denoise_mask.dtype),
                                positions=audio_positions[:, :na], frozen=True)
            dit._video_channel_condition_input = mx.concatenate(cc_parts, axis=1)
            try:
                out = denoise_loop(model=model, video_state=sub_v, audio_state=sub_a, video_text_embeds=video_embeds,
                                   audio_text_embeds=audio_embeds, sigmas=sigmas, show_progress=False)
            finally:
                dit._video_channel_condition_input = None
            w = (f1 - f0) * tpf
            new = out.video_latent[:, cur_off:cur_off + w]
            latent = mx.concatenate([latent[:, : f0 * tpf], new.astype(latent.dtype), latent[:, f1 * tpf:]], axis=1)
            mx.eval(latent)
            print(f"chunk {k + 1}/{len(vranges)} frames {f0}-{f1} ctx {sel}", flush=True)

        return self.video_patchifier.unpatchify(latent, (F, H, W))


__all__ = ["AvatarForeverPipeline"]
