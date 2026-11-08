#!/usr/bin/env python3
"""Build an MLX q8 pack from the Avatar-Forever PyTorch checkpoint, using only MLX (no mlx-forge install).

usage: af_convert.py SRC.safetensors BASE_PACK_DIR OUT_PACK_DIR

- The transformer keys (model.diffusion_model.*, minus the embeddings connectors) are renamed exactly like
  mlx-forge's ltx_23 recipe (sanitize_transformer_key), prefixed "transformer.", cast to bf16, and the
  transformer_blocks Linear weights are int8-quantized (group 64), like dgrauet/ltx-2.3-mlx-q8.
  The Avatar-Forever extras (video_channel_condition_proj / _gate) stay bf16.
- Every other file of the pack (VAE, audio VAE, vocoder, connector, upscalers) is symlinked from BASE_PACK_DIR
  (stock dgrauet/ltx-2.3-mlx-q8), on the assumption that Avatar-Forever only changes the transformer (its metadata
  names a base checkpoint plus a merged LoRA). Not verified tensor by tensor.
- embedded_config.json / config.json gain the checkpoint's video_channel_condition_* keys.

Written by NexAI Advisors, 2026. The converted weights are a Derivative of LTX-2: use and share them only under the
LTX-2 Community License (LICENSE-LTX-2).
"""
import json
import os
import struct
import sys

import mlx.core as mx

src, base, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return h.pop("__metadata__", {}), h


meta, _ = header(src)
cfg = json.loads(meta["config"])
tcfg = cfg["transformer"]
assert tcfg.get("video_channel_condition_in_channels", 0) > 0, "not an Avatar-Forever checkpoint"


def sanitize(k):  # mlx-forge recipes/ltx_23.py sanitize_transformer_key
    k = k.replace("model.diffusion_model.", "")
    for a, b in ((".to_out.0.", ".to_out."), (".ff.net.0.proj.", ".ff.proj_in."), (".ff.net.2.", ".ff.proj_out."),
                 (".audio_ff.net.0.proj.", ".audio_ff.proj_in."), (".audio_ff.net.2.", ".audio_ff.proj_out."),
                 (".linear_1.", ".linear1."), (".linear_2.", ".linear2.")):
        k = k.replace(a, b)
    return k


w = mx.load(src)
out_w = {}
for k, v in w.items():
    if not k.startswith("model.diffusion_model."):
        continue
    suf = k[len("model.diffusion_model."):]
    if suf.startswith(("video_embeddings_connector.", "audio_embeddings_connector.")):
        continue
    key = "transformer." + sanitize(k)
    v = v.astype(mx.bfloat16)
    bare = key[len("transformer."):]
    if "transformer_blocks" in bare and bare.endswith(".weight") and v.ndim == 2:
        q, s, b = mx.quantize(v, group_size=64, bits=8)
        out_w[key] = q
        out_w[key[: -len("weight")] + "scales"] = s
        out_w[key[: -len("weight")] + "biases"] = b
    else:
        out_w[key] = v

# Compare key sets with the stock pack (minus the Avatar-Forever extras).
_, stock = header(os.path.join(base, "transformer-distilled-1.1.safetensors"))
extra = sorted(set(out_w) - set(stock))
missing = sorted(set(stock) - set(out_w))
print("extra keys:", extra)
print("missing keys:", missing[:10], len(missing))
assert all("video_channel_condition" in k for k in extra), extra
assert not missing, missing[:10]

mx.save_safetensors(os.path.join(out, "transformer-distilled.safetensors"), out_w, metadata={"format": "mlx"})
print("transformer tensors:", len(out_w))

for f in os.listdir(base):
    if f.startswith("transformer-") or f in ("embedded_config.json", "config.json"):
        continue
    dst = os.path.join(out, f)
    if not os.path.exists(dst):
        os.symlink(os.path.join(base, f), dst)
for f in ("embedded_config.json", "config.json"):
    p = os.path.join(base, f)
    if not os.path.exists(p):
        continue
    c = json.load(open(p))
    t = c.get("transformer", c)
    for k in ("video_channel_condition_in_channels", "video_channel_condition_init", "video_channel_condition_mode"):
        t[k] = tcfg[k]
    json.dump(c, open(os.path.join(out, f), "w"), indent=1)
print("pack ready:", out)
