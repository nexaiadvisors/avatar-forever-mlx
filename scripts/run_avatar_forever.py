#!/usr/bin/env python3
"""Run an Avatar-Forever MLX pack on one audio file.

usage: run_avatar_forever.py --model DIR --image KEY.jpg --audio A.wav --out OUT.mp4 [--frames N] [--prompt P]
       [--height 512 --width 768 --fps 25 --chunk 4 --history 1 --seed 42 --low-ram]
"""
import argparse

from ltx_pipelines_mlx.ar_avatar_forever import AvatarForeverPipeline

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--image", required=True)
p.add_argument("--audio", required=True)
p.add_argument("--out", required=True)
p.add_argument("--frames", type=int, default=161)
p.add_argument("--prompt", default="A person talks naturally to the camera with clear lip movements, steady framing, photorealistic.")
p.add_argument("--height", type=int, default=512)
p.add_argument("--width", type=int, default=768)
p.add_argument("--fps", type=float, default=25.0)
p.add_argument("--chunk", type=int, default=4)
p.add_argument("--history", type=int, default=1)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--low-ram", action="store_true")
a = p.parse_args()

pipe = AvatarForeverPipeline(a.model, low_memory=True, low_ram_streaming=a.low_ram)
pipe.generate_and_save(a.prompt, a.out, audio_path=a.audio, height=a.height, width=a.width, num_frames=a.frames,
                       frame_rate=a.fps, seed=a.seed, image=a.image, chunk=a.chunk, history=a.history)
print("saved", a.out)
