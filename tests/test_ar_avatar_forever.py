"""Unit tests for the Avatar-Forever chunk scheduling (no model weights needed)."""
from ltx_pipelines_mlx.ar_avatar_forever import (
    aligned_audio_ranges,
    chunk_ranges,
    pixel_time_audio_ranges,
    selected_chunks,
)


def test_chunk_ranges_cover_all_frames():
    assert chunk_ranges(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert chunk_ranges(188, 4)[-1] == (184, 188)


def test_selected_chunks_sink_plus_one_history():
    assert [selected_chunks(k, 1, True) for k in range(4)] == [[0], [0, 1], [0, 1, 2], [0, 2, 3]]
    assert selected_chunks(3, 1, False) == [2, 3]
    assert selected_chunks(3, None, False) == [0, 1, 2, 3]


def _contiguous(ranges, total):
    assert ranges[0][0] == 0 and ranges[-1][1] == total
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))


def test_audio_ranges_are_contiguous():
    v = chunk_ranges(188, 4)
    _contiguous(aligned_audio_ranges(v, 188, 1497), 1497)
    _contiguous(pixel_time_audio_ranges(v, 1497, 1497), 1497)


def test_pixel_split_matches_pixel_time():
    # 1497 frames at 25 fps; audio at 25 tokens/s: chunk k (k >= 1) ends at pixel frame 1 + 8 * (4k + 3).
    v = chunk_ranges(188, 4)
    r = pixel_time_audio_ranges(v, 1497, 1497)
    assert r[0] == (0, 25)          # first chunk: 25 pixel frames, not the proportional 32
    assert r[1][1] - r[1][0] == 32  # later chunks: 32 pixel frames each
    up = aligned_audio_ranges(v, 188, 1497)
    assert up[0][1] - r[0][1] >= 6  # upstream gives chunk 0 about 0.28 s more audio
