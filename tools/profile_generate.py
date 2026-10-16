"""Where does generation time go? Per-stage timing for Generator.generate (see #14).

    python tools/profile_generate.py --device cuda            # the real 8B model (downloads weights, Mimi, watermarker)
    python tools/profile_generate.py --tiny                   # CPU smoke run on a tiny random-weight model

Reports time to first frame (prompt prefill + first frame), per-frame backbone vs. audio-decoder time
(the decoder runs audio_num_codebooks - 1 sequential steps per frame), Mimi decode, watermark + resample,
and the real-time factor (wall time / audio duration). Numbers are wall-clock with device synchronisation.
"""
import argparse
import os
import statistics
import sys
import time

os.environ.setdefault("NO_TORCH_COMPILE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

import generator as gen  # noqa: E402


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class StageTimer:
    def __init__(self, device):
        self.device = device
        self.t = {}

    def wrap(self, obj, attr, name):
        fn = getattr(obj, attr)

        def timed(*a, **k):
            _sync(self.device)
            t0 = time.perf_counter()
            out = fn(*a, **k)
            _sync(self.device)
            self.t.setdefault(name, []).append(time.perf_counter() - t0)
            return out

        setattr(obj, attr, timed)


def build(args):
    if not args.tiny:
        return gen.load_miso_8b(args.device, model_path_or_repo_id=args.model)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
    import test_tiny_model as t

    class _Patch:
        def setattr(self, o, n, v):
            setattr(o, n, v)

    return t.tiny_generator(t.tiny_model(), _Patch())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true", help="tiny random-weight model on CPU (no downloads)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", default=None, help="local checkpoint path or HF repo id")
    ap.add_argument("--text", default="Honestly, I didn't expect it to sound this natural. That's kind of amazing.")
    ap.add_argument("--max-ms", type=float, default=10_000)
    ap.add_argument("--runs", type=int, default=2, help="the first run is reported separately as warm-up")
    args = ap.parse_args()

    g = build(args)
    device = torch.device(g.device)
    m = g._model
    timer = StageTimer(device)
    timer.wrap(m, "generate_frame", "frame")
    timer.wrap(m.backbone, "forward", "backbone")
    timer.wrap(m.decoder, "forward", "decoder")
    timer.wrap(g._audio_tokenizer, "decode", "mimi_decode")
    timer.wrap(gen, "watermark", "watermark")

    for run in range(args.runs):
        timer.t.clear()
        _sync(device)
        t0 = time.perf_counter()
        audio = g.generate(args.text, speaker=0, context=[], max_audio_length_ms=args.max_ms)
        _sync(device)
        total = time.perf_counter() - t0
        frames = timer.t.get("frame", [])
        bb = timer.t.get("backbone", [])
        dec = timer.t.get("decoder", [])
        steps = max(1, m.config.audio_num_codebooks - 1)
        secs = audio.numel() / g.sample_rate
        label = "warm-up" if run == 0 and args.runs > 1 else f"run {run}"
        print(f"\n[{label}] {'tiny random model' if args.tiny else 'Miso TTS 8B'} on {device}; {len(frames)} frames -> {secs:.2f} s audio")
        if frames:
            print(f"  time to first frame (prefill + frame 1): {frames[0] * 1000:9.1f} ms")
            rest = frames[1:] or frames
            print(f"  per frame after the first: mean {statistics.mean(rest) * 1000:7.1f} ms, p95 {sorted(rest)[int(0.95 * (len(rest) - 1))] * 1000:7.1f} ms "
                  f"(a frame is 80 ms of audio)")
            print(f"    backbone per frame: {statistics.mean(bb[1:] or bb) * 1000:7.1f} ms   "
                  f"decoder per frame ({steps} steps): {sum(dec) / len(frames) * 1000:7.1f} ms")
        for k in ("mimi_decode", "watermark"):
            if k in timer.t:
                print(f"  {k:28s}: {sum(timer.t[k]) * 1000:9.1f} ms")
        print(f"  total {total * 1000:,.1f} ms; real-time factor {total / secs if secs else float('inf'):.2f} "
              f"(the whole utterance is returned at once, so this is also the time to first audio)")


if __name__ == "__main__":
    main()
