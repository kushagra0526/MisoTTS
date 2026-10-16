"""CPU smoke tests on a tiny random-weight model: no checkpoint, Mimi, tokenizer or watermark download needed.

Run: pytest -q tests
"""
import warnings

import pytest
import torch
from torchtune.models import llama3_2

import generator as gen
import models
from models import Model, ModelArgs, sample_topk

NUM_CODEBOOKS = 4
AUDIO_VOCAB = 2051


def _tiny_llama():
    return llama3_2.llama3_2(
        vocab_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        embed_dim=64,
        max_seq_len=2048,
        intermediate_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


models.FLAVORS.setdefault("llama-tiny-test", _tiny_llama)
TINY = ModelArgs(
    backbone_flavor="llama-tiny-test",
    decoder_flavor="llama-tiny-test",
    text_vocab_size=128,
    audio_vocab_size=AUDIO_VOCAB,
    audio_num_codebooks=NUM_CODEBOOKS,
)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    m = Model(TINY)
    with torch.no_grad():
        m.audio_head.normal_(0, 0.02)  # created with torch.empty in Model.__init__
    m.eval()
    m.setup_caches(1)
    return m


class FakeTokenizer:
    def encode(self, text):
        return [1] + [3 + (ord(c) % 100) for c in text] + [2]


class FakeMimi:
    """Mimi stand-in: 1920 samples per frame at 24 kHz (12.5 Hz frames)."""

    sample_rate = 24_000

    def encode(self, audio):
        frames = max(1, audio.shape[-1] // 1920)
        return torch.randint(0, 2048, (1, NUM_CODEBOOKS, frames))

    def decode(self, codes):  # (1, K, T)
        return torch.zeros(1, 1, codes.shape[-1] * 1920)


def tiny_generator(model, monkeypatch):
    g = gen.Generator.__new__(gen.Generator)  # skip __init__: it downloads Mimi, the tokenizer and the watermarker
    g._model = model
    g._text_tokenizer = FakeTokenizer()
    g._frame_size = NUM_CODEBOOKS + 1
    g._audio_tokenizer = FakeMimi()
    g._watermarker = None
    g.sample_rate = 24_000
    g.device = torch.device("cpu")
    monkeypatch.setattr(gen, "watermark", lambda wm, audio, sr, key: (audio, sr))
    return g


def test_generate_frame_shape_and_range():
    m = tiny_model()
    tokens = torch.zeros(1, 6, NUM_CODEBOOKS + 1, dtype=torch.long)
    tokens[..., -1] = torch.arange(6)
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[..., -1] = True
    pos = torch.arange(6).unsqueeze(0)
    with torch.inference_mode():
        out = m.generate_frame(tokens, mask, pos, temperature=0.9, topk=50)
    assert out.shape == (1, NUM_CODEBOOKS)
    assert int(out.min()) >= 0 and int(out.max()) < AUDIO_VOCAB


def test_caches_follow_model_device():
    # regression for #12: KV caches must be created on the model's device
    m = tiny_model()
    dev = next(m.parameters()).device
    for name, buf in m.named_buffers():
        assert buf.device == dev, name


def test_sample_topk_stays_in_topk():
    torch.manual_seed(0)
    logits = torch.randn(4, 100)
    top = torch.topk(logits, 5).indices
    for _ in range(50):
        s = sample_topk(logits, 5, 1.0)
        assert all(int(s[i, 0]) in top[i].tolist() for i in range(4))


def test_generate_is_reproducible_after_reset(monkeypatch):
    g = tiny_generator(tiny_model(), monkeypatch)
    torch.manual_seed(123)
    a = g.generate("hello there", speaker=0, context=[], max_audio_length_ms=400)
    torch.manual_seed(123)
    b = g.generate("hello there", speaker=0, context=[], max_audio_length_ms=400)
    assert torch.equal(a, b)  # caches are reset between calls


def test_generate_with_context_segment(monkeypatch):
    g = tiny_generator(tiny_model(), monkeypatch)
    ctx = [gen.Segment(speaker=1, text="hi", audio=torch.zeros(1920 * 3))]
    audio = g.generate("ok", speaker=0, context=ctx, max_audio_length_ms=240)
    assert audio.ndim == 1 and audio.numel() == 3 * 1920


def test_immediate_eos_returns_empty_audio(monkeypatch):
    m = tiny_model()
    monkeypatch.setattr(m, "generate_frame", lambda *a, **k: torch.zeros(1, NUM_CODEBOOKS, dtype=torch.int))
    g = tiny_generator(m, monkeypatch)
    with pytest.warns(UserWarning, match="end-of-speech"):
        audio = g.generate("hello", speaker=0, context=[], max_audio_length_ms=400)
    assert audio.ndim == 1 and audio.numel() == 0


def test_hitting_max_length_warns(monkeypatch):
    m = tiny_model()
    monkeypatch.setattr(m, "generate_frame", lambda *a, **k: torch.ones(1, NUM_CODEBOOKS, dtype=torch.int))
    g = tiny_generator(m, monkeypatch)
    with pytest.warns(UserWarning, match="max_audio_length_ms"):
        audio = g.generate("hello", speaker=0, context=[], max_audio_length_ms=400)
    assert audio.numel() == 5 * 1920
