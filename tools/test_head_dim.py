#!/usr/bin/env python3
"""
tools/test_head_dim.py — prove the engine runs heads that are not dim/n_heads wide.

Qwen3-0.6B has 16 heads of 128 over a 1024-wide stream, so its query is 2048
wide, twice the residual stream. This builds tiny random Qwen3 models with
transformers, one with head_dim decoupled from dim/n_heads (written as .mq v5)
and a control with the usual width (written as v4, header slot 16 left zero),
converts each with convert_hf.py and checks mote's greedy tokens against the
transformers reference on the same int8 weights.

The reference fake-quantises its weights with the converter's own int8 rule, so
both sides hold identical weights; what is left between them is mote quantising
each matmul input to int8. A tiny random model has small logit margins, so a
step only counts when the reference's top two logits are at least --margin
apart (0.25 against a logit spread of about 2.4; the flips seen sit under 0.1, in
the control as much as in the decoupled model), and every such step must agree
(the prompt is replayed teacher-forced, so one flip cannot cascade).

Needs torch and transformers. Builds nothing itself: run `make run_ids` first.

  python3 tools/test_head_dim.py [--run_ids ./run_ids]
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import convert_hf  # noqa: E402


def fake_q8(model, gs):
    """Round every 2-D weight exactly as convert_hf.quantize_flat does."""
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.ndim != 2:
                continue
            g = p.data.float().reshape(-1, gs)
            s = g.abs().amax(-1, keepdim=True) / 127.0
            s = torch.where(s == 0, torch.ones_like(s), s)
            p.data = (torch.clamp(torch.round(g / s), -127, 127) * s).reshape(p.shape)


def run_case(name, head_dim, run_ids, margin, tmp):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=head_dim,
                      max_position_embeddings=256, rope_theta=1e6, rms_norm_eps=1e-6,
                      tie_word_embeddings=True, initializer_range=0.2, torch_dtype="float32")
    model = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():                    # non-trivial norm gains, so a mis-permuted
        for n, p in model.named_parameters():  # q_norm/k_norm cannot pass by accident
            if p.ndim == 1:
                p.data = 1.0 + 0.3 * torch.randn_like(p)
    d = os.path.join(tmp, name)
    model.save_pretrained(d)
    mq = os.path.join(tmp, f"{name}.mq")
    sys.argv = ["convert_hf.py", d, mq, "--bits", "8", "--gs", "32", "--seqlen", "256"]
    convert_hf.main()
    h = struct.unpack("<17i", open(mq, "rb").read(68))
    fake_q8(model, 32)

    rng = np.random.default_rng(1)
    agree = decisive = 0
    worst = 0.0
    spread = []
    for trial in range(4):
        ids = [int(x) for x in rng.integers(0, 512, size=24)]
        out = subprocess.run([run_ids, mq, "1", ",".join(map(str, ids[:1]))], capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit(f"{name}: run_ids failed: {out.stderr.strip()}")
        # teacher-forced: the next-token argmax after each prefix of the prompt
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids])).logits[0]
        spread.append(float(logits.std()))
        for k in range(1, len(ids) + 1):
            got = subprocess.run([run_ids, mq, "1", ",".join(map(str, ids[:k]))],
                                 capture_output=True, text=True, check=True).stdout.split()
            mote_next = int(got[-1])
            top = torch.topk(logits[k - 1], 2)
            if float(top.values[0] - top.values[1]) < margin:
                continue
            decisive += 1
            if mote_next == int(top.indices[0]):
                agree += 1
            else:
                worst = max(worst, float(top.values[0] - top.values[1]))
    print(f"{name}: header v{h[1]} slot16={h[16]} head_dim={head_dim} "
          f"(dim/n_heads={cfg.hidden_size // cfg.num_attention_heads}): "
          f"{agree}/{decisive} decisive steps agree with transformers "
          f"(logit std {np.mean(spread):.2f}; widest gap mote disagreed on {worst:.3f})", flush=True)
    return h, agree, decisive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_ids", default=os.path.join(os.path.dirname(HERE), "run_ids"))
    ap.add_argument("--margin", type=float, default=0.25)
    a = ap.parse_args()
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for name, hd, want_v, want_slot in [("decoupled", 64, 5, 64), ("control", 32, 4, 0)]:
            h, agree, decisive = run_case(name, hd, a.run_ids, a.margin, tmp)
            if h[1] != want_v or h[16] != want_slot:
                print(f"  FAIL: expected v{want_v} with slot16={want_slot}")
                ok = False
            if decisive < 40 or agree != decisive:
                print(f"  FAIL: {decisive - agree} decisive steps disagree (of {decisive})")
                ok = False
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
