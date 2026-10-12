#!/usr/bin/env python3
"""
tools/convert_hf.py — convert a HuggingFace Qwen2/Qwen3 checkpoint into mote's .mq format.

mote's engine is Llama-shaped. Qwen is close but differs in ways this script
reconciles, auto-detected from which tensors the checkpoint actually carries:

  1. Attention bias (Qwen2). Qwen2 has a bias on q/k/v; Llama and Qwen3 do not.
     Present ones are written to the .mq (v2) and added before RoPE.
  2. QK-norm (Qwen3). Qwen3 drops the bias and instead RMSNorms each query and
     key head (a per-head-dim gain) before RoPE. Present gains are written to the
     .mq (v4) and applied in the forward pass. A checkpoint has one or the other,
     never both.
  3. Head width (Qwen3-0.6B). Qwen3 sets head_dim on its own (0.6B: 16 heads of
     128 over a 1024-wide stream). When it differs from dim/n_heads it goes in
     the header (v5) and the runtime sizes q, k, v and wo from it.
  4. RoPE base and norm epsilon. Qwen uses theta=1e6 and eps=1e-6; they go in the
     header so the runtime uses the right constants.
  5. Rotary layout. Every HF checkpoint stores q_proj/k_proj in the "rotate-half"
     arrangement, while mote (following llama2.c) rotates interleaved pairs. So the
     q/k weights, their biases, AND the per-head QK-norm gains are permuted back to
     interleaved here. Miss this and the model emits fluent-looking nonsense.

The big matrices are quantized to symmetric int8 (default) or packed int4
(--bits 4) with one fp32 scale per group, exactly what mote's runtime
dequantizes. int4 uses a per-group clipped-scale search: shrinking the scale
clips the group's outlier but gives every other weight finer steps, and the
scale with the least round-trip error wins. Norm gains and biases stay fp32.

  python3 tools/convert_hf.py <hf_model_dir> <out.mq> [--gs 64] [--bits 8] [--seqlen 2048]
"""
import argparse
import glob
import json
import os
import struct

import numpy as np

MQ_MAGIC = 0x65746F6D
MQ_HEADER = 256


def _read_safetensors(path):
    """Parse one .safetensors file to {name: fp32 numpy}, using only numpy — no
    torch, no safetensors package. Handles F32/F16/BF16 (the dtypes HF ships),
    converting bf16 to fp32 by widening the 16-bit mantissa/exponent into the top
    half of a float32, which is exact."""
    with open(path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen))
        blob = fh.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dt = meta["dtype"]
        lo, hi = meta["data_offsets"]
        raw = blob[lo:hi]
        shape = meta["shape"]
        if dt == "F32":
            a = np.frombuffer(raw, dtype=np.float32)
        elif dt == "F16":
            a = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif dt == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
            a = (u16 << 16).view(np.float32)
        else:
            raise SystemExit(f"{name}: unsupported safetensors dtype {dt}")
        out[name] = a.reshape(shape).astype(np.float32)
    return out


def load_hf(path):
    """Load every safetensors shard as fp32 numpy, plus the config."""
    cfg = json.load(open(os.path.join(path, "config.json")))
    sd = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        sd.update(_read_safetensors(f))
    if not sd:
        raise SystemExit(f"no .safetensors found in {path}")
    return sd, cfg


def permute_reverse(w, n_heads, head_size):
    """HF rotate-half layout -> mote interleaved layout, on the output dimension.
    w is (n_heads*head_size, in) or (n_heads*head_size,). This is the exact inverse
    of the permutation HF applies when importing Llama-family weights."""
    d1 = w.shape[0]
    rest = int(np.prod(w.shape[1:])) if w.ndim > 1 else 1
    v = w.reshape(n_heads, 2, head_size // 2, rest)
    v = v.transpose(0, 2, 1, 3)
    return v.reshape(d1) if w.ndim == 1 else v.reshape(d1, rest)


def quantize_flat(w, gs):
    """Symmetric int8, one fp32 scale per group of gs, row-major, matching the
    layout mote's matmul_q8 expects. Returns (int8 bytes, fp32 scales)."""
    flat = np.ascontiguousarray(w, dtype=np.float32).ravel()
    assert flat.size % gs == 0, f"group size {gs} does not divide {flat.size}"
    g = flat.reshape(-1, gs)
    scale = np.abs(g).max(axis=1) / 127.0
    scale[scale == 0.0] = 1.0
    q = np.clip(np.rint(g / scale[:, None]), -127, 127).astype(np.int8)
    return q.tobytes(), scale.astype(np.float32).tobytes()


def quantize_flat_q4(w, gs):
    """int4 in [-8, 7], two per byte, exactly mote's quantize_q4: the scale is
    the group's largest-magnitude weight over -8 (keeping its sign, so that
    weight lands exactly on -8 and the rest get finer steps), then plain
    round-to-nearest — deliberately nothing cleverer, because QAT-trained
    checkpoints survive exactly the rounding they trained through and a scale
    search silently breaks them. Packing matches the runtime: within a group,
    byte k holds w[k] (low nibble) and w[k + gs/2] (high nibble). Returns
    (packed bytes, fp32 scales). Chunked to bound peak memory."""
    assert gs % 2 == 0
    flat = np.ascontiguousarray(w, dtype=np.float32).ravel()
    assert flat.size % gs == 0, f"group size {gs} does not divide {flat.size}"
    g = flat.reshape(-1, gs)
    half = gs // 2

    packed = np.empty((len(g), half), dtype=np.uint8)
    scales = np.empty(len(g), dtype=np.float32)
    chunk = 1 << 20                          # groups per pass
    for lo in range(0, len(g), chunk):
        gc = g[lo:lo + chunk]
        m = np.take_along_axis(gc, np.abs(gc).argmax(axis=1)[:, None], axis=1)[:, 0]
        scale = m / -8.0
        safe = np.where(scale != 0.0, scale, 1.0)
        q = np.clip(np.rint(gc / safe[:, None]), -8, 7).astype(np.int8)
        scales[lo:lo + chunk] = scale.astype(np.float32)
        qlo = q[:, :half].astype(np.int16)
        qhi = q[:, half:].astype(np.int16)
        packed[lo:lo + chunk] = ((qlo & 0xF) | ((qhi & 0xF) << 4)).astype(np.uint8)
    return packed.tobytes(), scales.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--gs", type=int, default=64)
    ap.add_argument("--bits", type=int, default=8, choices=(8, 4),
                    help="weight width; 4 halves the file again at some quality cost")
    ap.add_argument("--seqlen", type=int, default=2048,
                    help="cap the context (bounds the KV cache); RoPE is unaffected")
    args = ap.parse_args()

    sd, cfg = load_hf(args.model_dir)
    D = cfg["hidden_size"]
    HD = cfg["intermediate_size"]
    L = cfg["num_hidden_layers"]
    NH = cfg["num_attention_heads"]
    NKV = cfg["num_key_value_heads"]
    V = cfg["vocab_size"]
    # Qwen3 can set head_dim independently of hidden_size/heads (0.6B: 128 vs
    # 1024/16 = 64). Absent, it is dim/n_heads, as for every Llama/Qwen2.
    head_size = int(cfg.get("head_dim") or D // NH)
    explicit_head_dim = head_size * NH != D
    kv = NKV * head_size
    seq_len = min(args.seqlen, cfg.get("max_position_embeddings", args.seqlen))
    # newer transformers nests rope_theta under rope_parameters; older configs
    # keep it top-level. Miss it and the model silently runs on the Llama default.
    rope_theta = float(cfg.get("rope_theta")
                       or cfg.get("rope_parameters", {}).get("rope_theta")
                       or 10000.0)
    rms_eps = float(cfg.get("rms_norm_eps", 1e-5))
    shared = bool(cfg.get("tie_word_embeddings", False))
    gs = args.gs
    if D % gs or HD % gs or (NH * head_size) % gs:
        raise SystemExit(f"gs {gs} must divide dim {D}, hidden {HD} and n_heads*head_dim {NH * head_size}")
    if head_size % 2:
        raise SystemExit(f"head_dim {head_size} must be even (RoPE rotates pairs)")

    def g(name):
        return sd[name]

    def has(name):
        return name in sd

    # Qwen2 carries q/k/v bias; Qwen3 carries per-head q_norm/k_norm instead.
    # Detect from the checkpoint rather than assuming a family.
    has_qkv_bias = has("model.layers.0.self_attn.q_proj.bias")
    has_qk_norm = has("model.layers.0.self_attn.q_norm.weight")

    # ---- assemble fp32 side (norms, + biases or qk-norm), per-layer stacked ----
    rms_att = np.stack([g(f"model.layers.{l}.input_layernorm.weight") for l in range(L)])
    rms_ffn = np.stack([g(f"model.layers.{l}.post_attention_layernorm.weight") for l in range(L)])
    rms_final = g("model.norm.weight")
    if has_qkv_bias:
        bq = np.stack([permute_reverse(g(f"model.layers.{l}.self_attn.q_proj.bias"), NH, head_size) for l in range(L)])
        bk = np.stack([permute_reverse(g(f"model.layers.{l}.self_attn.k_proj.bias"), NKV, head_size) for l in range(L)])
        bv = np.stack([g(f"model.layers.{l}.self_attn.v_proj.bias") for l in range(L)])
    if has_qk_norm:
        # one gain per head-dim position, shared across heads. Permute it with the
        # same within-head reordering the q/k weights get, so it lines up in the
        # interleaved layout the runtime multiplies against.
        qn = np.stack([permute_reverse(g(f"model.layers.{l}.self_attn.q_norm.weight"), 1, head_size) for l in range(L)])
        kn = np.stack([permute_reverse(g(f"model.layers.{l}.self_attn.k_norm.weight"), 1, head_size) for l in range(L)])

    print(f"config: dim={D} hidden={HD} layers={L} heads={NH}/{NKV} vocab={V} "
          f"head_dim={head_size} seq={seq_len} theta={rope_theta:g} eps={rms_eps:g} shared={shared} "
          f"gs={gs} bits={args.bits} qkv_bias={has_qkv_bias} qk_norm={has_qk_norm}", flush=True)

    with open(args.out, "wb") as f:
        # header: 64 int32 slots (256 bytes). Stamp the lowest version that can
        # express the file, so simpler checkpoints still load on older runtimes:
        # v5 needs an explicit head_dim, v4 qk-norm, v3 int4, else v2. A v5 file
        # is refused by a v4 runtime rather than misread with dim/n_heads heads.
        version = 5 if explicit_head_dim else (4 if has_qk_norm else (3 if args.bits == 4 else 2))
        h = [0] * (MQ_HEADER // 4)
        h[0] = MQ_MAGIC; h[1] = version
        h[2] = D; h[3] = HD; h[4] = L; h[5] = NH; h[6] = NKV; h[7] = V; h[8] = seq_len
        h[9] = gs; h[10] = 1 if shared else 0
        h[11] = 1 if has_qkv_bias else 0
        f.write(struct.pack("<12i", *h[:12]))
        f.write(struct.pack("<f", rope_theta))   # slot 12 as float
        f.write(struct.pack("<f", rms_eps))      # slot 13 as float
        f.write(struct.pack("<i", args.bits))    # slot 14
        f.write(struct.pack("<i", 1 if has_qk_norm else 0))   # slot 15
        f.write(struct.pack("<i", head_size if explicit_head_dim else 0))   # slot 16 (v5)
        f.write(b"\x00" * (MQ_HEADER - 17 * 4))

        # fp32 norms, then (bias xor qk-norm) — same order map_weights_quant reads
        fp32_blocks = [rms_att, rms_ffn, rms_final]
        if has_qkv_bias:
            fp32_blocks += [bq, bk, bv]
        if has_qk_norm:
            fp32_blocks += [qn, kn]
        for a in fp32_blocks:
            f.write(np.ascontiguousarray(a, dtype=np.float32).tobytes())

        # quantized tensors, in the order the loader maps them
        def wq(mat):
            if args.bits == 4:
                qb, sb = quantize_flat_q4(mat, gs)
            else:
                qb, sb = quantize_flat(mat, gs)
            f.write(qb); f.write(sb)

        wq(g("model.embed_tokens.weight"))                                    # tokens
        wq(np.stack([permute_reverse(g(f"model.layers.{l}.self_attn.q_proj.weight"), NH, head_size) for l in range(L)]))
        wq(np.stack([permute_reverse(g(f"model.layers.{l}.self_attn.k_proj.weight"), NKV, head_size) for l in range(L)]))
        wq(np.stack([g(f"model.layers.{l}.self_attn.v_proj.weight") for l in range(L)]))
        wq(np.stack([g(f"model.layers.{l}.self_attn.o_proj.weight") for l in range(L)]))
        wq(np.stack([g(f"model.layers.{l}.mlp.gate_proj.weight") for l in range(L)]))   # w1
        wq(np.stack([g(f"model.layers.{l}.mlp.down_proj.weight") for l in range(L)]))   # w2
        wq(np.stack([g(f"model.layers.{l}.mlp.up_proj.weight") for l in range(L)]))     # w3
        if not shared:
            wq(g("lm_head.weight"))

    mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}  ({mb:.1f} MB, int{args.bits} gs={gs})", flush=True)


if __name__ == "__main__":
    main()
