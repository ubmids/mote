# mote

A tiny language model, run from scratch in C.

`mote` is a complete transformer inference engine in a few hundred lines of
dependency-free C. No PyTorch, no NumPy, no BLAS, no framework. It loads a
pretrained model and generates text, and the whole forward pass, attention,
rotary embeddings, the KV cache, SwiGLU, sampling, is written out by hand so you
can read every step.

The name is the point. A mote is the smallest thing you can still see. The goal
of this project is to shrink a working language model down until it fits somewhere
absurd: a laptop with nothing installed, and eventually a microcontroller.

```
$ ./mote models/stories15M.bin -i "Once upon a time, there was a little robot"

Once upon a time, there was a little robot. The robot liked to crawl. One day,
the robot went to the park. He saw a big tree... At the top, he saw a big party.
The party was for the robot. The robot and the bird were friends. They played
and had fun.

[158 tokens, 109.4 tok/s]
```

## Quickstart

```bash
make                              # builds ./mote, needs only a C compiler
./scripts/download_model.sh       # fetches a 15M-param model + tokenizer (~58MB)
./mote models/stories15M.bin -i "Once upon a time"
```

Or run it quantized, 4x smaller (int8) or 6x smaller (int4) and noticeably faster:

```bash
make quantize                                        # builds the converter
./quantize models/stories15M.bin models/stories15M.mq
./mote models/stories15M.mq -i "Once upon a time"    # int8, same output quality
./quantize models/stories15M.bin models/stories15M_q4.mq 32 4
./mote models/stories15M_q4.mq -i "Once upon a time" # int4, still coherent
```

Options:

```
-z <path>   tokenizer file (default models/tokenizer.bin)
-t <float>  temperature, 0 = greedy/deterministic (default 1.0)
-p <float>  top-p / nucleus sampling (default 0.9, 0 disables)
-n <int>    number of steps (default 256)
-i <string> prompt
-s <int>    rng seed, for reproducible runs
```

Larger models (`42M`, `110M`) are one argument away:
`./scripts/download_model.sh 110M`.

## How it works

The engine is small enough to hold in your head:

| file | what it does |
|---|---|
| `src/model.c` | memory-maps a checkpoint and points the weights at it |
| `src/forward.c` | one token through the network, the core of the whole thing |
| `src/tokenizer.c` | byte-pair encode/decode between text and token ids |
| `src/sampler.c` | greedy, temperature, and top-p sampling |
| `src/grammar.c` | compiles a JSON schema into a per-step mask over the vocabulary |
| `src/quant.c` | int8 and int4 quantization: the pack, dequant, and integer matmuls |
| `src/mote.c` | the CLI and the generation loop |
| `tools/quantize.c` | converts an fp32 checkpoint to the quantized `.mq` format |

The model is a standard Llama-style decoder: pre-norm blocks, each with
grouped-query attention and a SwiGLU feed-forward, RMSNorm throughout, and
rotary position embeddings applied directly to the query and key vectors. A KV
cache keeps each step to attending over history rather than recomputing it.
Weights are memory-mapped, so a model is paged in on demand and startup is
instant.

There is exactly one performance-critical loop, the matrix-vector `matmul`, and
it is the only thing touched by optional multithreading:

```bash
make omp      # OpenMP build, if your compiler has it
```

## Quantization

`mote` also runs quantized weights, at two widths. int8 packs each big weight
matrix into 8-bit integers with one fp32 scale per group of values (symmetric
"Q8_0" quantization). int4 goes further: two weights per byte in [-8, 7], where
the scale carries the sign of the group's largest weight so that weight lands
exactly on -8 and every other weight gets a finer step (the "Q4_0" idea). In
both, the runtime quantizes each activation to int8 on the fly and the matmul
is integer work with the scales folded back in per group. The norm gains stay
fp32.

One binary runs all of it: a checkpoint's first bytes say whether it is fp32 or
`.mq`, and the header says how wide the weights are. And the quality is
measured, not asserted: `make perplexity` scores a model over held-out text
(TinyStories validation data, ~17k tokens). On the 15M model:

| weights | size | perplexity | cost |
|---|---|---|---|
| fp32 | 61 MB | 2.376 | — |
| int8 (gs 32) | 17 MB | 2.379 | +0.1% |
| int4 (gs 32) | 9.5 MB | 2.640 | +11% |

Getting the int4 number down was an exercise in measuring rather than trusting
intuition: a per-group clipped-scale search *lowers weight-space RMS error* and
*worsens* perplexity, because the weights it clips are precisely the ones that
matter. The signed-scale trick above beat every clipping scheme tried; the
small scale search kept in the converter is worth ~0.1% on top. Weight RMS
error and model quality are different objectives, only one of them is the point.

Because quantized weights are a quarter (or an eighth) of the size, quantized
models are also markedly faster to run, memory bandwidth is the bottleneck, not
arithmetic.

## Constrained decoding

A small model asked for a tool call writes valid JSON only because valid JSON is
likely, which is not the same as guaranteed, and the failures are the expensive
kind: a missing brace, an invented key, an enum value nobody defined. So the
sampler takes a mask. A JSON schema is compiled into a byte-level state machine;
at each step the machine is run over every token's raw bytes and the tokens that
could not continue a valid document have their logits pushed out of reach before
the draw. Malformed output stops being unlikely and becomes unreachable.

```c
mote_grammar *g = mote_grammar_compile(m, schema_json, err, sizeof(err));
mote_generate_constrained(m, prompt, &params, g, on_token, NULL);
```

Generation ends when the object closes, so the callback sees the document and
nothing else. If the token budget runs out first, the shortest completion the
schema allows is appended, so there is no truncated tail to handle. Passing no
grammar leaves the sampling path exactly as it was, one branch aside.

What is admitted is what a tool call needs: one flat object, keys in schema
order, values typed string (with optional `enum`, `minLength`, `maxLength`),
number, integer or boolean, plus `required`. A schema the machine cannot enforce
is refused at compile time rather than half-enforced. `make grammar_test` runs
the proof: the state machine against a table of legal and illegal documents,
forced completion from every truncation point of a valid one, and the real engine
under prompts written to break out — asking for prose, for extra keys, for an
enum value the schema does not list, for deliberately broken JSON. Every reply is
checked by a validator written from the schema by hand, sharing no code with the
grammar. Unconstrained, none of those ten replies conform; constrained, all ten
do. The mask costs 0.2 ms per step where the machine admits a couple of bytes
and 3.2 ms in the middle of a free string, against ~14 ms for the forward pass.

## In a browser

The engine also compiles to WebAssembly — same C files, one more host:

```bash
./web/build.sh                    # needs emscripten; emits web/mote.js + .wasm
python3 web/serve.py 8000         # http.server + the two isolation headers
# open http://localhost:8000/web/  (append ?auto to run a benchmark prompt)
```

The whole engine is ~87KB of wasm; the model download is the only heavy part.
The matmuls use hand-written wasm SIMD and a pthread worker pool (pthreads are
web workers; they need the cross-origin-isolation headers `serve.py` sends —
without them the engine still runs, single-threaded). Measured with the int4
Qwen2.5-0.5B (309 MB): 48 tok/s in desktop Chromium and 20 tok/s in iPhone
Safari, up from 13.6 and 6.5 single-threaded. Still shy of the native app, but
a real conversation from a URL with nothing installed.

## Roadmap

`mote` is being built in three milestones, from the general to the absurd. See
[ROADMAP.md](ROADMAP.md).

1. **Inference in pure C** — done. This is what you are running now.
2. **Quantize** the weights, with the math by hand — done at both widths: int8
   (4x smaller, quality intact) and int4 (6x smaller, quality measured above).
3. **Onto a microcontroller** — run it on a chip, not a computer.

## Acknowledgements

The pretrained weights are the **TinyStories** models trained by Andrej Karpathy,
and `mote` reads the compact checkpoint format from his `llama2.c` project. That
format and those models are what make a from-scratch engine testable on day one.
The transformer architecture is Meta AI's Llama. The code here is my own
implementation.

## License

MIT. See [LICENSE](LICENSE).
