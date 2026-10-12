# mote — a tiny language model, run from scratch in C.
#
#   make          optimized single-thread build   -> ./mote
#   make omp      multi-threaded (needs OpenMP)    -> ./mote
#   make quantize the fp32 -> int8/int4 converter  -> ./quantize
#   make run_ids  drive the engine on token IDs    -> ./run_ids
#   make grammar_test  constrained-decoding tests  -> ./grammar_test
#   make intent_build  compile a command registry    -> ./intent_build
#   make registry      build registry/apartment.txt  -> registry/apartment.{mint,h}
#   make intent_test   adversarial intent matching    -> ./intent_test
#   make debug    warnings + sanitizers            -> ./mote
#   make clean
#
# To import a HuggingFace Qwen2 checkpoint: tools/convert_hf.py writes a .mq the
# engine loads directly (see the tool's header for what it reconciles).

CC      ?= cc
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra
LDLIBS  := -lm
SRC     := $(wildcard src/*.c)
BIN     := mote

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

.PHONY: omp quantize run_ids tok_test qwen_chat registry debug clean
omp: $(SRC)
	$(CC) $(CFLAGS) -fopenmp -o $(BIN) $(SRC) $(LDLIBS)

# the converter reuses the engine's quantization core
quantize: tools/quantize.c src/quant.c src/parallel.c
	$(CC) $(CFLAGS) -o $@ tools/quantize.c src/quant.c src/parallel.c $(LDLIBS)

# run the engine on raw token IDs, no tokenizer — validates a freshly converted
# architecture against its reference model before its tokenizer is ported to C
run_ids: tools/run_ids.c src/error.c src/model.c src/forward.c src/quant.c src/parallel.c
	$(CC) $(CFLAGS) -Isrc -o $@ tools/run_ids.c src/error.c src/model.c src/forward.c src/quant.c src/parallel.c $(LDLIBS)

# byte-level BPE tokenizer (Qwen/GPT-2 family): a diff harness and a full offline
# chat turn (text -> ids -> forward -> ids -> text, entirely in C)
tok_test: tools/tok_test.c src/bpe.c
	$(CC) $(CFLAGS) -Isrc -o $@ tools/tok_test.c src/bpe.c $(LDLIBS)

qwen_chat: tools/qwen_chat.c src/error.c src/model.c src/forward.c src/quant.c src/parallel.c src/bpe.c
	$(CC) $(CFLAGS) -Isrc -o $@ tools/qwen_chat.c src/error.c src/model.c src/forward.c src/quant.c src/parallel.c src/bpe.c $(LDLIBS)

# same chat turn, but through the mote_lib public API (the surface iOS links against)
lib_chat: tools/lib_chat.c $(ENGINE) src/bpe.c src/mote_lib.c src/grammar.c src/intent.c
	$(CC) $(CFLAGS) -Isrc -o $@ tools/lib_chat.c $(ENGINE) src/bpe.c src/mote_lib.c src/grammar.c src/intent.c $(LDLIBS)

# constrained decoding: prove a schema-masked model cannot emit anything the
# schema forbids, however hard the prompt asks it to
grammar_test: tools/grammar_test.c $(ENGINE) src/bpe.c src/mote_lib.c src/grammar.c src/intent.c
	$(CC) $(CFLAGS) -Isrc -o $@ tools/grammar_test.c $(ENGINE) src/bpe.c src/mote_lib.c src/grammar.c src/intent.c $(LDLIBS)

# emit a model as a const C array to link into firmware (weights in flash)
blob2c: tools/blob2c.c
	$(CC) $(CFLAGS) -o $@ tools/blob2c.c

# intent matching: embed a command registry once at build time, then match an
# utterance against it with one forward pass and no generation at all
intent_build: tools/intent_build.c $(ENGINE) src/bpe.c src/mote_lib.c src/grammar.c src/intent.c
	$(CC) $(CFLAGS) -Isrc -o $@ tools/intent_build.c $(ENGINE) src/bpe.c src/mote_lib.c src/grammar.c src/intent.c $(LDLIBS)

intent_test: tools/intent_test.c $(ENGINE) src/bpe.c src/mote_lib.c src/grammar.c src/intent.c
	$(CC) $(CFLAGS) -Isrc -o $@ tools/intent_test.c $(ENGINE) src/bpe.c src/mote_lib.c src/grammar.c src/intent.c $(LDLIBS)

# compile the starter registry, and a second copy with centering off so the test
# can measure what centering buys instead of taking it on faith
MODEL ?= models/default.mq
TOK   ?= models/qwen.mtok
registry: intent_build
	./intent_build       $(MODEL) $(TOK) registry/apartment.txt registry/apartment
	./intent_build --raw $(MODEL) $(TOK) registry/apartment.txt registry/apartment_raw

# measure model quality (perplexity), for comparing fp32 vs quantized
ENGINE := src/error.c src/model.c src/forward.c src/quant.c src/tokenizer.c src/sampler.c src/parallel.c
perplexity: tools/perplexity.c $(ENGINE) src/bpe.c
	$(CC) $(CFLAGS) -o $@ tools/perplexity.c $(ENGINE) src/bpe.c $(LDLIBS)

debug: $(SRC)
	$(CC) -O1 -g -std=c11 -Wall -Wextra -fsanitize=address,undefined \
	      -o $(BIN) $(SRC) $(LDLIBS)

clean:
	rm -f $(BIN) quantize blob2c perplexity run_ids tok_test qwen_chat lib_chat grammar_test \
	      intent_build intent_test
