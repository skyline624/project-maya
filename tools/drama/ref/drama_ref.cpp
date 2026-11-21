// Reference side of the P1 parity test (docs/plan-drama3-maya.md section 10): feeds rows (token ids or embedding rows) to the
// Drama 3 backbone through llama.cpp and writes the final-norm output `h` and the logits of [eos, semantic 0..4095] per row.
//   drama_ref make-rows --model bb.gguf --mmproj mm.gguf --prompt TEXT [--codes N] [--seed S] [--suffix TEXT] --out x.rows
//   drama_ref run       --model bb.gguf --rows x.rows --out x.out [--gpu] [--threads N]
//   drama_ref tower     --model bb.gguf --mmproj mm.gguf --rows x.rows --codes x.codes --out x.tower [--gpu]   (P2 reference)
//   drama_ref generate  --model bb.gguf --mmproj mm.gguf --prompt TEXT --seed S [--frames N] --codes-out g.codes --wav-out g.wav
//                       [--rows-out g.rows] [--gpu]                                                             (P3 reference)
//   make-rows also takes --codes-out x.codes.
// Voice cloning (J4): make-rows and generate take --ref-wav ref.wav --ref-text TRANSCRIPT, the target text being --prompt: the prompt is llama.cpp's cloned one
// (reference-bracket-v1, drama3_gen_audio_pipeline::build_reference_prompt): head tokens, one embedding row per frame of the clip (the speaker encoder's output
// rows), tail tokens.  make-rows --codes-out and generate --ref-codes-out write the clip's 10 codes per frame row (rows order); generate --rows-out holds the
// prompt rows then the generated frames' rows; generate --check-prompt captures the embeddings (`model.input_embed`) of the text rows llama.cpp's own
// pipeline decodes and compares them with the token rows of the prompt built here.  A text argument that starts with @ is read from that file.
// Formats (little-endian): .rows = "DRR1" u32 n_rows u32 n_embd, per row i32 tok (tok == -1: n_embd f32 follow);
//                          .out  = "DRO1" u32 n_rows u32 n_embd u32 n_out(4097), per row n_embd f32 h + n_out f32 logits,
//                          .codes = "DRC1" u32 n_frames u32 n_cb(10), per frame i32 c0 (semantic index, not a token id), c1..c9,
//                          .tower = "DRT1" u32 n_rows u32 n_layers u32 n_embd, f32 [n_rows][n_layers][n_embd] = tower_out of each layer.
#include "llama.h"
#include "gguf.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <string>
#include <vector>

static constexpr int N_CB = 10;
static constexpr int32_t EOS_ID = 248046, SEM_BEGIN = 248099, N_SEM = 4096, N_OUT = N_SEM + 1, N_CB_ROWS = 1024;

[[noreturn]] static void die(const std::string & msg) { fprintf(stderr, "drama_ref: %s\n", msg.c_str()); exit(1); }

struct Row { int32_t tok; std::vector<float> embd; };

static void write_rows(const std::string & path, const std::vector<Row> & rows, uint32_t n_embd) {
    std::ofstream f(path, std::ios::binary);
    if (!f) die("cannot write " + path);
    const uint32_t n = (uint32_t) rows.size();
    f.write("DRR1", 4); f.write((const char *) &n, 4); f.write((const char *) &n_embd, 4);
    for (const Row & r : rows) {
        f.write((const char *) &r.tok, 4);
        if (r.tok < 0) f.write((const char *) r.embd.data(), (std::streamsize) n_embd * 4);
    }
    if (!f) die("write failed: " + path);
}

static std::vector<Row> read_rows(const std::string & path, uint32_t & n_embd) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot read " + path);
    char magic[4]; uint32_t n = 0;
    f.read(magic, 4); f.read((char *) &n, 4); f.read((char *) &n_embd, 4);
    if (!f || memcmp(magic, "DRR1", 4) != 0) die("bad .rows header: " + path);
    std::vector<Row> rows(n);
    for (Row & r : rows) {
        f.read((char *) &r.tok, 4);
        if (r.tok < -1) die("bad row tag");
        if (r.tok == -1) { r.embd.resize(n_embd); f.read((char *) r.embd.data(), (std::streamsize) n_embd * 4); }
    }
    if (!f) die("truncated .rows: " + path);
    return rows;
}

static void write_codes(const std::string & path, const std::vector<std::vector<int32_t>> & frames) {
    std::ofstream f(path, std::ios::binary);
    if (!f) die("cannot write " + path);
    const uint32_t hdr[2] = { (uint32_t) frames.size(), (uint32_t) N_CB };
    f.write("DRC1", 4); f.write((const char *) hdr, 8);
    for (const auto & fr : frames) {
        if (fr.size() != (size_t) N_CB) die("frame with the wrong code count");
        f.write((const char *) fr.data(), N_CB * 4);
    }
    if (!f) die("write failed: " + path);
}

static std::vector<std::vector<int32_t>> read_codes(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot read " + path);
    char magic[4]; uint32_t hdr[2] = { 0, 0 };
    f.read(magic, 4); f.read((char *) hdr, 8);
    if (!f || memcmp(magic, "DRC1", 4) != 0 || hdr[1] != (uint32_t) N_CB) die("bad .codes header: " + path);
    std::vector<std::vector<int32_t>> frames(hdr[0], std::vector<int32_t>(N_CB));
    for (auto & fr : frames) f.read((char *) fr.data(), N_CB * 4);
    if (!f) die("truncated .codes: " + path);
    return frames;
}

// n_elems BF16 values of the tensor `name`, starting byte_off bytes into it, as f32. Only that slice is read.
static std::vector<float> read_bf16(const std::string & path, const char * name, size_t tensor_bytes_expected, size_t byte_off, size_t n_elems) {
    gguf_init_params p = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), p);
    if (!g) die("cannot open gguf " + path);
    const int64_t id = gguf_find_tensor(g, name);
    if (id < 0) die(std::string("tensor not found: ") + name);
    if (gguf_get_tensor_type(g, id) != GGML_TYPE_BF16) die(std::string("not BF16: ") + name);
    if (gguf_get_tensor_size(g, id) != tensor_bytes_expected) die(std::string("unexpected size for ") + name);
    const size_t base = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id);
    gguf_free(g);
    std::ifstream f(path, std::ios::binary);
    f.seekg((std::streamoff) (base + byte_off));
    std::vector<uint16_t> raw(n_elems);
    f.read((char *) raw.data(), (std::streamsize) (n_elems * 2));
    if (!f) die(std::string("short read of ") + name);
    std::vector<float> out(n_elems);
    for (size_t i = 0; i < n_elems; i++) { const uint32_t b = (uint32_t) raw[i] << 16; memcpy(&out[i], &b, 4); }
    return out;
}

// <arch>.embedding_length of a GGUF (llama_model_n_embd is 0 for a vocab-only load)
static uint32_t gguf_n_embd(const std::string & path) {
    gguf_init_params p = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), p);
    if (!g) die("cannot open gguf " + path);
    const int64_t ka = gguf_find_key(g, "general.architecture");
    if (ka < 0) die("no general.architecture");
    const int64_t ke = gguf_find_key(g, (std::string(gguf_get_val_str(g, ka)) + ".embedding_length").c_str());
    if (ke < 0) die("no embedding_length");
    const uint32_t n = gguf_get_val_u32(g, ke);
    gguf_free(g);
    return n;
}

static std::vector<int32_t> tokenize(const llama_vocab * vocab, const std::string & text) {
    std::vector<int32_t> ids(text.size() + 16);
    // same flags as mtmd-helper-gen tokenize_segments(): no BOS/special added, special tokens parsed
    int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), ids.data(), (int32_t) ids.size(), false, true);
    if (n < 0) { ids.resize(-n); n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), ids.data(), (int32_t) ids.size(), false, true); }
    if (n < 0) die("tokenization failed");
    ids.resize(n);
    return ids;
}

static std::string slurp(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot read " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static std::map<std::string, std::string> parse_args(int argc, char ** argv, int from) {
    std::map<std::string, std::string> a;
    for (int i = from; i < argc; i++) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) die("unexpected argument " + k);
        k = k.substr(2);
        if (k == "gpu") { a[k] = "1"; continue; }
        if (i + 1 >= argc) die("missing value for --" + k);
        a[k] = argv[++i];
        if (!a[k].empty() && a[k][0] == '@') a[k] = slurp(a[k].substr(1));
    }
    return a;
}

// ---- voice cloning: the reference clip's rows and codes, and the cloned prompt (mtmd-helper-gen.cpp, drama3_gen_audio_pipeline)

// offset of the leading speaker tag <|speaker:N|> after optional whitespace, npos without one
static size_t speaker_tag_pos(const std::string & text) {
    const size_t lead = text.find_first_not_of(" \t\n\r\f\v");
    static const std::string open = "<|speaker:";
    if (lead == std::string::npos || text.compare(lead, open.size(), open) != 0) return std::string::npos;
    size_t i = lead + open.size();
    const size_t digits = i;
    while (i < text.size() && text[i] >= '0' && text[i] <= '9') i++;
    return i > digits && text.compare(i, 2, "|>") == 0 ? lead : std::string::npos;
}

// each segment tokenized alone, as tokenize_segments() does
static void tokenize_segments(const llama_vocab * vocab, const std::vector<std::string> & segs, std::vector<Row> & rows) {
    for (const std::string & seg : segs) for (int32_t t : tokenize(vocab, seg)) rows.push_back({ t, {} });
}

struct RefClip { std::vector<float> rows; std::vector<int32_t> codes; size_t n_frames = 0; };  // rows: n_frames * n_embd; codes: n_frames * N_CB

// the clip through the speaker encoder as encode_speaker() does (rows = the encoder's output), and its codes as mtmd_gen_state_set_ref() keeps them
static RefClip encode_ref(mtmd_context * mctx, mtmd_bitmap * bmp, uint32_t n_embd) {
    const std::string marker = mtmd_get_marker(mctx);
    mtmd_input_text text{ marker.c_str(), marker.size(), false, true };
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bp = bmp;
    if (mtmd_tokenize(mctx, chunks, &text, &bp, 1) != 0) die("tokenize of the reference clip failed");
    const mtmd_input_chunk * audio = nullptr;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks) && !audio; i++) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_AUDIO) audio = c;
    }
    if (!audio) die("no audio chunk");
    if (mtmd_encode_chunk(mctx, audio) != 0) die("speaker encode failed");
    RefClip clip;
    clip.n_frames = mtmd_input_chunk_get_n_tokens(audio);
    const float * e = mtmd_get_output_embd(mctx);
    clip.rows.assign(e, e + clip.n_frames * n_embd);
    mtmd_gen_state * st = mtmd_gen_state_init(mctx, mtmd_gen_state_params_default());
    size_t n_codes = 0;
    if (!st || mtmd_gen_state_set_ref(mctx, st) != 0) die("the encoder gave no reference codes");
    const int32_t * codes = mtmd_gen_state_get_ref_codes(st, &n_codes);
    if (!codes || n_codes != clip.n_frames * N_CB) die("reference codes: " + std::to_string(n_codes) + " for " + std::to_string(clip.n_frames) + " frames");
    clip.codes.assign(codes, codes + n_codes);
    mtmd_gen_state_free(st);
    mtmd_input_chunks_free(chunks);
    return clip;
}

// reference-bracket-v1 with one reference (build_reference_prompt): the speaker tag follows "Text: " directly (default speaker 0), an untagged target reuses it
static std::vector<Row> clone_prompt_rows(const llama_vocab * vocab, std::string text, std::string ref_text, const RefClip & clip, uint32_t n_embd) {
    std::string tag = "<|speaker:0|>";
    const size_t lead = speaker_tag_pos(ref_text);
    if (lead == std::string::npos) {
        ref_text = tag + " " + ref_text;
    } else {
        ref_text = ref_text.substr(lead);
        tag = ref_text.substr(0, ref_text.find("|>") + 2);
    }
    if (speaker_tag_pos(text) == std::string::npos) text = tag + " " + text;
    std::vector<Row> rows;
    tokenize_segments(vocab, { "<|im_start|>system\n", "convert the provided text to speech reference to the following:", "\n\n[Reference 1]\nText: " + ref_text + "\nSpeech: " }, rows);
    for (size_t f = 0; f < clip.n_frames; f++) {
        Row r; r.tok = -1; r.embd.assign(clip.rows.begin() + (long) (f * n_embd), clip.rows.begin() + (long) ((f + 1) * n_embd));
        rows.push_back(std::move(r));
    }
    tokenize_segments(vocab, { "<|im_end|>\n", "<|im_start|>user\n", text, "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>" }, rows);
    return rows;
}

// max abs difference between the clip's rows and (E_lm[c0] + E_cb1[c1]) / sqrt(2) of its codes
static float rows_vs_codes(const std::string & mmproj, uint32_t n_embd, const RefClip & clip) {
    const size_t d = n_embd, tab = d * N_SEM * 2;
    const std::vector<float> e_lm  = read_bf16(mmproj, "a.gen.code.out_embd.weight", tab, 0, d * N_SEM);
    const std::vector<float> e_cb1 = read_bf16(mmproj, "a.gen.code.embd.weight", tab * 10, tab, d * N_SEM);
    float worst = 0.0f;
    for (size_t f = 0; f < clip.n_frames; f++) {
        const int32_t * c = &clip.codes[f * N_CB];
        for (size_t j = 0; j < d; j++) worst = std::max(worst, fabsf(clip.rows[f * d + j] - (e_lm[(size_t) c[0] * d + j] + e_cb1[(size_t) c[1] * d + j]) / sqrtf(2.0f)));
    }
    return worst;
}

static std::vector<std::vector<int32_t>> frames_of(const RefClip & clip) {
    std::vector<std::vector<int32_t>> fr;
    for (size_t f = 0; f < clip.n_frames; f++) fr.emplace_back(clip.codes.begin() + (long) (f * N_CB), clip.codes.begin() + (long) ((f + 1) * N_CB));
    return fr;
}

static int make_rows(std::map<std::string, std::string> & a) {
    for (const char * k : { "model", "mmproj", "prompt", "out" }) if (!a.count(k)) die(std::string("missing --") + k);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(a["model"].c_str(), mp);
    if (!model) die("cannot load model vocab");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const uint32_t n_embd = gguf_n_embd(a["model"]);
    fprintf(stderr, "vocab %d tokens, n_embd %u, add_bos %d\n", llama_vocab_n_tokens(vocab), n_embd, (int) llama_vocab_get_add_bos(vocab));
    // the helper never adds BOS for Drama 3 (add_special = false): the rows start with <|im_start|>
    if (a.count("ref-wav")) {  // llama.cpp's cloned prompt: the clip goes through the speaker encoder (the codec side of the mmproj, no text model)
        if (!a.count("ref-text") || a.count("codes") || a.count("suffix")) die("--ref-wav needs --ref-text, and excludes --codes and --suffix");
        mtmd_context_params p = mtmd_context_params_default();
        p.use_gpu = a.count("gpu") > 0;
        p.n_threads = 12;
        p.warmup = false;
        p.print_timings = false;
        mtmd_context * mctx = mtmd_init_from_file(a["mmproj"].c_str(), nullptr, p);
        if (!mctx || mtmd_gen_audio_get_info(mctx).type != MTMD_GEN_AUDIO_TYPE_DRAMA3 || !mtmd_support_audio(mctx)) die("cannot load a Drama 3 speaker encoder from " + a["mmproj"]);
        mtmd_helper_bitmap_wrapper w = mtmd_helper_bitmap_init_from_file(mctx, a["ref-wav"].c_str(), false, mtmd_helper_init_opt_default());
        if (!w.bitmap) die("cannot read the audio file " + a["ref-wav"]);
        const RefClip clip = encode_ref(mctx, w.bitmap, n_embd);
        const std::vector<Row> rows = clone_prompt_rows(vocab, a["prompt"], a["ref-text"], clip, n_embd);
        write_rows(a["out"], rows, n_embd);
        if (a.count("codes-out")) write_codes(a["codes-out"], frames_of(clip));
        fprintf(stderr, "wrote %zu rows (%zu reference frames, %.2f s of audio) to %s; frame rows vs (E_lm[c0] + E_cb1[c1]) / sqrt(2): max abs diff %.3g\n", rows.size(), clip.n_frames,
                (double) clip.n_frames * 2048.0 / mtmd_gen_audio_get_info(mctx).sample_rate, a["out"].c_str(), rows_vs_codes(a["mmproj"], n_embd, clip));
        mtmd_bitmap_free(w.bitmap);
        mtmd_free(mctx);
        llama_model_free(model);
        return 0;
    }
    std::vector<Row> rows;
    for (int32_t t : tokenize(vocab, a["prompt"])) rows.push_back({ t, {} });
    const size_t n_prompt = rows.size();
    std::vector<std::vector<int32_t>> codes;  // per frame: c0, c1, then c2..c9

    const int n_codes = a.count("codes") ? atoi(a["codes"].c_str()) : 0;
    if (n_codes > 0) {
        const size_t d = n_embd, tab = d * N_SEM * 2;  // BF16 [d, 4096]: semantic band E_lm, then slice 1 of [d, 4096, 10] = E_cb1
        const char * mm = a["mmproj"].c_str();
        const std::vector<float> e_lm  = read_bf16(mm, "a.gen.code.out_embd.weight", tab, 0, d * N_SEM);
        const std::vector<float> e_cb1 = read_bf16(mm, "a.gen.code.embd.weight", tab * 10, tab, d * N_SEM);
        std::mt19937 rng(a.count("seed") ? (uint32_t) atoi(a["seed"].c_str()) : 0u);
        std::uniform_int_distribution<int> d0(1, N_SEM - 1), d1(0, N_CB_ROWS - 1);
        const float inv_sqrt2 = 1.0f / sqrtf(2.0f);  // ggml_scale(1/sqrtf(2)) in drama3-gen.cpp
        for (int i = 0; i < n_codes; i++) {
            const int c0 = d0(rng), c1 = d1(rng);
            codes.push_back({ c0, c1 });
            Row r; r.tok = -1; r.embd.resize(d);
            for (size_t j = 0; j < d; j++) r.embd[j] = (e_lm[(size_t) c0 * d + j] + e_cb1[(size_t) c1 * d + j]) * inv_sqrt2;
            rows.push_back(std::move(r));
        }
        // c2..c9 come after every c0/c1, so the rows (and f1/f2/f3.rows) do not change
        for (auto & c : codes) for (int g = 2; g < N_CB; g++) c.push_back(d1(rng));
    }
    size_t n_suffix = 0;
    if (a.count("suffix")) for (int32_t t : tokenize(vocab, a["suffix"])) { rows.push_back({ t, {} }); n_suffix++; }
    write_rows(a["out"], rows, n_embd);
    if (a.count("codes-out")) write_codes(a["codes-out"], codes);
    fprintf(stderr, "wrote %zu rows (%zu prompt tokens, %d frames, %zu suffix tokens) to %s\n", rows.size(), n_prompt, std::max(n_codes, 0), n_suffix, a["out"].c_str());
    llama_model_free(model);
    return 0;
}

struct Backbone { llama_model * model = nullptr; llama_context * ctx = nullptr; uint32_t n_embd = 0; bool mrope = false; };

// backbone model + context as run() always did; n_rows sizes the context. n_embd_rows != 0 is checked against the model.
static std::vector<float> g_h;   // generate --check-prompt: the token embeddings (`model.input_embed`, the get_rows of token_embd) of every text ubatch, in order
static bool g_h_on = false;
static bool h_cb(ggml_tensor * t, bool ask, void *) {
    if (strcmp(t->name, "model.input_embed") != 0) return !ask;
    if (ask) return g_h_on;
    if (t->type != GGML_TYPE_F32) die("model.input_embed is not f32");
    const size_t at = g_h.size(), nb = ggml_nbytes(t);
    g_h.resize(at + nb / 4);
    ggml_backend_tensor_get(t, g_h.data() + at, 0, nb);
    return true;
}

static Backbone load_backbone(std::map<std::string, std::string> & a, int n_rows, uint32_t n_embd_rows) {
    const bool gpu = a.count("gpu") > 0;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t none[1] = { nullptr };
    if (gpu) {
        mp.n_gpu_layers = 999;
        mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
    } else {
        mp.n_gpu_layers = 0;
        mp.devices = none;  // CPU only, no GPU device is used
    }
    Backbone b;
    b.model = llama_model_load_from_file(a["model"].c_str(), mp);
    if (!b.model) die("cannot load model");
    b.n_embd = (uint32_t) llama_model_n_embd(b.model);
    if (n_embd_rows && b.n_embd != n_embd_rows) die("n_embd mismatch between model and .rows");
    const llama_vocab * vocab = llama_model_get_vocab(b.model);
    if (SEM_BEGIN + N_SEM > llama_vocab_n_tokens(vocab)) die("vocab too small");
    b.mrope = llama_model_rope_type(b.model) == LLAMA_ROPE_TYPE_MROPE || llama_model_rope_type(b.model) == LLAMA_ROPE_TYPE_IMROPE;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) std::max(4096, n_rows + 64);
    cp.n_batch = cp.n_ubatch = 512;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cp.n_threads = cp.n_threads_batch = a.count("threads") ? atoi(a["threads"].c_str()) : 12;
    if (!gpu) { cp.offload_kqv = false; cp.op_offload = false; }
    if (a.count("check-prompt")) cp.cb_eval = h_cb;
    b.ctx = llama_init_from_model(b.model, cp);
    if (!b.ctx) die("cannot create context");
    // as the server does for TTS slots: `h` (result_norm) is returned next to the logits, pooling NONE gives one row per output
    llama_set_embeddings(b.ctx, true);
    return b;
}

// backbone over the rows, per_row(row, h, logits) after each homogeneous sub-batch
template <class F> static void decode_rows(const Backbone & b, const std::vector<Row> & rows, F && per_row) {
    const int n_rows = (int) rows.size();
    const uint32_t n_batch = llama_n_batch(b.ctx);
    llama_batch_ext * batch = llama_batch_ext_init(b.ctx);
    int pos = 0;
    while (pos < n_rows) {
        // consecutive homogeneous sub-batch: tokens only, or embedding rows only, at most n_batch rows
        const bool is_tok = rows[pos].tok >= 0;
        int n = 0;
        while (pos + n < n_rows && n < (int) n_batch && (rows[pos + n].tok >= 0) == is_tok) n++;
        llama_batch_ext_clear(batch);
        for (int i = 0; i < n; i++) {
            const Row & r = rows[pos + i];
            const llama_pos p = pos + i;
            int32_t idx;
            if (is_tok) {
                idx = llama_batch_ext_add_token(batch, 0, r.tok);
                if (idx >= 0) llama_batch_ext_set_pos(batch, idx, &p);  // one position per text token, as batch_set_tokens()
            } else {
                const llama_embd e = { r.embd.data(), 1, (size_t) b.n_embd };
                idx = llama_batch_ext_add_embd(batch, 0, e);
                const llama_pos p4[4] = { p, p, p, p };  // M-RoPE: all sections share the position (set_position_mrope_1d)
                if (idx >= 0) llama_batch_ext_set_pos(batch, idx, b.mrope ? p4 : &p);
            }
            if (idx != i) die("batch add failed at row " + std::to_string(pos + i));
            llama_batch_ext_set_output_logits(batch, idx, true);
        }
        if (llama_process(b.ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) die("llama_process failed at row " + std::to_string(pos));
        for (int i = 0; i < n; i++) {
            const float * h = llama_get_embeddings_ith(b.ctx, i);
            const float * lg = llama_get_logits_ith(b.ctx, i);
            if (!h || !lg) die("no output for row " + std::to_string(pos + i));
            per_row(pos + i, h, lg);
        }
        pos += n;
    }
    llama_batch_ext_free(batch);
}

static int run(std::map<std::string, std::string> & a) {
    for (const char * k : { "model", "rows", "out" }) if (!a.count(k)) die(std::string("missing --") + k);
    const bool gpu = a.count("gpu") > 0;
    uint32_t n_embd_rows = 0;
    const std::vector<Row> rows = read_rows(a["rows"], n_embd_rows);
    const int n_rows = (int) rows.size();
    if (n_rows == 0) die("no rows");
    Backbone b = load_backbone(a, n_rows, n_embd_rows);

    std::ofstream out(a["out"], std::ios::binary);
    if (!out) die("cannot write " + a["out"]);
    const uint32_t hdr[3] = { (uint32_t) n_rows, b.n_embd, (uint32_t) N_OUT };
    out.write("DRO1", 4); out.write((const char *) hdr, 12);

    std::vector<float> sel(N_OUT);
    const auto t0 = std::chrono::steady_clock::now();
    decode_rows(b, rows, [&](int, const float * h, const float * lg) {
        sel[0] = lg[EOS_ID];
        memcpy(&sel[1], lg + SEM_BEGIN, (size_t) N_SEM * 4);
        out.write((const char *) h, (std::streamsize) b.n_embd * 4);
        out.write((const char *) sel.data(), (std::streamsize) N_OUT * 4);
    });
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "%d rows in %.1f s: %.2f rows/s (%s)\n", n_rows, s, n_rows / s, gpu ? "GPU" : "CPU");
    llama_free(b.ctx);
    llama_model_free(b.model);
    if (!out) die("write failed");
    return 0;
}

// ---- tower / generate: tensors of the mtmd graphs captured through the context's cb_eval (what MTMD_DEBUG_GRAPH does in mtmd-cli) ----

struct Capture {
    bool want_count = false;                 // only count the tower rows run (no data requested)
    bool want_tower = false;                 // tower_out-<il> of every layer
    bool want_gen   = false;                 // out_codes (the 10 codes of a frame) and gen_audio_out (next backbone row)
    int  n_count = 0;
    std::vector<std::vector<float>> tower;   // by layer, last tower call
    std::vector<int32_t> codes;              // last tower call
    std::vector<float> embd;
    bool got_codes = false, got_embd = false;
};
static Capture g_cap;

static bool capture_cb(ggml_tensor * t, bool ask, void *) {
    const char * n = t->name;
    const bool is_tower = strncmp(n, "tower_out-", 10) == 0;
    const bool is_codes = strcmp(n, "out_codes") == 0, is_embd = strcmp(n, "gen_audio_out") == 0;
    if (!is_tower && !is_codes && !is_embd) return !ask;  // ask: not wanted; compute: keep going
    if (ask) {
        if (is_tower && g_cap.want_count && atoi(n + 10) == 0) g_cap.n_count++;
        return (is_tower && g_cap.want_tower) || ((is_codes || is_embd) && g_cap.want_gen);
    }
    const size_t nb = ggml_nbytes(t);
    if (is_tower) {
        if (t->type != GGML_TYPE_F32) die("tower_out is not f32");
        const size_t il = (size_t) atoi(n + 10);
        if (g_cap.tower.size() <= il) g_cap.tower.resize(il + 1);
        g_cap.tower[il].resize(nb / 4);
        ggml_backend_tensor_get(t, g_cap.tower[il].data(), 0, nb);
    } else if (is_codes) {
        if (t->type != GGML_TYPE_I32) die("out_codes is not i32");
        g_cap.codes.resize(nb / 4);
        ggml_backend_tensor_get(t, g_cap.codes.data(), 0, nb);
        g_cap.got_codes = true;
    } else {
        if (t->type != GGML_TYPE_F32) die("gen_audio_out is not f32");
        g_cap.embd.resize(nb / 4);
        ggml_backend_tensor_get(t, g_cap.embd.data(), 0, nb);
        g_cap.got_embd = true;
    }
    return true;
}

static mtmd_context * init_mtmd(const std::string & mmproj, const llama_model * model, bool gpu) {
    mtmd_context_params p = mtmd_context_params_default();
    p.use_gpu = gpu;
    p.n_threads = 12;
    p.warmup = false;
    p.print_timings = false;
    p.cb_eval = capture_cb;
    mtmd_context * m = mtmd_init_from_file(mmproj.c_str(), model, p);
    if (!m) die("cannot load mmproj " + mmproj);
    if (mtmd_gen_audio_get_info(m).type != MTMD_GEN_AUDIO_TYPE_DRAMA3) die("mmproj is not a Drama 3 generator");
    return m;
}

// The tower over every backbone row as llama.cpp's prefill does it (drama3_gen_audio_pipeline::run_prompt_tower):
//   row r: input h_r; code0 = c0 of row r+1 if that is a frame row, else -1; the row's own 10 codes are the input frame
//   when r is a frame row; the 10 codes of row r+1 are forced (teacher forcing) when it is a frame row.
// Difference with the pipeline: it skips the tower on the last prompt row (always a text row); here every row is run,
// the last one without a next row (code0 = -1).
static int tower(std::map<std::string, std::string> & a) {
    for (const char * k : { "model", "mmproj", "rows", "codes", "out" }) if (!a.count(k)) die(std::string("missing --") + k);
    const bool gpu = a.count("gpu") > 0;
    uint32_t n_embd_rows = 0;
    const std::vector<Row> rows = read_rows(a["rows"], n_embd_rows);
    const std::vector<std::vector<int32_t>> codes = read_codes(a["codes"]);
    const int n_rows = (int) rows.size();
    size_t n_frame_rows = 0;
    for (const Row & r : rows) n_frame_rows += r.tok < 0;
    if (n_frame_rows != codes.size()) die("the .codes frame count differs from the frame rows of the .rows");
    // index of the frame in .codes for each row (-1: text row)
    std::vector<int> fidx(n_rows, -1);
    for (int r = 0, f = 0; r < n_rows; r++) if (rows[r].tok < 0) fidx[r] = f++;

    Backbone b = load_backbone(a, n_rows, n_embd_rows);
    std::vector<float> h((size_t) n_rows * b.n_embd);
    const auto t0 = std::chrono::steady_clock::now();
    decode_rows(b, rows, [&](int r, const float * hr, const float *) { memcpy(&h[(size_t) r * b.n_embd], hr, (size_t) b.n_embd * 4); });
    fprintf(stderr, "backbone: %d rows in %.1f s\n", n_rows, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    // sanity: frame rows must be the (E_lm[c0] + E_cb1[c1]) / sqrt(2) of their codes
    if (!codes.empty()) {
        const size_t d = b.n_embd, tab = d * N_SEM * 2;
        const std::vector<float> e_lm  = read_bf16(a["mmproj"], "a.gen.code.out_embd.weight", tab, 0, d * N_SEM);
        const std::vector<float> e_cb1 = read_bf16(a["mmproj"], "a.gen.code.embd.weight", tab * 10, tab, d * N_SEM);
        float worst = 0.0f;
        for (int r = 0; r < n_rows; r++) if (fidx[r] >= 0) {
            const std::vector<int32_t> & c = codes[fidx[r]];
            for (size_t j = 0; j < d; j++) worst = std::max(worst, fabsf(rows[r].embd[j] - (e_lm[(size_t) c[0] * d + j] + e_cb1[(size_t) c[1] * d + j]) / sqrtf(2.0f)));
        }
        fprintf(stderr, "frame rows vs their codes: max abs diff %.3g%s\n", worst, worst > 1e-3f ? "  <-- WARNING: .rows and .codes do not match" : "");
    }

    mtmd_context * mctx = init_mtmd(a["mmproj"], b.model, gpu);
    mtmd_gen_state_params sp = mtmd_gen_state_params_default();
    sp.seed = 1;
    mtmd_gen_state * st = mtmd_gen_state_init(mctx, sp);
    if (!st) die("cannot create the gen state");
    g_cap.want_tower = g_cap.want_gen = true;

    std::vector<std::vector<std::vector<float>>> out;  // [row][layer][n_embd]
    size_t n_layers = 0, n_forced_mismatch = 0;
    const auto t1 = std::chrono::steady_clock::now();
    for (int r = 0; r < n_rows; r++) {
        const bool cur = fidx[r] >= 0, next = r + 1 < n_rows && fidx[r + 1] >= 0;
        std::vector<int32_t> frame;
        int32_t code0 = -1;
        if (next) {
            frame.assign(N_CB, -1);  // [input frame, -1 for a text row | forced next frame]
            if (cur) frame = codes[fidx[r]];
            frame.insert(frame.end(), codes[fidx[r + 1]].begin(), codes[fidx[r + 1]].end());
            code0 = codes[fidx[r + 1]][0];
        } else if (cur) {
            frame = codes[fidx[r]];
        }
        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
        inp.code0 = code0;
        inp.embd = &h[(size_t) r * b.n_embd];
        inp.codes = frame.empty() ? nullptr : frame.data();
        inp.n_codes = frame.size();
        mtmd_gen_out o{};
        g_cap.tower.clear(); g_cap.got_codes = false;
        if (mtmd_gen_audio_process(mctx, st, &inp, &o) != 0) die("tower failed at row " + std::to_string(r));
        // llama.cpp prunes the output of the last layer from the graph of a forced row (only its K/V are needed): written as NaN
        if (g_cap.tower.empty()) die("tower_out not captured at row " + std::to_string(r));
        n_layers = std::max(n_layers, g_cap.tower.size());
        if (next && g_cap.got_codes && (g_cap.codes.size() != (size_t) N_CB ||
                !std::equal(codes[fidx[r + 1]].begin(), codes[fidx[r + 1]].end(), g_cap.codes.begin()))) n_forced_mismatch++;
        out.push_back(g_cap.tower);
    }
    const size_t n_embd_t = out[0][0].size();
    size_t n_missing = 0;
    for (auto & row : out) {
        row.resize(n_layers);
        for (auto & l : row) if (l.empty()) { l.assign(n_embd_t, NAN); n_missing++; }
    }
    fprintf(stderr, "tower: %d rows in %.1f s (%s), %zu layers (%zu layer outputs pruned by llama.cpp, written as NaN); forced codes echoed differently on %zu rows\n",
            n_rows, std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count(), gpu ? "GPU" : "CPU", n_layers, n_missing, n_forced_mismatch);

    std::ofstream f(a["out"], std::ios::binary);
    if (!f) die("cannot write " + a["out"]);
    const uint32_t hdr[3] = { (uint32_t) n_rows, (uint32_t) n_layers, (uint32_t) n_embd_t };
    f.write("DRT1", 4); f.write((const char *) hdr, 12);
    for (const auto & row : out) for (const auto & l : row) f.write((const char *) l.data(), (std::streamsize) l.size() * 4);
    if (!f) die("write failed");
    mtmd_gen_state_free(st);
    mtmd_free(mctx);
    llama_free(b.ctx);
    llama_model_free(b.model);
    return 0;
}

static std::vector<std::string> prompt_segments(std::string text) {  // drama3_gen_audio_pipeline::prompt_segments (no reference)
    const size_t lead = text.find_first_not_of(" \t\n\r\f\v");
    bool tagged = false;
    if (lead != std::string::npos && text.compare(lead, 10, "<|speaker:") == 0) {
        size_t i = lead + 10;
        const size_t digits = i;
        while (i < text.size() && text[i] >= '0' && text[i] <= '9') i++;
        tagged = i > digits && text.compare(i, 2, "|>") == 0;
    }
    if (!tagged) text = "<|speaker:0|> " + text;
    return { "<|im_start|>system\n", "convert the provided text to speech", "<|im_end|>\n",
             "<|im_start|>user\n", text, "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>" };
}

// A real generation through llama.cpp's own pipeline (mtmd_helper_gen_audio, the code path of llama-tts). The frames' codes and
// next-row embeddings are not exposed by the API: they are read from out_codes and gen_audio_out with the cb_eval.
static int generate(std::map<std::string, std::string> & a) {
    for (const char * k : { "model", "mmproj", "prompt", "seed", "codes-out", "wav-out" }) if (!a.count(k)) die(std::string("missing --") + k);
    const bool gpu = a.count("gpu") > 0, clone = a.count("ref-wav") > 0;
    if (clone && !a.count("ref-text")) die("--ref-wav needs --ref-text");
    Backbone b = load_backbone(a, 4096, 0);
    mtmd_context * mctx = init_mtmd(a["mmproj"], b.model, gpu);
    RefClip clip;
    mtmd_helper_bitmap_wrapper wav{};
    if (clone) {  // my own encode of the clip first: its rows and codes, since the pipeline keeps its codes to itself
        wav = mtmd_helper_bitmap_init_from_file(mctx, a["ref-wav"].c_str(), false, mtmd_helper_init_opt_default());
        if (!wav.bitmap) die("cannot read the audio file " + a["ref-wav"]);
        clip = encode_ref(mctx, wav.bitmap, b.n_embd);
        fprintf(stderr, "reference: %zu frames; frame rows vs (E_lm[c0] + E_cb1[c1]) / sqrt(2): max abs diff %.3g\n", clip.n_frames, rows_vs_codes(a["mmproj"], b.n_embd, clip));
    }
    mtmd_helper::gen_audio gen;
    gen.init(b.ctx, mctx);
    if (!gen.valid()) die("the context cannot run the audio generation");
    mtmd_helper_gen_audio_inp inp = mtmd_helper_gen_audio_inp_default();
    inp.seq_id = 0;
    inp.prompt = a["prompt"].c_str();
    inp.prompt_len = a["prompt"].size();
    inp.top_k = 40;    // llama-tts defaults; the drama3 pipeline ignores both
    inp.top_p = 0.95f;
    inp.seed = (uint32_t) atoll(a["seed"].c_str());
    inp.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    if (clone) {
        inp.speaker_ref = wav.bitmap;
        inp.speaker_ref_text = a["ref-text"].c_str();
    }
    if (gen.set_input(&inp) != 0) die("set_input failed");
    if (clone) {  // the pipeline encoded the clip again in set_input: its rows are still the context's output buffer
        const float * pe = mtmd_get_output_embd(mctx);
        const size_t n = clip.rows.size();
        float worst = 0.0f;
        for (size_t i = 0; i < n; i++) worst = std::max(worst, fabsf(pe[i] - clip.rows[i]));
        fprintf(stderr, "the pipeline's reference rows vs the encode above: max abs diff %.3g\n", worst);
        clip.rows.assign(pe, pe + n);
    }
    g_h_on = a.count("check-prompt") > 0;

    g_cap.want_count = true;  // one tower row per prompt row but the last
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        const int32_t ret = gen.step_prompt(llama_n_batch(b.ctx));
        if (ret < 0) die("prompt processing failed");
        if (ret == 0) break;
    }
    g_cap.want_count = false;
    g_h_on = false;
    const int n_prompt_tower = g_cap.n_count;
    const auto t1 = std::chrono::steady_clock::now();

    std::vector<Row> rows;
    const llama_vocab * vocab = llama_model_get_vocab(b.model);
    if (clone) rows = clone_prompt_rows(vocab, a["prompt"], a["ref-text"], clip, b.n_embd);
    else for (const std::string & seg : prompt_segments(a["prompt"])) for (int32_t t : tokenize(vocab, seg)) rows.push_back({ t, {} });
    if ((int) rows.size() - 1 != n_prompt_tower)
        die("tokenization mismatch: " + std::to_string(rows.size()) + " prompt tokens here, tower ran on " + std::to_string(n_prompt_tower) + " rows");
    const size_t n_prompt = rows.size();

    g_cap.want_gen = true;
    const int max_new = a.count("frames") ? atoi(a["frames"].c_str()) : gen.max_frames(0);
    std::vector<std::vector<int32_t>> frames;
    int n_frames = 0;
    bool stop = false;
    while (!stop && n_frames < max_new) {
        const float * h_next = nullptr;
        g_cap.got_codes = g_cap.got_embd = false;
        if (gen.step_gen(LLAMA_TOKEN_NULL, nullptr, &h_next, &stop) != 0) die("step_gen failed at frame " + std::to_string(n_frames));
        if (!h_next) break;  // stopped without generating a frame
        if (!g_cap.got_codes || !g_cap.got_embd || g_cap.codes.size() != (size_t) N_CB || g_cap.embd.size() != b.n_embd)
            die("out_codes / gen_audio_out not captured at frame " + std::to_string(n_frames));
        frames.push_back(g_cap.codes);
        Row r; r.tok = -1; r.embd = g_cap.embd;
        rows.push_back(std::move(r));
        n_frames++;
    }
    g_cap.want_gen = false;
    const auto t2 = std::chrono::steady_clock::now();

    int32_t sample_rate = 0;
    const char * data = nullptr;
    size_t data_len = 0;
    int64_t n_samples = 0;
    if (gen.get_output(&sample_rate, &data, &data_len, &n_samples) != 0) die("get_output failed");
    std::ofstream w(a["wav-out"], std::ios::binary);
    if (!w) die("cannot write " + a["wav-out"]);
    w.write(data, (std::streamsize) data_len);
    w.close();
    write_codes(a["codes-out"], frames);
    if (clone && a.count("ref-codes-out")) write_codes(a["ref-codes-out"], frames_of(clip));
    if (a.count("check-prompt")) {  // the rows the pipeline fed the backbone vs the prompt built here: text rows by their token embedding, speech rows as they are
        const size_t d = b.n_embd, n_tok = [&] { size_t n = 0; for (size_t r = 0; r < n_prompt; r++) n += rows[r].tok >= 0; return n; }();
        const size_t got = g_h.size() / d;
        const bool all = false;  // embedding batches feed the graph a leaf tensor the callback never sees: the reference rows are checked by the count and make-rows' own check
        fprintf(stderr, "check-prompt: llama.cpp's pipeline embedded %zu text rows (%zu text + %zu reference rows in the prompt built here)\n", got, n_tok, n_prompt - n_tok);
        if (got != n_tok) die("check-prompt: unexpected number of captured rows");
        const size_t n_vocab = (size_t) llama_vocab_n_tokens(llama_model_get_vocab(b.model));
        double worst = 0; size_t bad = 0, k = 0;
        for (size_t r = 0; r < n_prompt; r++) {
            if (rows[r].tok < 0 && !all) continue;
            const std::vector<float> e = rows[r].tok >= 0 ? read_bf16(a["model"], "token_embd.weight", n_vocab * d * 2, (size_t) rows[r].tok * d * 2, d) : rows[r].embd;
            double m = 0;
            for (size_t j = 0; j < d; j++) m = std::max(m, (double) fabsf(e[j] - g_h[k * d + j]));
            worst = std::max(worst, m);
            bad += m > 1e-6;
            k++;
        }
        fprintf(stderr, "check-prompt: %zu rows compared, max abs difference %.3g, %zu rows differ: %s\n", k, worst, bad, bad ? "MISMATCH" : "IDENTICAL");
    }
    if (a.count("rows-out")) write_rows(a["rows-out"], rows, b.n_embd);
    const auto s = [](auto x, auto y) { return std::chrono::duration<double>(y - x).count(); };
    fprintf(stderr, "prompt %zu rows (%.1f s), %d frames (%.1f s, %s), %s: %.2f s of audio at %d Hz (%lld samples), %zu bytes of wav\n",
            n_prompt, s(t0, t1), n_frames, s(t1, t2), stop ? "stopped by EOS" : "frame limit", gpu ? "GPU" : "CPU",
            sample_rate > 0 ? (double) n_samples / sample_rate : 0.0, sample_rate, (long long) n_samples, data_len);
    gen.ctx.reset();
    if (clone) mtmd_bitmap_free(wav.bitmap);
    mtmd_free(mctx);
    llama_free(b.ctx);
    llama_model_free(b.model);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) die("usage: drama_ref make-rows|run|tower|generate [options]");
    auto a = parse_args(argc, argv, 2);
    const std::string cmd = argv[1];
    if (cmd == "make-rows") return make_rows(a);
    if (cmd == "run") return run(a);
    if (cmd == "tower") return tower(a);
    if (cmd == "generate") return generate(a);
    die("unknown subcommand " + cmd);
}
