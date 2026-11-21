// Reference side of the P1 parity test (docs/plan-drama3-maya.md section 10): feeds rows (token ids or embedding rows) to the
// Drama 3 backbone through llama.cpp and writes the final-norm output `h` and the logits of [eos, semantic 0..4095] per row.
//   drama_ref make-rows --model bb.gguf --mmproj mm.gguf --prompt TEXT [--codes N] [--seed S] [--suffix TEXT] --out x.rows
//   drama_ref run       --model bb.gguf --rows x.rows --out x.out [--gpu] [--threads N]
// Formats (little-endian): .rows = "DRR1" u32 n_rows u32 n_embd, per row i32 tok (tok == -1: n_embd f32 follow);
//                          .out  = "DRO1" u32 n_rows u32 n_embd u32 n_out(4097), per row n_embd f32 h + n_out f32 logits.
#include "llama.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

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

static std::map<std::string, std::string> parse_args(int argc, char ** argv, int from) {
    std::map<std::string, std::string> a;
    for (int i = from; i < argc; i++) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) die("unexpected argument " + k);
        k = k.substr(2);
        if (k == "gpu") { a[k] = "1"; continue; }
        if (i + 1 >= argc) die("missing value for --" + k);
        a[k] = argv[++i];
    }
    return a;
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
    std::vector<Row> rows;
    for (int32_t t : tokenize(vocab, a["prompt"])) rows.push_back({ t, {} });
    const size_t n_prompt = rows.size();

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
            Row r; r.tok = -1; r.embd.resize(d);
            for (size_t j = 0; j < d; j++) r.embd[j] = (e_lm[(size_t) c0 * d + j] + e_cb1[(size_t) c1 * d + j]) * inv_sqrt2;
            rows.push_back(std::move(r));
        }
    }
    size_t n_suffix = 0;
    if (a.count("suffix")) for (int32_t t : tokenize(vocab, a["suffix"])) { rows.push_back({ t, {} }); n_suffix++; }
    write_rows(a["out"], rows, n_embd);
    fprintf(stderr, "wrote %zu rows (%zu prompt tokens, %d frames, %zu suffix tokens) to %s\n", rows.size(), n_prompt, std::max(n_codes, 0), n_suffix, a["out"].c_str());
    llama_model_free(model);
    return 0;
}

static int run(std::map<std::string, std::string> & a) {
    for (const char * k : { "model", "rows", "out" }) if (!a.count(k)) die(std::string("missing --") + k);
    const bool gpu = a.count("gpu") > 0;
    uint32_t n_embd_rows = 0;
    const std::vector<Row> rows = read_rows(a["rows"], n_embd_rows);
    const int n_rows = (int) rows.size();
    if (n_rows == 0) die("no rows");

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
    llama_model * model = llama_model_load_from_file(a["model"].c_str(), mp);
    if (!model) die("cannot load model");
    const uint32_t n_embd = (uint32_t) llama_model_n_embd(model);
    if (n_embd != n_embd_rows) die("n_embd mismatch between model and .rows");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (SEM_BEGIN + N_SEM > llama_vocab_n_tokens(vocab)) die("vocab too small");
    const bool mrope = llama_model_rope_type(model) == LLAMA_ROPE_TYPE_MROPE || llama_model_rope_type(model) == LLAMA_ROPE_TYPE_IMROPE;

    const uint32_t n_batch = 512;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) std::max(4096, n_rows + 64);
    cp.n_batch = cp.n_ubatch = n_batch;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cp.n_threads = cp.n_threads_batch = a.count("threads") ? atoi(a["threads"].c_str()) : 12;
    if (!gpu) { cp.offload_kqv = false; cp.op_offload = false; }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die("cannot create context");
    // as the server does for TTS slots: `h` (result_norm) is returned next to the logits, pooling NONE gives one row per output
    llama_set_embeddings(ctx, true);

    std::ofstream out(a["out"], std::ios::binary);
    if (!out) die("cannot write " + a["out"]);
    const uint32_t hdr[3] = { (uint32_t) n_rows, n_embd, (uint32_t) N_OUT };
    out.write("DRO1", 4); out.write((const char *) hdr, 12);

    std::vector<float> sel(N_OUT);
    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    const auto t0 = std::chrono::steady_clock::now();
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
                const llama_embd e = { r.embd.data(), 1, (size_t) n_embd };
                idx = llama_batch_ext_add_embd(batch, 0, e);
                const llama_pos p4[4] = { p, p, p, p };  // M-RoPE: all sections share the position (set_position_mrope_1d)
                if (idx >= 0) llama_batch_ext_set_pos(batch, idx, mrope ? p4 : &p);
            }
            if (idx != i) die("batch add failed at row " + std::to_string(pos + i));
            llama_batch_ext_set_output_logits(batch, idx, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) die("llama_process failed at row " + std::to_string(pos));
        for (int i = 0; i < n; i++) {
            const float * h = llama_get_embeddings_ith(ctx, i);
            const float * lg = llama_get_logits_ith(ctx, i);
            if (!h || !lg) die("no output for row " + std::to_string(pos + i));
            sel[0] = lg[EOS_ID];
            memcpy(&sel[1], lg + SEM_BEGIN, (size_t) N_SEM * 4);
            out.write((const char *) h, (std::streamsize) n_embd * 4);
            out.write((const char *) sel.data(), (std::streamsize) N_OUT * 4);
        }
        pos += n;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "%d rows in %.1f s: %.2f rows/s (%s)\n", n_rows, s, n_rows / s, gpu ? "GPU" : "CPU");
    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    if (!out) die("write failed");
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) die("usage: drama_ref make-rows|run [options]");
    auto a = parse_args(argc, argv, 2);
    const std::string cmd = argv[1];
    if (cmd == "make-rows") return make_rows(a);
    if (cmd == "run") return run(a);
    die("unknown subcommand " + cmd);
}
