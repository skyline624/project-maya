// src/drama/drama_main.cpp - maya-drama: Drama 3 on a Maya fork.  --check loads, verifies and reports the weights (J1);
// --dump runs the backbone over a .rows file and writes h and the logits of every row (J2, parity against llama.cpp).
//   maya-drama --check --model <backbone.gguf> --mmproj <mmproj.gguf> [--devs 0,1] [--layer-split 20] [--verify]
//   maya-drama --dump  --model <backbone.gguf> --mmproj <mmproj.gguf> --rows in.rows --out out.out [--ctx 4096] [--devs 0,1] [--layer-split 20]
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "strata/drama/drama_forward.hpp"
#include "strata/drama/drama_model.hpp"

namespace {

int usage() {
    std::puts("usage: maya-drama --check --model <backbone.gguf> --mmproj <mmproj.gguf> [--devs 0,1] [--layer-split 20] [--verify]\n"
              "       maya-drama --dump  --model <backbone.gguf> --mmproj <mmproj.gguf> --rows in.rows --out out.out [--ctx 4096] [--devs 0,1] [--layer-split 20]\n"
              "  --check        load both GGUFs onto the GPUs, report VRAM per device, print OK (exit 0) or ERROR <reason> (exit 1)\n"
              "  --dump         run the backbone over the rows of --rows (DRR1: token ids or f32 embedding rows) and write h and the 4097 logits of each row (DRO1) to --out;\n"
              "                 env DRAMA_DUMP_LAYERS=<file> [DRAMA_DUMP_ROWS=n] also writes the input and the residual after every layer, for the first n rows (default 1)\n"
              "  --ctx          rows of KV cache per attention layer (default 4096)\n"
              "  --devs         CUDA devices, one per part (default 0,1)\n"
              "  --layer-split  first layer of each later part, comma separated (default: an even split)\n"
              "  --verify       read back the first and last 4 KiB of every tensor (and of one expert per layer) and compare with the file");
    return 2;
}

std::vector<int> ints(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');) v.push_back(std::atoi(t.c_str()));
    return v;
}

int fail(const std::string& why) {
    std::printf("ERROR %s\n", why.c_str());
    return 1;
}

struct Row {
    int32_t tok = 0;
    std::vector<float> emb; // tok == -1
};

template <class T> bool rd(FILE* f, T& v) { return std::fread(&v, sizeof v, 1, f) == 1; }

// .rows: "DRR1", u32 n_rows, u32 n_embd; per row i32 tok (>= 0 a token id; -1 followed by n_embd f32), little-endian
bool read_rows(const std::string& path, uint32_t n_embd, std::vector<Row>& rows, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    char magic[4];
    uint32_t n = 0, e = 0;
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "DRR1", 4) == 0 && rd(f, n) && rd(f, e);
    if (!ok) err = path + ": not a DRR1 file";
    else if (e != n_embd) { err = path + ": n_embd " + std::to_string(e) + ", the model has " + std::to_string(n_embd); ok = false; }
    for (uint32_t i = 0; ok && i < n; ++i) {
        Row r;
        if (!rd(f, r.tok)) { err = path + ": truncated at row " + std::to_string(i); ok = false; }
        else if (r.tok == -1) {
            r.emb.resize(e);
            if (std::fread(r.emb.data(), 4, e, f) != e) { err = path + ": truncated embedding at row " + std::to_string(i); ok = false; }
        } else if (r.tok < 0) { err = path + ": row " + std::to_string(i) + " has token " + std::to_string(r.tok); ok = false; }
        rows.push_back(std::move(r));
    }
    std::fclose(f);
    return ok;
}

// .out: "DRO1", u32 n_rows, u32 n_embd, u32 n_out; per row n_embd f32 h (after output_norm), then n_out f32 logits
int dump(strata::drama::DramaModel& m, const std::string& rows_path, const std::string& out_path, int ctx) {
    std::vector<Row> rows;
    std::string err;
    if (!read_rows(rows_path, (uint32_t) m.geometry().n_embd, rows, err)) return fail(err);
    strata::drama::DramaRunner r(m);
    if (!r.init(ctx, err)) return fail(err);
    r.reset();
    FILE* f = std::fopen(out_path.c_str(), "wb");
    if (!f) return fail("cannot write " + out_path);
    const uint32_t n = (uint32_t) rows.size(), e = (uint32_t) m.geometry().n_embd, no = (uint32_t) r.n_out();
    std::fwrite("DRO1", 1, 4, f);
    std::fwrite(&n, 4, 1, f);
    std::fwrite(&e, 4, 1, f);
    std::fwrite(&no, 4, 1, f);
    std::vector<float> h(e), lg(no);
    double total = 0, first = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!r.step(rows[i].tok, rows[i].emb.data(), h.data(), lg.data(), err)) { std::fclose(f); return fail("row " + std::to_string(i) + ": " + err); }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        total += ms;
        if (i == 0) first = ms;
        std::fwrite(h.data(), 4, e, f);
        std::fwrite(lg.data(), 4, no, f);
    }
    std::fclose(f);
    for (const auto& l : r.report()) std::printf("INFO %s\n", l.c_str());
    std::printf("INFO %u rows: %.2f ms/row mean, %.2f ms/row without the first (first %.1f ms)\n", n, n ? total / n : 0.0, n > 1 ? (total - first) / (n - 1) : 0.0, first);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    bool check = false, verify = false, do_dump = false;
    int ctx = 4096;
    std::string model, mmproj, rows_path, out_path;
    std::vector<int> devs = {0, 1}, split;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto val = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--check") check = true;
        else if (a == "--dump") do_dump = true;
        else if (a == "--rows") rows_path = val();
        else if (a == "--out") out_path = val();
        else if (a == "--ctx") ctx = std::atoi(val());
        else if (a == "--verify") verify = true;
        else if (a == "--model") model = val();
        else if (a == "--mmproj") mmproj = val();
        else if (a == "--devs") devs = ints(val());
        else if (a == "--layer-split") split = ints(val());
        else return usage();
    }
    if (!check && !do_dump) return usage();
    if (model.empty() || mmproj.empty()) return fail("--check and --dump need --model and --mmproj");
    if (do_dump && (rows_path.empty() || out_path.empty())) return fail("--dump needs --rows and --out");

    strata::drama::DramaModel m;
    std::string err;
    if (!m.load(model, mmproj, devs, split, err)) return fail(err);
    for (const auto& l : m.geometry().describe()) std::printf("INFO %s\n", l.c_str());
    if (verify && !m.verify(err)) return fail(err);
    for (const auto& l : m.report()) std::printf("INFO %s\n", l.c_str());
    if (do_dump) {
        const int rc = dump(m, rows_path, out_path, ctx);
        if (rc == 0) std::puts("OK");
        return rc;
    }
    std::puts("OK");
    return 0;
}
