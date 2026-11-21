// src/drama/drama_main.cpp - maya-drama: Drama 3 on a Maya fork.  --check loads, verifies and reports the weights (J1);
// --dump runs the backbone over a .rows file and writes h and the logits of every row (J2, parity against llama.cpp);
// --generate runs backbone + codebook tower + sampling into a .codes file (J3), with voice cloning and a per-voice prefix cache (J4); --tower-dump and --replay
// are the P2 and P3 parity tests.
//   maya-drama --check --model <backbone.gguf> --mmproj <mmproj.gguf> [--devs 0,1] [--layer-split 20] [--verify]
//   maya-drama --dump  --model <backbone.gguf> --mmproj <mmproj.gguf> --rows in.rows --out out.out [--ctx 4096] [--devs 0,1] [--layer-split 20]
//   maya-drama --generate --model .. --mmproj .. --rows prompt.rows [--codes ref.codes] --codes-out g.codes [--max-frames 1500] [--seed 1] [--sampling official|llama]
//   maya-drama --generate --model .. --mmproj .. --texts list.txt [--repeat K] --prefix-rows N --codes ref.codes --codes-out g%d.codes [--check-prefix]
//   (J5, every mode but --check/--tower-dump) [--expert-vram-mib N] [--vram-reserve-mib N] [--expert-profile f] [--expert-profile-save f] [--expert-prefetch 0|1]
//   maya-drama --tower-dump --model .. --mmproj .. --rows f.rows [--codes f.codes] --h-from f.out --out f.tower
//   maya-drama --replay --model .. --mmproj .. --rows g.rows --codes g.codes [--ref-codes ref.codes] [--sampling llama]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "drama_sampling.hpp"
#include "strata/drama/drama_forward.hpp"
#include "strata/drama/drama_model.hpp"
#include "strata/drama/drama_serve.hpp"
#include "strata/drama/drama_tower.hpp"

namespace {

using namespace strata::drama;

int usage() {
    std::puts("usage: maya-drama --check --model <backbone.gguf> --mmproj <mmproj.gguf> [--devs 0,1] [--layer-split 20] [--verify]\n"
              "       maya-drama --dump  --model <backbone.gguf> --mmproj <mmproj.gguf> --rows in.rows --out out.out [--ctx 4096] [--devs 0,1] [--layer-split 20]\n"
              "       maya-drama --generate --model .. --mmproj .. --rows prompt.rows [--codes ref.codes] --codes-out g.codes [--max-frames 1500] [--seed 1] [--sampling official|llama]\n"
              "       maya-drama --generate --model .. --mmproj .. --texts list.txt [--repeat K] --prefix-rows N --codes ref.codes --codes-out g%d.codes [--check-prefix]\n"
              "       maya-drama --tower-dump --model .. --mmproj .. --rows f.rows [--codes f.codes] --h-from f.out --out f.tower\n"
              "       maya-drama --replay --model .. --mmproj .. --rows g.rows --codes g.codes [--ref-codes ref.codes] [--sampling llama]\n"
              "  --check        load both GGUFs onto the GPUs, report VRAM per device, print OK (exit 0) or ERROR <reason> (exit 1)\n"
              "  --dump         run the backbone over the rows of --rows (DRR1: token ids or f32 embedding rows) and write h and the 4097 logits of each row (DRO1) to --out;\n"
              "                 env DRAMA_DUMP_LAYERS=<file> [DRAMA_DUMP_ROWS=n] also writes the input and the residual after every layer, for the first n rows (default 1)\n"
              "  --generate     prompt rows (DRR1, ending with <|voice|>) through backbone and tower, then sample frames until eos or --max-frames;\n"
              "                 writes the frames to --codes-out (DRC1: u32 n_frames, u32 10, i32 frame-major; c0 0..4095, c1..c9 0..1023).\n"
              "                 Voice cloning: the prompt's embedding rows are the reference frames, --codes (DRC1) holds their 10 codes each: the tower takes them as input frames\n"
              "                 and teacher-forces them, as llama.cpp's prefill does.  --texts: a file listing one .rows prompt per line instead of --rows; --repeat K generates K of them\n"
              "                 (cycling), --codes-out then holds a %d.  --prefix-rows N: the first N rows (the voice, identical in every prompt) are prefilled once, saved on the\n"
              "                 devices and restored for each later prompt; --check-prefix redoes each later prompt from a fresh prefill and compares the first frames' logits bit for bit\n"
              "  --tower-dump   P2: the tower over every row, h taken from the DRO1 file --h-from (the reference's), the codes of the reference frames from --codes\n"
              "                 (DRC1, one frame per embedding row of --rows); writes DRT1: u32 n_rows, u32 13, u32 2048, f32 tower_out per row and layer\n"
              "  --replay       P3: teacher-forced replay of a recorded generation (--rows: the prompt, optionally followed by its frame rows; --codes: its frames; --ref-codes: the\n"
              "                 codes of the prompt's reference frames, when it clones a voice); prints the share of the recorded main tokens and head codes the fork could have drawn\n"
              "  --sampling     official (SGLang's rules, default) or llama (llama.cpp's), see docs/plan-drama3-maya.md 9.4\n"
              "  --ctx          rows of KV cache per attention layer (default 4096; --generate: prompt + max frames)\n"
              "  --devs         CUDA devices, one per part (default 0,1)\n"
              "  --layer-split  first layer of each later part, comma separated (default: an even split)\n"
              "  --expert-vram-mib N       experts kept in VRAM per device (default: all that fit after the other allocations and --vram-reserve-mib); the rest stays in pinned RAM\n"
              "  --vram-reserve-mib N      VRAM left free per device when --expert-vram-mib is not given (default 1536)\n"
              "  --expert-profile f        usage counters of an earlier run: the initial fill takes the hottest experts, and the LFU starts from them\n"
              "  --expert-profile-save f   write the usage counters of this run (reads the router's choice back at every layer, even when everything is resident)\n"
              "  --expert-prefetch 0|1     copy the next layer's predicted experts ahead of time (default 1)\n"
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

ExpertOptions g_experts; // J5 options, set by main
bool g_verify = false;

// the pool of a runner, built after every other allocation on the devices
bool setup_experts(DramaRunner& bb, std::string& err) {
    if (!bb.init_experts(g_experts, err)) return false;
    for (const auto& l : bb.pool_report()) std::printf("INFO %s\n", l.c_str());
    return !g_verify || bb.verify_experts(err);
}

// the pool's counters over a stretch of work that took `ms` (0: unknown)
void print_pool(const char* what, int text, const PoolStats& s, double ms) {
    const uint64_t n = s.hits + s.misses;
    const double rows = (double) std::max<uint64_t>(1, s.steps), mib = (double) s.bytes / (1 << 20);
    std::printf("INFO text %d %s: experts hit %llu miss %llu (hit rate %.2f %%), prefetch %llu copies %llu used %llu skipped; %.1f MiB moved = %.2f MiB/row; read-back wait %.3f ms/row%s", text, what,
                (unsigned long long) s.hits, (unsigned long long) s.misses, n ? 100.0 * (double) s.hits / (double) n : 100.0, (unsigned long long) s.pf_issued, (unsigned long long) s.pf_useful,
                (unsigned long long) s.pf_skipped, mib, mib / rows, s.sync_ms / rows, ms > 0 ? "" : "\n");
    if (ms > 0) std::printf("; %.2f GB/s average over the %.0f ms\n", (double) s.bytes / 1e9 / (ms * 1e-3), ms);
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

// .codes: "DRC1", u32 n_frames, u32 n_cb (10), i32 frame-major
bool read_codes(const std::string& path, std::vector<int32_t>& codes, uint32_t& n_frames, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    char magic[4];
    uint32_t cb = 0;
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "DRC1", 4) == 0 && rd(f, n_frames) && rd(f, cb);
    if (!ok) err = path + ": not a DRC1 file";
    else if (cb != (uint32_t) kCodebooks) { err = path + ": " + std::to_string(cb) + " codes per frame, expected 10"; ok = false; }
    else {
        codes.resize((size_t) n_frames * cb);
        if (std::fread(codes.data(), 4, codes.size(), f) != codes.size()) { err = path + ": truncated"; ok = false; }
    }
    std::fclose(f);
    return ok;
}

bool write_codes(const std::string& path, const std::vector<int32_t>& codes, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { err = "cannot write " + path; return false; }
    const uint32_t n = (uint32_t) (codes.size() / kCodebooks), cb = kCodebooks;
    std::fwrite("DRC1", 1, 4, f);
    std::fwrite(&n, 4, 1, f);
    std::fwrite(&cb, 4, 1, f);
    std::fwrite(codes.data(), 4, codes.size(), f);
    std::fclose(f);
    return true;
}

// .out: "DRO1", u32 n_rows, u32 n_embd, u32 n_out; per row n_embd f32 h (after output_norm), then n_out f32 logits
int dump(strata::drama::DramaModel& m, const std::string& rows_path, const std::string& out_path, int ctx) {
    std::vector<Row> rows;
    std::string err;
    if (!read_rows(rows_path, (uint32_t) m.geometry().n_embd, rows, err)) return fail(err);
    strata::drama::DramaRunner r(m);
    if (!r.init(ctx, err) || !setup_experts(r, err)) return fail(err);
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

double ms_since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }

struct GenOpts {
    std::string rows, codes, ref_codes, codes_out, h_from, out, texts;
    int ctx = 0, max_frames = 1500, prefix_rows = 0, repeat = 0;
    uint64_t seed = 1;
    bool llama = false, check_prefix = false;
};

// a prompt: its rows, and the codes of its reference frames (the embedding rows, one block: voice cloning) that the tower takes as input and teacher-forces
struct Prompt {
    std::vector<Row> rows;
    std::vector<int32_t> ref_codes; // frame-major, one frame per embedding row, in row order
    long ref_begin = 0;
    bool speech(long r) const { return r >= ref_begin && (size_t) (r - ref_begin) < ref_codes.size() / kCodebooks; }
    const int32_t* frame(long r) const { return ref_codes.data() + (size_t) (r - ref_begin) * kCodebooks; }
};

// reads the codes of the embedding rows among the first n_prompt rows of p (none: codes_path must be empty)
bool bind_ref(Prompt& p, size_t n_prompt, const std::string& codes_path, std::string& err) {
    long first = -1;
    size_t n = 0;
    for (size_t r = 0; r < n_prompt; ++r)
        if (p.rows[r].tok < 0) { if (first < 0) first = (long) r; ++n; }
    for (size_t r = 0; r < n; ++r)
        if (p.rows[(size_t) first + r].tok >= 0) { err = "the embedding rows of the prompt are not one block"; return false; }
    if (n == 0) {
        if (codes_path.empty()) return true;
        err = "the prompt has no embedding rows but " + codes_path + " was given";
        return false;
    }
    if (codes_path.empty()) { err = "the prompt has " + std::to_string(n) + " embedding rows: their frames are needed (--codes, or --ref-codes for --replay)"; return false; }
    uint32_t nf = 0;
    if (!read_codes(codes_path, p.ref_codes, nf, err)) return false;
    if (nf != n) { err = codes_path + " has " + std::to_string(nf) + " frames, the prompt " + std::to_string(n) + " embedding rows"; return false; }
    p.ref_begin = first;
    return true;
}

// rows [from, to) of a prompt of n_prompt rows through the backbone and the tower, as the reference's prefill does it (run_prompt_tower): the tower runs on every
// row but the last, with the reference frame of a speech row as input and the codes of a next speech row forced.  h and lg end up as the last row's.
bool prefill(DramaRunner& bb, TowerRunner& tw, const Prompt& p, size_t from, size_t to, size_t n_prompt, std::vector<float>& h, std::vector<float>& lg, std::string& err) {
    for (size_t r = from; r < to; ++r) {
        const Row& row = p.rows[r];
        if (!bb.step(row.tok, row.tok < 0 ? row.emb.data() : nullptr, h.data(), lg.data(), err)) { err = "row " + std::to_string(r) + ": " + err; return false; }
        if (r + 1 < n_prompt) {
            TowerStep t;
            t.h = h.data();
            if (p.speech((long) r)) t.frame_in = p.frame((long) r);
            if (p.speech((long) r + 1)) {
                t.code0 = p.frame((long) r + 1)[0];
                t.forced = p.frame((long) r + 1) + 1;
            }
            if (!tw.step(t, err)) { err = "tower row " + std::to_string(r) + ": " + err; return false; }
        }
    }
    return true;
}

TowerSampling tower_sampling(const GenOpts& o) {
    TowerSampling s;
    s.llama = o.llama;
    s.seed = o.seed;
    return s;
}

constexpr int kTraceFrames = 3; // the first frames whose logits the prefix-cache check compares

struct Trace {
    std::vector<std::vector<float>> lg, h; // before each of the first frames
    std::vector<int32_t> codes;
};

struct GenStats {
    size_t n = 0;
    double t_bb = 0, t_tw = 0, t_total = 0;
    int stop = 0; // 0 max frames, 1 eos, 2 context
};

// frames until eos, max_frames or the end of the context, from the state a prefill left (h, lg: the last prompt row's)
bool gen_frames(DramaRunner& bb, TowerRunner& tw, std::vector<float>& h, std::vector<float>& lg, const GenOpts& o, int ctx, int max_frames, std::vector<int32_t>& codes,
                GenStats& st, Trace* trace, std::string& err) {
    MainSampler ms(o.llama ? SamplingMode::Llama : SamplingMode::Official, o.seed);
    std::vector<float> next(h.size());
    int32_t frame[kCodebooks];
    for (int f = 0; f < max_frames; ++f) {
        const auto t0 = std::chrono::steady_clock::now();
        if (trace && f < kTraceFrames) { trace->lg.push_back(lg); trace->h.push_back(h); }
        const int pick = ms.sample(lg.data(), (uint32_t) f);
        if (pick == 0) { st.stop = 1; break; }
        if (bb.pos() >= ctx) { st.stop = 2; break; }
        ms.push(pick);
        TowerStep t;
        t.h = h.data();
        t.code0 = pick - 1;
        t.frame_in = f ? frame : nullptr;
        t.ctr = (uint32_t) f;
        t.codes_out = frame;
        t.next_emb = next.data();
        const auto t1 = std::chrono::steady_clock::now();
        if (!tw.step(t, err)) { err = "frame " + std::to_string(f) + ": tower: " + err; return false; }
        const double d_tw = ms_since(t1);
        for (int i = 0; i < kCodebooks; ++i) codes.push_back(i == 0 ? frame[0] : std::min<int32_t>(frame[i], kResidualCodes - 1)); // llama.cpp clamps at the codec
        if (trace && f < kTraceFrames) trace->codes.insert(trace->codes.end(), codes.end() - kCodebooks, codes.end());
        const auto t2 = std::chrono::steady_clock::now();
        if (!bb.step(-1, next.data(), h.data(), lg.data(), err)) { err = "frame " + std::to_string(f) + ": backbone: " + err; return false; }
        st.t_bb += ms_since(t2);
        st.t_tw += d_tw;
        st.t_total += ms_since(t0);
        ++st.n;
    }
    return true;
}

std::string out_path(const std::string& pattern, int i) { // a printf pattern with one %d (the text's index), or a plain path
    char b[1024];
    if (pattern.find("%d") == std::string::npos) return pattern;
    std::snprintf(b, sizeof b, pattern.c_str(), i);
    return b;
}

// --generate: one or several prompts (--texts: a file listing .rows files, one per line, --repeat K of them, cycling) through backbone and tower.  With --prefix-rows N
// the first N rows (the voice: system prompt and reference, identical in every prompt) are prefilled once, saved on the devices, and restored for the later prompts.
int generate(DramaModel& m, const GenOpts& o) {
    std::string err;
    const int E = m.geometry().n_embd;
    std::vector<std::string> paths;
    if (o.texts.empty()) paths.push_back(o.rows);
    else {
        std::ifstream f(o.texts);
        if (!f) return fail("cannot open " + o.texts);
        for (std::string l; std::getline(f, l);) {
            while (!l.empty() && (l.back() == '\r' || l.back() == ' ')) l.pop_back();
            if (!l.empty()) paths.push_back(l);
        }
        if (paths.empty()) return fail(o.texts + " lists no .rows file");
    }
    const int K = o.repeat > 0 ? o.repeat : (int) paths.size();
    const size_t N = (size_t) o.prefix_rows;
    if (K > 1 && o.codes_out.find("%d") == std::string::npos) return fail("--codes-out needs a %d for several texts");
    std::vector<Prompt> P(paths.size());
    size_t longest = 0;
    for (size_t i = 0; i < paths.size(); ++i) {
        if (!read_rows(paths[i], (uint32_t) E, P[i].rows, err) || !bind_ref(P[i], P[i].rows.size(), o.codes, err)) return fail(err);
        if (P[i].rows.empty()) return fail(paths[i] + " is empty");
        longest = std::max(longest, P[i].rows.size());
        if (N == 0) continue;
        if (N >= P[i].rows.size() || P[i].speech((long) N)) return fail(paths[i] + ": --prefix-rows " + std::to_string(N) + " must end before the first text row after the reference");
        for (size_t r = 0; r < N; ++r) {
            const Row &a = P[0].rows[r], &b = P[i].rows[r];
            if (a.tok != b.tok || (a.tok < 0 && std::memcmp(a.emb.data(), b.emb.data(), (size_t) E * 4) != 0)) return fail(paths[i] + ": row " + std::to_string(r) + " differs from the first prompt's: no common prefix of " + std::to_string(N) + " rows");
        }
        if (P[i].ref_codes != P[0].ref_codes) return fail(paths[i] + ": reference codes differ from the first prompt's");
    }
    const int ctx = o.ctx > 0 ? o.ctx : (int) longest + o.max_frames + 8;
    DramaRunner bb(m);
    TowerRunner tw(m);
    if (!bb.init(ctx, err) || !tw.init(ctx, tower_sampling(o), err) || !setup_experts(bb, err)) return fail(err);
    for (const auto& l : bb.report()) std::printf("INFO %s\n", l.c_str());
    for (const auto& l : tw.report()) std::printf("INFO %s\n", l.c_str());
    std::vector<float> h((size_t) E), lg((size_t) bb.n_out());
    DeviceSnapshot snap_bb, snap_tw;
    bool all_same = true;
    for (int i = 0; i < K; ++i) {
        const Prompt& p = P[(size_t) i % P.size()];
        const size_t n_prompt = p.rows.size();
        double t_prefix = 0, t_save = 0, t_restore = 0;
        size_t from = 0;
        auto t0 = std::chrono::steady_clock::now();
        if (N > 0 && i > 0) {
            if (!bb.restore(snap_bb, err) || !tw.restore(snap_tw, err)) return fail(err);
            t_restore = ms_since(t0);
            from = N;
        } else {
            bb.reset();
            tw.reset();
            if (N > 0) {
                if (!prefill(bb, tw, p, 0, N, n_prompt, h, lg, err)) return fail(err);
                t_prefix = ms_since(t0);
                t0 = std::chrono::steady_clock::now();
                if (!bb.save(snap_bb, err) || !tw.save(snap_tw, err)) return fail(err);
                t_save = ms_since(t0);
                from = N;
            }
        }
        t0 = std::chrono::steady_clock::now();
        if (!prefill(bb, tw, p, from, n_prompt, n_prompt, h, lg, err)) return fail(err);
        const double t_tail = ms_since(t0);
        if (N == 0) std::printf("INFO text %d: prefill %zu rows in %.0f ms\n", i, n_prompt, t_tail);
        else if (i == 0) std::printf("INFO text %d: prefill %zu rows = prefix %zu rows %.0f ms (+ save %.0f ms, %.1f MiB on the devices) + the other %zu rows %.0f ms\n", i, n_prompt, N, t_prefix, t_save,
                                     (double) (snap_bb.bytes() + snap_tw.bytes()) / (1 << 20), n_prompt - N, t_tail);
        else std::printf("INFO text %d: prefill %zu rows = prefix restored in %.0f ms (instead of %zu rows) + the other %zu rows %.0f ms\n", i, n_prompt, t_restore, N, n_prompt - N, t_tail);

        if (bb.has_pool()) print_pool("prefill", i, bb.expert_stats(true), 0);
        std::vector<int32_t> codes;
        GenStats st;
        Trace tr;
        if (!gen_frames(bb, tw, h, lg, o, ctx, o.max_frames, codes, st, o.check_prefix && N > 0 && i > 0 ? &tr : nullptr, err)) return fail(err);
        if (bb.has_pool()) print_pool("generation", i, bb.expert_stats(true), st.t_total);
        const std::string path = out_path(o.codes_out, i);
        if (!write_codes(path, codes, err)) return fail(err);
        const double frame_s = 2048.0 / 44100.0, per = st.n ? st.t_total / (double) st.n : 0.0;
        std::printf("INFO text %d: %zu frames (%.2f s of audio), stopped by %s; per frame: backbone %.2f ms, tower %.2f ms, sampling+other %.2f ms, total %.2f ms; RTF %.3f; wrote %s\n", i, st.n,
                    (double) st.n * frame_s, st.stop == 1 ? "eos" : st.stop == 2 ? "the context" : "--max-frames", st.n ? st.t_bb / (double) st.n : 0.0, st.n ? st.t_tw / (double) st.n : 0.0,
                    st.n ? (st.t_total - st.t_bb - st.t_tw) / (double) st.n : 0.0, per, per / (frame_s * 1000.0), path.c_str());

        if (o.check_prefix && N > 0 && i > 0) { // the same text after a fresh full prefill: the first frames' logits must be bit for bit those of the restored prefix
            bb.reset();
            tw.reset();
            const auto tf = std::chrono::steady_clock::now();
            if (!prefill(bb, tw, p, 0, n_prompt, n_prompt, h, lg, err)) return fail(err);
            const double t_full = ms_since(tf);
            std::vector<int32_t> c2;
            GenStats s2;
            Trace t2;
            if (!gen_frames(bb, tw, h, lg, o, ctx, kTraceFrames, c2, s2, &t2, err)) return fail(err);
            bool same = t2.lg.size() == tr.lg.size() && t2.codes == tr.codes;
            double dmax = 0;
            for (size_t f = 0; same && f < tr.lg.size(); ++f) {
                same = std::memcmp(tr.lg[f].data(), t2.lg[f].data(), tr.lg[f].size() * 4) == 0 && std::memcmp(tr.h[f].data(), t2.h[f].data(), tr.h[f].size() * 4) == 0;
                for (size_t j = 0; j < tr.lg[f].size(); ++j) dmax = std::max(dmax, (double) std::fabs(tr.lg[f][j] - t2.lg[f][j]));
            }
            std::printf("CHECK prefix text %d: the first %zu frames after a restored prefix vs a fresh full prefill (%.0f ms): logits, h and codes bit-identical %s (max abs logit difference %.3g)\n", i,
                        t2.lg.size(), t_full, same ? "yes" : "NO", dmax);
            all_same = all_same && same;
        }
    }
    if (!bb.save_expert_profile(err)) return fail(err);
    for (const auto& l : bb.pool_report()) std::printf("INFO %s\n", l.c_str());
    return all_same ? 0 : fail("the prefix cache changed the logits");
}

// DRO1 file: h of every row
bool read_out(const std::string& path, uint32_t n_embd, std::vector<float>& h, uint32_t& n_rows, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    char magic[4];
    uint32_t e = 0, no = 0;
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "DRO1", 4) == 0 && rd(f, n_rows) && rd(f, e) && rd(f, no);
    if (!ok) err = path + ": not a DRO1 file";
    else if (e != n_embd) { err = path + ": n_embd mismatch"; ok = false; }
    else {
        h.resize((size_t) n_rows * e);
        std::vector<float> skip(no);
        for (uint32_t i = 0; ok && i < n_rows; ++i)
            if (std::fread(h.data() + (size_t) i * e, 4, e, f) != e || std::fread(skip.data(), 4, no, f) != no) { err = path + ": truncated at row " + std::to_string(i); ok = false; }
    }
    std::fclose(f);
    return ok;
}

// P2: the tower over every row (the last one without a next row) as llama.cpp's prefill does it (run_prompt_tower), h from the reference
int tower_dump(DramaModel& m, const GenOpts& o) {
    std::vector<Row> rows;
    std::string err;
    const uint32_t E = (uint32_t) m.geometry().n_embd;
    if (!read_rows(o.rows, E, rows, err)) return fail(err);
    std::vector<float> h;
    uint32_t n_h = 0;
    if (!read_out(o.h_from, E, h, n_h, err)) return fail(err);
    if (n_h != rows.size()) return fail("--h-from has " + std::to_string(n_h) + " rows, --rows " + std::to_string(rows.size()));
    std::vector<int32_t> codes;
    uint32_t n_frames = 0;
    if (!o.codes.empty() && !read_codes(o.codes, codes, n_frames, err)) return fail(err);
    const long ref_begin = [&] { for (size_t r = 0; r < rows.size(); ++r) if (rows[r].tok < 0) return (long) r; return (long) rows.size(); }();
    size_t n_speech = 0;
    for (const auto& r : rows) n_speech += r.tok < 0;
    if (n_speech != n_frames) return fail(std::to_string(n_speech) + " embedding rows but " + std::to_string(n_frames) + " frames in --codes");
    const auto speech = [&](long r) { return r >= 0 && (size_t) r < rows.size() && rows[(size_t) r].tok < 0; };
    const auto frame_of = [&](long r) { return codes.data() + (size_t) (r - ref_begin) * kCodebooks; };
    TowerRunner tw(m);
    const int ctx = o.ctx > 0 ? o.ctx : (int) rows.size() + 8;
    if (!tw.init(ctx, tower_sampling(o), err)) return fail(err);
    tw.reset();
    for (const auto& l : tw.report()) std::printf("INFO %s\n", l.c_str());
    const uint32_t n_out = (uint32_t) rows.size(), n_layer = (uint32_t) m.geometry().t_layers;
    FILE* f = std::fopen(o.out.c_str(), "wb");
    if (!f) return fail("cannot write " + o.out);
    std::fwrite("DRT1", 1, 4, f);
    std::fwrite(&n_out, 4, 1, f);
    std::fwrite(&n_layer, 4, 1, f);
    std::fwrite(&E, 4, 1, f);
    std::vector<float> lo((size_t) n_layer * E);
    double total = 0;
    for (long r = 0; r < (long) n_out; ++r) {
        const bool cur = speech(r), next = speech(r + 1);
        TowerStep t;
        t.h = h.data() + (size_t) r * E;
        t.frame_in = cur ? frame_of(r) : nullptr;
        if (next) {
            t.code0 = frame_of(r + 1)[0];
            t.forced = frame_of(r + 1) + 1;
        }
        t.layer_out = lo.data();
        const auto t0 = std::chrono::steady_clock::now();
        if (!tw.step(t, err)) { std::fclose(f); return fail("row " + std::to_string(r) + ": " + err); }
        total += ms_since(t0);
        std::fwrite(lo.data(), 4, lo.size(), f);
    }
    std::fclose(f);
    std::printf("INFO %u rows through the tower, %.2f ms/row\n", n_out, n_out ? total / n_out : 0.0);
    return 0;
}

// P3: teacher-forced replay of a generation recorded from llama.cpp.  For every frame: is the recorded main token one the fork could draw (temperature 1,
// top_p 0.9, no top_k; or, since a repeat is redrawn, RAS's top_k 30 / top_p 0.95)?  Is each recorded head code in the head's nucleus (top_k 30, top_p 0.9)?
int replay(DramaModel& m, const GenOpts& o) {
    Prompt pr;
    std::vector<Row>& rows = pr.rows;
    std::string err;
    const int E = m.geometry().n_embd;
    if (!read_rows(o.rows, (uint32_t) E, rows, err)) return fail(err);
    std::vector<int32_t> codes;
    uint32_t F = 0;
    if (!read_codes(o.codes, codes, F, err)) return fail(err);
    size_t n_prompt = rows.size();
    while (n_prompt > 0 && rows[n_prompt - 1].tok < 0) --n_prompt; // trailing embedding rows are the frames' own rows: only checked
    if (n_prompt == 0) return fail("--rows holds no token row");
    if (!bind_ref(pr, n_prompt, o.ref_codes, err)) return fail(err); // a cloned prompt: its reference frames sit among the first n_prompt rows
    const size_t n_gen_rows = rows.size() - n_prompt;
    const int ctx = o.ctx > 0 ? o.ctx : (int) (n_prompt + F) + 8;
    DramaRunner bb(m);
    TowerRunner tw(m);
    if (!bb.init(ctx, err) || !tw.init(ctx, tower_sampling(o), err) || !setup_experts(bb, err)) return fail(err);
    bb.reset();
    tw.reset();
    for (const auto& l : tw.report()) std::printf("INFO %s\n", l.c_str());
    std::vector<float> h((size_t) E), lg((size_t) bb.n_out()), next((size_t) E), hl((size_t) 9 * 4096);
    if (!prefill(bb, tw, pr, 0, n_prompt, n_prompt, h, lg, err)) return fail(err);
    const SamplingMode mode = o.llama ? SamplingMode::Llama : SamplingMode::Official;
    MainSampler ms(mode, o.seed);
    const SampleSpec hs = head_spec(mode, 1.f, 30, 0.9f);
    std::vector<int> ids(4096);
    for (int i = 0; i < 4096; ++i) ids[(size_t) i] = i;
    long main_in = 0, main_ras = 0, main_any = 0, eos_p_n = 0;
    long head_in[9] = {0};
    double emb_diff = 0, mass = 0;
    for (uint32_t f = 0; f < F; ++f) {
        const int32_t* c = codes.data() + (size_t) f * kCodebooks;
        const int pos = c[0] + 1; // position in the 4097 logits
        const auto nuc0 = ms.nucleus(lg.data(), false);
        const bool a = std::find(nuc0.begin(), nuc0.end(), pos) != nuc0.end();
        const auto ras = ms.nucleus(lg.data(), true);
        const bool b = std::find(ras.begin(), ras.end(), pos) != ras.end();
        main_in += a;
        main_ras += b;
        main_any += a || b;
        { // probability of the recorded token
            const Sorted s = ms.candidates(lg.data(), 1.f);
            for (size_t i = 0; i < s.ids.size(); ++i)
                if (s.ids[i] == pos) { mass += s.w[i] / s.z; ++eos_p_n; break; }
        }
        ms.push(pos);
        TowerStep t;
        t.h = h.data();
        t.code0 = c[0];
        t.frame_in = f ? codes.data() + (size_t) (f - 1) * kCodebooks : nullptr;
        t.forced = c + 1;
        t.ctr = f;
        t.next_emb = next.data();
        t.head_logits = hl.data();
        if (!tw.step(t, err)) return fail("frame " + std::to_string(f) + ": tower: " + err);
        for (int ih = 0; ih < 9; ++ih) {
            const std::vector<float> v(hl.begin() + (size_t) ih * 4096, hl.begin() + (size_t) (ih + 1) * 4096);
            const auto nuc = nucleus_ids(sort_candidates(v, ids, 1.f), hs);
            head_in[ih] += std::find(nuc.begin(), nuc.end(), c[ih + 1]) != nuc.end();
        }
        if (f < n_gen_rows && rows[n_prompt + f].tok < 0)
            for (int i = 0; i < E; ++i) emb_diff = std::max(emb_diff, (double) std::fabs(rows[n_prompt + f].emb[(size_t) i] - next[(size_t) i]));
        if (f + 1 < F && !bb.step(-1, next.data(), h.data(), lg.data(), err)) return fail("frame " + std::to_string(f) + ": backbone: " + err);
    }
    const double d = F ? 100.0 / F : 0.0;
    std::printf("INFO replay: %u frames, sampling %s\n", F, o.llama ? "llama" : "official");
    std::printf("P3 main token: in the temperature-1 top_p-0.9 nucleus %.2f %% (%ld/%u); in it or in RAS's nucleus %.2f %%; in RAS's alone %.2f %%; mean probability of the recorded token %.3f\n",
                main_in * d, main_in, F, main_any * d, main_ras * d, eos_p_n ? mass / (double) eos_p_n : 0.0);
    for (int ih = 0; ih < 9; ++ih) std::printf("P3 head %d (code %d): in the nucleus %.2f %% (%ld/%u)\n", ih, ih + 1, head_in[ih] * d, head_in[ih], F);
    if (n_gen_rows) std::printf("INFO frame rows of --rows vs the fork's next embedding: max abs difference %.3g (%zu rows)\n", emb_diff, n_gen_rows);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (strata::drama::serve_requested(argc, argv)) return strata::drama::serve_main(argc, argv); // J6: the service, its own options
    bool check = false, verify = false, do_dump = false, do_gen = false, do_tdump = false, do_replay = false;
    int ctx = 4096;
    GenOpts go;
    std::string model, mmproj, rows_path, out_path;
    std::vector<int> devs = {0, 1}, split;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto val = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--check") check = true;
        else if (a == "--dump") do_dump = true;
        else if (a == "--generate") do_gen = true;
        else if (a == "--tower-dump") do_tdump = true;
        else if (a == "--replay") do_replay = true;
        else if (a == "--rows") rows_path = val();
        else if (a == "--out") out_path = val();
        else if (a == "--codes") go.codes = val();
        else if (a == "--codes-out") go.codes_out = val();
        else if (a == "--ref-codes") go.ref_codes = val();
        else if (a == "--texts") go.texts = val();
        else if (a == "--repeat") go.repeat = std::atoi(val());
        else if (a == "--prefix-rows") go.prefix_rows = std::atoi(val());
        else if (a == "--check-prefix") go.check_prefix = true;
        else if (a == "--h-from") go.h_from = val();
        else if (a == "--max-frames") go.max_frames = std::atoi(val());
        else if (a == "--seed") go.seed = std::strtoull(val(), nullptr, 10);
        else if (a == "--sampling") {
            const std::string v = val();
            if (v != "official" && v != "llama") return fail("--sampling is official or llama");
            go.llama = v == "llama";
        }
        else if (a == "--ctx") ctx = std::atoi(val());
        else if (a == "--verify") verify = g_verify = true;
        else if (a == "--expert-vram-mib") g_experts.vram_mib = std::atoll(val());
        else if (a == "--vram-reserve-mib") g_experts.reserve_mib = std::atoll(val());
        else if (a == "--expert-profile") g_experts.profile = val();
        else if (a == "--expert-profile-save") g_experts.profile_save = val();
        else if (a == "--expert-prefetch") g_experts.prefetch = std::atoi(val()) != 0;
        else if (a == "--model") model = val();
        else if (a == "--mmproj") mmproj = val();
        else if (a == "--devs") devs = ints(val());
        else if (a == "--layer-split") split = ints(val());
        else return usage();
    }
    if (!check && !do_dump && !do_gen && !do_tdump && !do_replay) return usage();
    if (model.empty() || mmproj.empty()) return fail("every mode needs --model and --mmproj");
    if (do_dump && (rows_path.empty() || out_path.empty())) return fail("--dump needs --rows and --out");
    if (do_gen && ((rows_path.empty() == go.texts.empty()) || go.codes_out.empty())) return fail("--generate needs --codes-out and one of --rows or --texts");
    if (do_gen && go.prefix_rows < 0) return fail("--prefix-rows must be >= 0");
    if (do_tdump && (rows_path.empty() || go.h_from.empty() || out_path.empty())) return fail("--tower-dump needs --rows, --h-from and --out");
    if (do_replay && (rows_path.empty() || go.codes.empty())) return fail("--replay needs --rows and --codes");
    go.rows = rows_path;
    go.out = out_path;
    go.ctx = ctx == 4096 ? 0 : ctx; // 0: sized by the mode

    strata::drama::DramaModel m;
    std::string err;
    if (!m.load(model, mmproj, devs, split, err, check)) return fail(err); // only --check uploads the experts itself; the runners' pool owns them (J5)
    for (const auto& l : m.geometry().describe()) std::printf("INFO %s\n", l.c_str());
    if (verify && !m.verify(err)) return fail(err);
    for (const auto& l : m.report()) std::printf("INFO %s\n", l.c_str());
    int rc = 0;
    if (do_dump) rc = dump(m, rows_path, out_path, ctx);
    else if (do_gen) rc = generate(m, go);
    else if (do_tdump) rc = tower_dump(m, go);
    else if (do_replay) rc = replay(m, go);
    if (rc == 0) std::puts("OK");
    return rc;
}
