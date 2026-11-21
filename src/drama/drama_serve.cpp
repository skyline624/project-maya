// src/drama/drama_serve.cpp - maya-drama --serve: JSON lines on stdin/stdout, see include/strata/drama/drama_serve.hpp.
// The prefill, the frame loop and the .codes files repeat drama_main.cpp's (generate); they are kept apart so that file stays the parity tool.
#include "strata/drama/drama_serve.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "drama_sampling.hpp"
#include "strata/drama/drama_forward.hpp"
#include "strata/drama/drama_model.hpp"
#include "strata/drama/drama_tower.hpp"

namespace strata::drama {
namespace {

double ms_since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }

// ---- a flat JSON object: strings, numbers, true/false/null (as 1/0/0) and arrays of integers
struct Json {
    std::map<std::string, std::string> str;
    std::map<std::string, double> num;
    std::map<std::string, std::vector<int32_t>> arr;
    std::string s(const std::string& k, const std::string& d = "") const { auto i = str.find(k); return i == str.end() ? d : i->second; }
    double n(const std::string& k, double d) const { auto i = num.find(k); return i == num.end() ? d : i->second; }
};

struct Parser {
    const std::string& t;
    size_t i = 0;
    void ws() { while (i < t.size() && std::isspace((unsigned char) t[i])) ++i; }
    bool eat(char c) { ws(); if (i < t.size() && t[i] == c) { ++i; return true; } return false; }
    bool string(std::string& out) {
        if (!eat('"')) return false;
        out.clear();
        while (i < t.size() && t[i] != '"') {
            char c = t[i++];
            if (c != '\\') { out += c; continue; }
            if (i >= t.size()) return false;
            c = t[i++];
            switch (c) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'u': { // BMP code point as UTF-8
                    if (i + 4 > t.size()) return false;
                    const unsigned u = (unsigned) std::strtoul(t.substr(i, 4).c_str(), nullptr, 16);
                    i += 4;
                    if (u < 0x80) out += (char) u;
                    else if (u < 0x800) { out += (char) (0xC0 | u >> 6); out += (char) (0x80 | (u & 0x3F)); }
                    else { out += (char) (0xE0 | u >> 12); out += (char) (0x80 | ((u >> 6) & 0x3F)); out += (char) (0x80 | (u & 0x3F)); }
                    break;
                }
                default: out += c; // " \ /
            }
        }
        return i < t.size() && t[i++] == '"';
    }
    bool number(double& v) {
        ws();
        char* e = nullptr;
        v = std::strtod(t.c_str() + i, &e);
        if (e == t.c_str() + i) return false;
        i = (size_t) (e - t.c_str());
        return true;
    }
    bool object(Json& j) {
        if (!eat('{')) return false;
        if (eat('}')) return true;
        do {
            std::string k;
            if (!string(k) || !eat(':')) return false;
            ws();
            if (i >= t.size()) return false;
            if (t[i] == '"') { if (!string(j.str[k])) return false; }
            else if (t[i] == '[') {
                ++i;
                auto& a = j.arr[k];
                if (!eat(']')) {
                    do { double v; if (!number(v)) return false; a.push_back((int32_t) v); } while (eat(','));
                    if (!eat(']')) return false;
                }
            } else if (t.compare(i, 4, "true") == 0) { j.num[k] = 1; i += 4; }
            else if (t.compare(i, 5, "false") == 0) { j.num[k] = 0; i += 5; }
            else if (t.compare(i, 4, "null") == 0) { i += 4; }
            else if (!number(j.num[k])) return false;
        } while (eat(','));
        return eat('}');
    }
};

std::string esc(const std::string& s) {
    std::string o;
    for (const unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += (char) c; }
        else if (c < 0x20) o += ' ';
        else o += (char) c;
    }
    return o;
}

void reply(const std::string& line) {
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}

// ---- .codes: "DRC1", u32 n_frames, u32 10, i32 frame-major
bool read_codes(const std::string& path, std::vector<int32_t>& codes, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    char magic[4];
    uint32_t n = 0, cb = 0;
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "DRC1", 4) == 0 && std::fread(&n, 4, 1, f) == 1 && std::fread(&cb, 4, 1, f) == 1 && cb == (uint32_t) kCodebooks;
    if (!ok) err = path + ": not a DRC1 file with 10 codes per frame";
    else {
        codes.resize((size_t) n * cb);
        ok = std::fread(codes.data(), 4, codes.size(), f) == codes.size();
        if (!ok) err = path + ": truncated";
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
    return std::fclose(f) == 0;
}

// ---- a prompt: its rows (token id, or -1 for a reference frame) and the codes of the reference frames
struct Prompt {
    std::vector<int32_t> rows;
    std::vector<int32_t> ref; // frame-major
    long ref_begin = 0;
    bool speech(long r) const { return r >= ref_begin && (size_t) (r - ref_begin) < ref.size() / kCodebooks; }
    const int32_t* frame(long r) const { return ref.data() + (size_t) (r - ref_begin) * kCodebooks; }
};

// the prefill of drama_main.cpp: rows [from, to) through the backbone and the tower (run on every row but the last of the n_prompt rows, the codes of a next reference frame
// forced).  A reference frame's embedding row is the tower's `next_emb` of the row before it, (E_lm[c0] + E_cb1[c1]) / sqrt(2), which the speaker encoder's rows equal to 1.2e-7.
bool prefill(DramaRunner& bb, TowerRunner& tw, const Prompt& p, size_t from, size_t to, std::vector<float>& emb, std::vector<float>& h, std::vector<float>& lg, std::string& err) {
    const size_t n_prompt = p.rows.size();
    for (size_t r = from; r < to; ++r) {
        const bool sp = p.speech((long) r);
        if (!bb.step(sp ? -1 : p.rows[r], sp ? emb.data() : nullptr, h.data(), lg.data(), err)) { err = "row " + std::to_string(r) + ": " + err; return false; }
        if (r + 1 < n_prompt) {
            TowerStep t;
            t.h = h.data();
            if (sp) t.frame_in = p.frame((long) r);
            if (p.speech((long) r + 1)) {
                t.code0 = p.frame((long) r + 1)[0];
                t.forced = p.frame((long) r + 1) + 1;
                t.next_emb = emb.data();
            }
            if (!tw.step(t, err)) { err = "tower row " + std::to_string(r) + ": " + err; return false; }
        }
    }
    return true;
}

struct Voice {
    std::string key;
    size_t rows = 0;
    DeviceSnapshot bb, tw;
};

struct Server {
    DramaModel& m;
    DramaRunner bb;
    TowerRunner tw;
    bool llama;
    int ctx, cap;
    std::list<std::unique_ptr<Voice>> voices; // most recently used first
    std::vector<float> emb, h, lg;

    Server(DramaModel& model, bool llama_, int ctx_, int cap_) : m(model), bb(model), tw(model), llama(llama_), ctx(ctx_), cap(cap_) {}

    bool init(std::string& err) {
        TowerSampling s;
        s.llama = llama;
        s.seed = 1; // a request's seed goes into the draw counters (frame + 4096 * (seed - 1)), so seed 1 is --generate --seed 1
        if (!bb.init(ctx, err) || !tw.init(ctx, s, err)) return false;
        emb.resize((size_t) m.geometry().n_embd);
        h.resize(emb.size());
        lg.resize((size_t) bb.n_out());
        return true;
    }

    // returns the answer line
    std::string gen(const Json& q) {
        const std::string id = fmt("%lld", (long long) q.n("id", 0));
        const auto bad = [&](const std::string& why) { return "{\"id\":" + id + ",\"ok\":false,\"error\":\"" + esc(why) + "\"}"; };
        std::string err;
        const auto tk = q.arr.find("tokens");
        if (tk == q.arr.end() || tk->second.empty()) return bad("tokens: a non-empty array is needed");
        Prompt p;
        const long ref_at = (long) q.n("ref_at", -1);
        if (ref_at >= 0) {
            if (ref_at > (long) tk->second.size() || !read_codes(q.s("ref_codes"), p.ref, err)) return bad(err.empty() ? "ref_at is past the tokens" : err);
            p.ref_begin = ref_at;
            p.rows.assign(tk->second.begin(), tk->second.begin() + ref_at);
            p.rows.insert(p.rows.end(), p.ref.size() / kCodebooks, -1);
            p.rows.insert(p.rows.end(), tk->second.begin() + ref_at, tk->second.end());
        } else {
            p.rows = tk->second;
            p.ref_begin = (long) p.rows.size();
        }
        for (const int32_t t : p.rows)
            if (t < -1 || t >= m.geometry().n_vocab) return bad("token id " + std::to_string(t) + " out of range");
        const size_t n_prompt = p.rows.size();
        const std::string out = q.s("out");
        if (out.empty()) return bad("out: a path is needed");
        if ((int) n_prompt >= ctx) return bad("the prompt has " + std::to_string(n_prompt) + " rows, the context " + std::to_string(ctx));
        const long N = (long) q.n("prefix_rows", 0);
        const std::string key = q.s("voice");
        if (N < 0 || (N > 0 && (key.empty() || (size_t) N >= n_prompt || p.speech(N)))) return bad("prefix_rows must be 0, or end after the voice and before the text with a voice key");
        const uint64_t seed = (uint64_t) q.n("seed", 1);
        const int max_frames = (int) q.n("max_frames", 1024);

        const auto t_req = std::chrono::steady_clock::now();
        size_t from = 0;
        const char* pre = "none";
        double t_prefix = 0;
        auto t0 = std::chrono::steady_clock::now();
        if (N > 0) {
            auto v = std::find_if(voices.begin(), voices.end(), [&](const std::unique_ptr<Voice>& x) { return x->key == key && x->rows == (size_t) N; });
            if (v != voices.end()) {
                if (!bb.restore((*v)->bb, err) || !tw.restore((*v)->tw, err)) return bad(err);
                voices.splice(voices.begin(), voices, v);
                pre = "hit";
            } else {
                bb.reset();
                tw.reset();
                if (!prefill(bb, tw, p, 0, (size_t) N, emb, h, lg, err)) return bad(err);
                voices.remove_if([&](const std::unique_ptr<Voice>& x) { return x->key == key; });
                while ((int) voices.size() >= cap) voices.pop_back(); // frees its device memory before the new snapshot takes some
                auto nv = std::make_unique<Voice>();
                nv->key = key;
                nv->rows = (size_t) N;
                if (!bb.save(nv->bb, err) || !tw.save(nv->tw, err)) return bad(err);
                voices.push_front(std::move(nv));
                pre = "miss";
            }
            from = (size_t) N;
            t_prefix = ms_since(t0);
        } else {
            bb.reset();
            tw.reset();
        }
        t0 = std::chrono::steady_clock::now();
        if (!prefill(bb, tw, p, from, n_prompt, emb, h, lg, err)) return bad(err);
        const double t_prefill = ms_since(t0);

        MainSampler ms(llama ? SamplingMode::Llama : SamplingMode::Official, seed);
        const uint32_t ctr0 = (uint32_t) ((seed - 1) * 4096);
        std::vector<int32_t> codes;
        int32_t frame[kCodebooks];
        double t_bb = 0, t_tw = 0;
        int n = 0;
        const char* stop = "max_frames";
        const auto tg = std::chrono::steady_clock::now();
        for (int f = 0; f < max_frames; ++f) {
            const int pick = ms.sample(lg.data(), (uint32_t) f);
            if (pick == 0) { stop = "eos"; break; }
            if (bb.pos() >= ctx) { stop = "context"; break; }
            ms.push(pick);
            TowerStep t;
            t.h = h.data();
            t.code0 = pick - 1;
            t.frame_in = f ? frame : nullptr;
            t.ctr = ctr0 + (uint32_t) f;
            t.codes_out = frame;
            t.next_emb = emb.data();
            const auto t1 = std::chrono::steady_clock::now();
            if (!tw.step(t, err)) return bad("frame " + std::to_string(f) + ": tower: " + err);
            t_tw += ms_since(t1);
            for (int i = 0; i < kCodebooks; ++i) codes.push_back(i == 0 ? frame[0] : std::min<int32_t>(frame[i], kResidualCodes - 1)); // llama.cpp clamps at the codec
            const auto t2 = std::chrono::steady_clock::now();
            if (!bb.step(-1, emb.data(), h.data(), lg.data(), err)) return bad("frame " + std::to_string(f) + ": backbone: " + err);
            t_bb += ms_since(t2);
            ++n;
        }
        const double t_gen = ms_since(tg);
        if (!write_codes(out, codes, err)) return bad(err);
        return fmt("{\"id\":%s,\"ok\":true,\"frames\":%d,\"stop\":\"%s\",\"prefix\":\"%s\",\"rows\":%zu,\"prefix_ms\":%.1f,\"prefill_ms\":%.1f,\"gen_ms\":%.1f,\"backbone_ms\":%.1f,\"tower_ms\":%.1f,\"total_ms\":%.1f}",
                   id.c_str(), n, stop, pre, n_prompt, t_prefix, t_prefill, t_gen, t_bb, t_tw, ms_since(t_req));
    }
};

std::vector<int> ints(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');) v.push_back(std::atoi(t.c_str()));
    return v;
}

} // namespace

bool serve_requested(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--serve") == 0) return true;
    return false;
}

int serve_main(int argc, char** argv) {
    std::string model, mmproj, sampling = "official";
    std::vector<int> devs = {0, 1}, split;
    int ctx = 4096, cap = 4;
    ExpertOptions xo; // J5 options, same as --generate
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto val = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--serve") continue;
        else if (a == "--model") model = val();
        else if (a == "--mmproj") mmproj = val();
        else if (a == "--devs") devs = ints(val());
        else if (a == "--layer-split") split = ints(val());
        else if (a == "--ctx") ctx = std::atoi(val());
        else if (a == "--voices") cap = std::atoi(val());
        else if (a == "--sampling") sampling = val();
        else if (a == "--expert-vram-mib") xo.vram_mib = std::atoll(val());
        else if (a == "--vram-reserve-mib") xo.reserve_mib = std::atoll(val());
        else if (a == "--expert-profile") xo.profile = val();
        else if (a == "--expert-profile-save") xo.profile_save = val();
        else if (a == "--expert-prefetch") xo.prefetch = std::atoi(val()) != 0;
        else { std::printf("ERROR unknown option %s\n", a.c_str()); return 2; }
    }
    if (model.empty() || mmproj.empty() || ctx < 64 || cap < 1 || (sampling != "official" && sampling != "llama")) {
        std::puts("usage: maya-drama --serve --model <backbone.gguf> --mmproj <mmproj.gguf> [--devs 0,1] [--layer-split 20] [--ctx 4096] [--voices 4] [--sampling official|llama] [J5 expert options]");
        return 2;
    }
    DramaModel m;
    std::string err;
    if (!m.load(model, mmproj, devs, split, err, false)) { std::printf("ERROR %s\n", err.c_str()); return 1; }
    for (const auto& l : m.report()) std::printf("INFO %s\n", l.c_str());
    Server s(m, sampling == "llama", ctx, cap);
    if (!s.init(err)) { std::printf("ERROR %s\n", err.c_str()); return 1; }
    for (const auto& l : s.bb.report()) std::printf("INFO %s\n", l.c_str());
    for (const auto& l : s.tw.report()) std::printf("INFO %s\n", l.c_str());
    reply(fmt("{\"event\":\"ready\",\"ctx\":%d,\"voices\":%d,\"sampling\":\"%s\"}", ctx, cap, sampling.c_str()));
    for (std::string line; std::getline(std::cin, line);) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        Json q;
        Parser ps{line};
        if (!ps.object(q)) { reply("{\"ok\":false,\"error\":\"bad JSON line\"}"); continue; }
        const std::string op = q.s("op");
        if (op == "quit") break;
        else if (op == "gen") reply(s.gen(q));
        else reply(fmt("{\"id\":%lld,\"ok\":false,\"error\":\"unknown op\"}", (long long) q.n("id", 0)));
    }
    return 0;
}

} // namespace strata::drama
