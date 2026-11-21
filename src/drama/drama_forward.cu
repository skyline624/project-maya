// src/drama/drama_forward.cu - Drama 3 backbone forward pass, one row at a time (J2).
//
// Layer (qwen35moe, llama.cpp src/models/qwen35moe.cpp): x += mix(rms(x) * attn_norm); x += ffn(rms(x) * post_attention_norm), where mix is
// the gated delta-net (30 layers) or the full attention (every 4th) and ffn is softmax top-8 routed Q8_0 experts + a sigmoid-gated shared
// expert.  One stream per device; the layers of a part run on its stream, the residual hops devices at the boundary.
#include "strata/drama/drama_forward.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "drama_kernels.hpp"

namespace strata::drama {

namespace {

#define CK(x)                                                                          \
    do {                                                                               \
        const cudaError_t e_ = (x);                                                    \
        if (e_ != cudaSuccess) { err = std::string(#x) + ": " + cudaGetErrorString(e_); return false; } \
    } while (0)

const float* F(const DTensor& t) { return (const float*) t.p; }

} // namespace

struct DramaRunner::Impl {
    const DramaModel& m;
    const DramaGeometry& g;
    int ctx = 0, pos = 0;
    struct Part {
        int dev = 0;
        cudaStream_t st = nullptr;
        float* mem = nullptr;
        uint64_t bytes = 0;
        // scratch slices of `mem`
        float *x, *xn, *xb, *xq, *mix, *qkv, *z, *cv, *gout, *ab, *qfull, *kvec, *vvec, *qbuf, *aout, *rl, *wts, *act, *moe, *shg, *shu, *shact, *shout, *sg, *hfin, *logits;
        int* ids;
    };
    struct LState { float *S = nullptr, *conv = nullptr, *K = nullptr, *V = nullptr; };
    std::vector<Part> parts;
    std::vector<LState> ls;
    std::vector<uint64_t> state_bytes; // per part
    const char* dump_path = nullptr;   // DRAMA_DUMP_LAYERS=<file>: per row (first DRAMA_DUMP_ROWS, default 1) the input then the residual after each layer
    int dump_rows = 1;
    bool emu = false; // DRAMA_EMU=1: round the activations like the reference does

    explicit Impl(const DramaModel& model) : m(model), g(model.geometry()) {}
    ~Impl() {
        for (auto& l : ls) {
            cudaFree(l.S); cudaFree(l.conv); cudaFree(l.K); cudaFree(l.V);
        }
        for (auto& p : parts) {
            if (p.mem) { cudaSetDevice(p.dev); cudaFree(p.mem); }
            if (p.st) cudaStreamDestroy(p.st);
        }
    }

    // the activation a BF16 GEMV reads: x itself, or (DRAMA_EMU) a BF16-rounded copy in `dst` (default P.xb)
    const float* bf16_in(Part& P, const float* x, int n, float* dst = nullptr) {
        if (!emu) return x;
        k::round_bf16(P.st, x, dst ? dst : P.xb, n);
        return dst ? dst : P.xb;
    }
    void layer(int il, Part& P);
    void gdn(int il, const DramaLayer& L, Part& P);
    void attn(int il, const DramaLayer& L, Part& P);
    void ffn(const DramaLayer& L, Part& P);
};

DramaRunner::DramaRunner(const DramaModel& model) : m_(new Impl(model)) {}
DramaRunner::~DramaRunner() = default;
int DramaRunner::pos() const { return m_->pos; }
int DramaRunner::n_out() const { return (int) m_->m.head().ids.size(); }

bool DramaRunner::init(int ctx, std::string& err) {
    Impl& s = *m_;
    const DramaGeometry& g = s.g;
    if (ctx < 1) { err = "ctx must be >= 1"; return false; }
    if (g.n_embd % 256 || g.ssm_inner != g.ssm_heads * 128 || g.ssm_conv != 4 || g.head_dim != 256 || g.n_ff_exp % 64 || g.n_embd % 64 || g.n_ff_shexp % 8 || g.n_expert > 512) {
        err = "geometry not supported by the J2 kernels";
        return false;
    }
    s.ctx = ctx;
    s.dump_path = std::getenv("DRAMA_DUMP_LAYERS");
    s.emu = std::getenv("DRAMA_EMU") != nullptr;
    if (const char* r = std::getenv("DRAMA_DUMP_ROWS")) s.dump_rows = std::max(1, std::atoi(r));
    // attention scores of one head live in shared memory: ctx floats (+ the static 33)
    if ((size_t) ctx * 4 > 96 * 1024) { err = "ctx too large for the attention kernel's shared-memory scores (max 24576)"; return false; }

    int np = 0;
    for (const auto& L : s.m.layers()) np = std::max(np, L.part + 1);
    s.parts.assign((size_t) np, Impl::Part());
    s.state_bytes.assign((size_t) np, 0);
    for (int il = 0; il < g.n_layer; ++il) s.parts[(size_t) s.m.layers()[(size_t) il].part].dev = s.m.layers()[(size_t) il].dev;
    const uint64_t E = (uint64_t) g.n_embd, CONV = (uint64_t) g.conv_dim(), INNER = (uint64_t) g.ssm_inner, NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t NOUT = s.m.head().ids.size(), NU = (uint64_t) g.n_expert_used, FE = (uint64_t) g.n_ff_exp, FS = (uint64_t) g.n_ff_shexp;
    for (auto& P : s.parts) {
        CK(cudaSetDevice(P.dev));
        CK(cudaStreamCreateWithFlags(&P.st, cudaStreamNonBlocking));
        if (!k::attn_allow_smem((size_t) ctx * 4)) { err = "attention: cannot raise the shared-memory limit"; return false; }
        uint64_t n = 0;
        const auto take = [&](uint64_t cnt) { const uint64_t o = n; n += (cnt + 63) / 64 * 64; return o; }; // 256-byte slices
        const uint64_t o_x = take(E), o_xn = take(E), o_xb = take(E), o_xq = take(E), o_mix = take(E), o_qkv = take(CONV), o_z = take(INNER), o_cv = take(CONV), o_gout = take(INNER), o_ab = take(2 * (uint64_t) g.ssm_heads),
                       o_qfull = take(NH * HD * 2), o_kvec = take(NKV * HD), o_vvec = take(NKV * HD), o_qbuf = take(NH * HD), o_aout = take(NH * HD), o_rl = take((uint64_t) g.n_expert),
                       o_wts = take(NU), o_ids = take(NU), o_act = take(NU * FE), o_moe = take(E), o_shg = take(FS), o_shu = take(FS), o_shact = take(FS), o_shout = take(E), o_sg = take(1),
                       o_h = take(E), o_lg = take(NOUT);
        P.bytes = n * 4;
        CK(cudaMalloc((void**) &P.mem, P.bytes));
        float* b = P.mem;
        P.x = b + o_x; P.xn = b + o_xn; P.xb = b + o_xb; P.xq = b + o_xq; P.mix = b + o_mix; P.qkv = b + o_qkv; P.z = b + o_z; P.cv = b + o_cv; P.gout = b + o_gout; P.ab = b + o_ab;
        P.qfull = b + o_qfull; P.kvec = b + o_kvec; P.vvec = b + o_vvec; P.qbuf = b + o_qbuf; P.aout = b + o_aout; P.rl = b + o_rl; P.wts = b + o_wts;
        P.ids = (int*) (b + o_ids); P.act = b + o_act; P.moe = b + o_moe; P.shg = b + o_shg; P.shu = b + o_shu; P.shact = b + o_shact; P.shout = b + o_shout;
        P.sg = b + o_sg; P.hfin = b + o_h; P.logits = b + o_lg;
    }
    // per-layer state
    s.ls.assign((size_t) g.n_layer, Impl::LState());
    for (int il = 0; il < g.n_layer; ++il) {
        const DramaLayer& L = s.m.layers()[(size_t) il];
        Impl::LState& st = s.ls[(size_t) il];
        CK(cudaSetDevice(L.dev));
        uint64_t& acc = s.state_bytes[(size_t) L.part];
        if (L.kind == LayerKind::Gdn) {
            const uint64_t sb = (uint64_t) g.ssm_heads * 128 * 128 * 4, cb = CONV * (uint64_t) (g.ssm_conv - 1) * 4;
            CK(cudaMalloc((void**) &st.S, sb));
            CK(cudaMalloc((void**) &st.conv, cb));
            acc += sb + cb;
        } else {
            const uint64_t kb = (uint64_t) ctx * NKV * HD * 4;
            CK(cudaMalloc((void**) &st.K, kb));
            CK(cudaMalloc((void**) &st.V, kb));
            acc += 2 * kb;
        }
    }
    s.pos = 0;
    return true;
}

void DramaRunner::reset() {
    Impl& s = *m_;
    s.pos = 0;
    for (int il = 0; il < s.g.n_layer; ++il) {
        const DramaLayer& L = s.m.layers()[(size_t) il];
        cudaSetDevice(L.dev);
        if (s.ls[(size_t) il].S) {
            cudaMemset(s.ls[(size_t) il].S, 0, (size_t) s.g.ssm_heads * 128 * 128 * 4);
            cudaMemset(s.ls[(size_t) il].conv, 0, (size_t) s.g.conv_dim() * (size_t) (s.g.ssm_conv - 1) * 4);
        }
    }
    for (auto& P : s.parts) { cudaSetDevice(P.dev); cudaDeviceSynchronize(); }
}

// ---------------------------------------------------------------- layers

void DramaRunner::Impl::gdn(int il, const DramaLayer& L, Part& P) {
    const cudaStream_t s = P.st;
    const int E = g.n_embd, CONV = g.conv_dim(), INNER = g.ssm_inner, H = g.ssm_heads;
    LState& st = ls[(size_t) il];
    const float* xi = bf16_in(P, P.xn, E);
    k::gemv_bf16(s, L.qkv.p, xi, P.qkv, CONV, E);
    k::gemv_bf16(s, L.gate.p, xi, P.z, INNER, E);
    k::gemv_bf16(s, L.alpha.p, xi, P.ab, H, E);
    k::gemv_bf16(s, L.beta.p, xi, P.ab + H, H, E);
    k::gdn_conv(s, P.qkv, st.conv, F(L.conv1d), P.cv, CONV, g.ssm_conv);
    k::gdn_l2(s, P.cv, 2 * g.ssm_groups, g.ssm_state, g.rms_eps); // q heads then k heads
    k::gdn_delta(s, st.S, P.cv, P.ab, P.ab + H, F(L.dt_bias), F(L.a), P.gout, H, g.ssm_groups);
    k::gdn_outnorm(s, P.gout, P.z, F(L.ssm_norm), H, g.rms_eps);
    k::gemv_bf16(s, L.ssm_out.p, bf16_in(P, P.gout, INNER, P.gout), P.mix, E, INNER);
}

void DramaRunner::Impl::attn(int il, const DramaLayer& L, Part& P) {
    const cudaStream_t s = P.st;
    const int E = g.n_embd, NH = g.n_head, NKV = g.n_head_kv, HD = g.head_dim;
    LState& st = ls[(size_t) il];
    const float* xi = bf16_in(P, P.xn, E);
    k::gemv_bf16(s, L.q.p, xi, P.qfull, NH * HD * 2, E);
    k::gemv_bf16(s, L.k.p, xi, P.kvec, NKV * HD, E);
    k::gemv_bf16(s, L.v.p, xi, P.vvec, NKV * HD, E);
    k::attn_prep(s, P.qfull, P.kvec, P.vvec, F(L.q_norm), F(L.k_norm), P.qbuf, st.K, st.V, pos, NH, NKV, g.rope_dim, g.rope_base, g.rms_eps);
    k::attn_decode(s, P.qbuf, P.qfull, st.K, st.V, P.aout, pos + 1, NH, NKV, 1.0f / std::sqrt((float) HD));
    k::gemv_bf16(s, L.o.p, bf16_in(P, P.aout, NH * HD, P.aout), P.mix, E, NH * HD);
}

void DramaRunner::Impl::ffn(const DramaLayer& L, Part& P) {
    const cudaStream_t s = P.st;
    const int E = g.n_embd, NE = g.n_expert, NU = g.n_expert_used, FE = g.n_ff_exp, FS = g.n_ff_shexp;
    const size_t esz = (size_t) (L.exps_gate.bytes / (uint64_t) NE); // gate, up and down experts are all the same size
    k::gemv_f32(s, L.gate_inp.p, P.xn, P.rl, NE, E);
    k::router(s, P.rl, P.ids, P.wts, NE, NU);
    const float* xq = P.xn;
    if (emu) { k::round_q8(s, P.xn, P.xq, E); xq = P.xq; }
    k::moe_gate_up(s, L.exps_gate.p, L.exps_up.p, esz, P.ids, xq, P.act, NU, FE, E / 32);
    if (emu) k::round_q8(s, P.act, P.act, NU * FE);
    k::moe_down(s, L.exps_down.p, esz, P.ids, P.wts, P.act, P.moe, E, FE / 32, NU, FE);
    const float* xi = bf16_in(P, P.xn, E);
    k::gemv_bf16(s, L.shexp_gate.p, xi, P.shg, FS, E);
    k::gemv_bf16(s, L.shexp_up.p, xi, P.shu, FS, E);
    k::silu_mul(s, P.shg, P.shu, P.shact, FS);
    k::gemv_bf16(s, L.shexp_down.p, bf16_in(P, P.shact, FS, P.shact), P.shout, E, FS);
    k::gemv_f32(s, L.shexp_inp.p, P.xn, P.sg, 1, E);
    k::ffn_combine(s, P.x, P.moe, P.shout, P.sg, E);
}

void DramaRunner::Impl::layer(int il, Part& P) {
    const DramaLayer& L = m.layers()[(size_t) il];
    const cudaStream_t s = P.st;
    const int E = g.n_embd;
    k::rmsnorm(s, P.x, F(L.attn_norm), P.xn, E, 1, g.rms_eps);
    if (L.kind == LayerKind::Gdn) gdn(il, L, P); else attn(il, L, P);
    k::add(s, P.x, P.mix, E);
    k::rmsnorm(s, P.x, F(L.post_norm), P.xn, E, 1, g.rms_eps);
    ffn(L, P);
}

bool DramaRunner::step(int tok, const float* emb, float* h_out, float* logits_out, std::string& err) {
    Impl& s = *m_;
    const DramaGeometry& g = s.g;
    const int E = g.n_embd;
    if (s.parts.empty()) { err = "init() not called"; return false; }
    if (s.pos >= s.ctx) { err = "context full (" + std::to_string(s.ctx) + " rows)"; return false; }
    std::vector<float> in((size_t) E);
    if (tok >= 0) {
        if ((int64_t) tok >= g.n_vocab) { err = "token id " + std::to_string(tok) + " outside the vocabulary"; return false; }
        const uint16_t* r = (const uint16_t*) s.m.token_embd_row(tok);
        for (int i = 0; i < E; ++i) { const uint32_t b = (uint32_t) r[i] << 16; std::memcpy(&in[(size_t) i], &b, 4); } // BF16 -> f32
    } else {
        if (!emb) { err = "embedding row missing"; return false; }
        std::memcpy(in.data(), emb, (size_t) E * 4);
    }
    const bool dump = s.dump_path && s.pos < s.dump_rows;
    FILE* df = dump ? std::fopen(s.dump_path, s.pos == 0 ? "wb" : "ab") : nullptr;
    std::vector<float> dv((size_t) E);
    const auto dump_x = [&](Impl::Part& P) {
        if (!df) return;
        cudaStreamSynchronize(P.st);
        cudaMemcpy(dv.data(), P.x, (size_t) E * 4, cudaMemcpyDeviceToHost);
        std::fwrite(dv.data(), 4, (size_t) E, df);
    };
    Impl::Part* cur = &s.parts[0];
    CK(cudaSetDevice(cur->dev));
    CK(cudaMemcpyAsync(cur->x, in.data(), (size_t) E * 4, cudaMemcpyHostToDevice, cur->st));
    CK(cudaStreamSynchronize(cur->st)); // `in` dies with this call
    dump_x(*cur);
    for (int il = 0; il < g.n_layer; ++il) {
        Impl::Part* P = &s.parts[(size_t) s.m.layers()[(size_t) il].part];
        if (P != cur) { // the residual hops to the next device
            CK(cudaSetDevice(cur->dev));
            CK(cudaStreamSynchronize(cur->st));
            CK(cudaMemcpyPeer(P->x, P->dev, cur->x, cur->dev, (size_t) E * 4));
            cur = P;
        }
        CK(cudaSetDevice(cur->dev));
        s.layer(il, *cur);
        dump_x(*cur);
    }
    if (df) std::fclose(df);
    // final norm and the compact lm_head, on the last part
    const DramaHead& hd = s.m.head();
    CK(cudaSetDevice(cur->dev));
    k::rmsnorm(cur->st, cur->x, F(hd.output_norm), cur->hfin, E, 1, g.rms_eps);
    k::gemv_bf16(cur->st, hd.lm_head.p, s.bf16_in(*cur, cur->hfin, E), cur->logits, (int) hd.ids.size(), E);
    if (h_out) CK(cudaMemcpyAsync(h_out, cur->hfin, (size_t) E * 4, cudaMemcpyDeviceToHost, cur->st));
    if (logits_out) CK(cudaMemcpyAsync(logits_out, cur->logits, hd.ids.size() * 4, cudaMemcpyDeviceToHost, cur->st));
    CK(cudaStreamSynchronize(cur->st));
    CK(cudaGetLastError());
    ++s.pos;
    return true;
}

std::vector<std::string> DramaRunner::report() const {
    const Impl& s = *m_;
    std::vector<std::string> v;
    for (size_t p = 0; p < s.parts.size(); ++p) {
        size_t fr = 0, tot = 0;
        cudaSetDevice(s.parts[p].dev);
        cudaMemGetInfo(&fr, &tot);
        char b[256];
        std::snprintf(b, sizeof b, "dev %d: state %.1f MiB (ctx %d), scratch %.2f MiB; free now %.2f / %.2f GiB", s.parts[p].dev, (double) s.state_bytes[p] / (1 << 20), s.ctx,
                      (double) s.parts[p].bytes / (1 << 20), (double) fr / (1ull << 30), (double) tot / (1ull << 30));
        v.push_back(b);
    }
    return v;
}

} // namespace strata::drama
