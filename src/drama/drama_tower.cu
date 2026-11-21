// src/drama/drama_tower.cu - Drama 3 codebook tower, one backbone row at a time (J3).
//
// llama.cpp tools/mtmd/models/drama3-gen.cpp (tower_layer, build_tower), in f32.  Layer il of one row:
//   e     = enorm * rms(mask * E_il[code] + extra)          extra: layer 0 only, sum of codes 2..9 of the input frame in the merged table
//   mixed = eh_proj([e ; hnorm * rms(x)])                   [4096] -> [2048]
//   x'    = mixed + o(attn(rope(qnorm(q)), rope(knorm(k)), v) * sigmoid(gate)),  ln1 * rms(mixed) in, GQA 16/2, head_dim 256, scale 1/16,
//           rope: GGML_ROPE_TYPE_NORMAL (adjacent pairs) on the first 64 dims, theta 1e7, position = row index, causal over every row so far
//   x''   = x' + down(silu(gate(ln2 * rms(x'))) * up(...)),  x = out_norm * rms(x'')
// Layers 0..3 take code0; layer 4 + ih takes the code the previous head gave (code0 for ih = 0) and its output feeds head ih (code ih + 1).
#include "strata/drama/drama_tower.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "drama_kernels.hpp"
#include "drama_sampling.hpp"

namespace strata::drama {

namespace {

#define CK(x)                                                                          \
    do {                                                                               \
        const cudaError_t e_ = (x);                                                    \
        if (e_ != cudaSuccess) { err = std::string(#x) + ": " + cudaGetErrorString(e_); return false; } \
    } while (0)

constexpr int kE = 2048, kHd = 256, kNH = 16, kNKV = 2, kRot = 64, kFF = 6144, kCb = 4096, kSortThreads = 1024;
constexpr size_t kSortSmem = (size_t) kCb * (8 + 4 + 4); // keys, weights, ids

const float* F(const DTensor& t) { return (const float*) t.p; }

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// every warp reduces the same partials: the result is in all threads, `sh` free again on return
__device__ float bsum(float v, float* sh) {
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    v = wsum(v);
    if (l == 0) sh[w] = v;
    __syncthreads();
    const float t = wsum(l < nw ? sh[l] : 0.f);
    __syncthreads();
    return t;
}
__device__ float bmax(float v, float* sh) {
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    v = wmax(v);
    if (l == 0) sh[w] = v;
    __syncthreads();
    const float t = wmax(l < nw ? sh[l] : -3.4e38f);
    __syncthreads();
    return t;
}
__device__ __forceinline__ float bf(uint32_t b16) { return __uint_as_float(b16 << 16); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.f / (1.f + expf(-x)); }

// cat[0..E) = enorm * rms(mask * tab[code] + extra), cat[E..2E) = hnorm * rms(x).  sc = {mask, extra_on}; frame: the input frame's codes
__global__ void embed_cat_k(const uint16_t* tab, const uint16_t* merged, const int* code, const int* frame, const float* sc, int use_extra, const float* enorm,
                            const float* hnorm, const float* x, float* cat, float eps) {
    __shared__ float sh[32];
    const float mask = sc[0];
    const bool extra = use_extra && sc[1] != 0.f;
    const uint16_t* row = tab + (size_t) code[0] * kE;
    float s = 0.f;
    for (int i = threadIdx.x; i < kE; i += blockDim.x) {
        float v = mask * bf(row[i]);
        if (extra)
            for (int g = 2; g < 10; ++g) v += bf(merged[((size_t) g * kCb + frame[g]) * kE + i]);
        cat[i] = v;
        s += v * v;
    }
    s = bsum(s, sh);
    float r = 1.f / sqrtf(s / (float) kE + eps);
    for (int i = threadIdx.x; i < kE; i += blockDim.x) cat[i] *= r * enorm[i];
    s = 0.f;
    for (int i = threadIdx.x; i < kE; i += blockDim.x) s += x[i] * x[i];
    s = bsum(s, sh);
    r = 1.f / sqrtf(s / (float) kE + eps);
    for (int i = threadIdx.x; i < kE; i += blockDim.x) cat[kE + i] = x[i] * r * hnorm[i];
}

// blocks 0..15: q heads, 16..17: k heads (norm, rope; k goes to the cache at pos), 18..19: v heads (to the cache)
__global__ void qk_prep_k(const float* qraw, const float* kraw, const float* vraw, const float* qn, const float* kn, float* qout, float* kc, float* vc, int pos,
                          float eps, float base) {
    __shared__ float sh[32], v[kHd];
    const int b = blockIdx.x, d = threadIdx.x;
    if (b >= kNH + kNKV) {
        const int h = b - kNH - kNKV;
        vc[((size_t) pos * kNKV + h) * kHd + d] = vraw[h * kHd + d];
        return;
    }
    const bool is_q = b < kNH;
    const float x = (is_q ? qraw + (size_t) b * kHd : kraw + (size_t) (b - kNH) * kHd)[d];
    const float* w = is_q ? qn : kn;
    float* dst = is_q ? qout + (size_t) b * kHd : kc + ((size_t) pos * kNKV + (b - kNH)) * kHd;
    const float ss = bsum(x * x, sh);
    v[d] = x * (1.f / sqrtf(ss / (float) kHd + eps)) * w[d];
    __syncthreads();
    if (d < kRot / 2) { // pair (2d, 2d+1), theta = pos * base^(-2d/kRot) as ggml's rope
        const float theta = (float) pos * powf(powf(base, -2.0f / (float) kRot), (float) d);
        const float c = cosf(theta), sn = sinf(theta), x0 = v[2 * d], x1 = v[2 * d + 1];
        dst[2 * d] = x0 * c - x1 * sn;
        dst[2 * d + 1] = x0 * sn + x1 * c;
    } else if (d >= kRot) {
        dst[d] = v[d];
    }
}

// one block per query head, 8 warps: scores to shared memory, softmax, weighted sum of V, output gate
__global__ void tower_attn_k(const float* q, const float* gate, const float* kc, const float* vc, float* out, int n, float scale) {
    extern __shared__ float sc[];
    __shared__ float sh[32];
    const int h = blockIdx.x, kvh = h / (kNH / kNKV), tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    float qr[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) qr[t] = q[h * kHd + lane * 8 + t];
    for (int j = warp; j < n; j += 8) {
        const float4* kp = (const float4*) (kc + ((size_t) j * kNKV + kvh) * kHd + lane * 8);
        const float4 a = kp[0], b = kp[1];
        float s = qr[0] * a.x + qr[1] * a.y + qr[2] * a.z + qr[3] * a.w + qr[4] * b.x + qr[5] * b.y + qr[6] * b.z + qr[7] * b.w;
        s = wsum(s);
        if (lane == 0) sc[j] = s * scale;
    }
    __syncthreads();
    float m = -3.4e38f;
    for (int j = tid; j < n; j += blockDim.x) m = fmaxf(m, sc[j]);
    m = bmax(m, sh);
    float sum = 0.f;
    for (int j = tid; j < n; j += blockDim.x) { sc[j] = expf(sc[j] - m); sum += sc[j]; }
    sum = bsum(sum, sh); // also the barrier that publishes sc
    float o = 0.f;
    for (int j = 0; j < n; ++j) o += sc[j] * vc[((size_t) j * kNKV + kvh) * kHd + tid];
    out[h * kHd + tid] = o / sum * sigmoid_f(gate[h * kHd + tid]);
}

__device__ __forceinline__ uint32_t ordered_bits(float f) {
    const uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ float from_ordered(uint32_t o) { return __uint_as_float((o & 0x80000000u) ? (o & 0x7fffffffu) : ~o); }

// one block: bitonic sort of the 4096 logits (value descending, id ascending), weights, then thread 0 draws as drama_sampling.hpp says
__global__ void head_sample_k(const float* logits, int* code_out, SampleSpec sp, uint64_t seed, uint32_t ctr, uint32_t stream) {
    extern __shared__ unsigned char smem[];
    __shared__ float sh[32];
    uint64_t* key = (uint64_t*) smem;
    float* w = (float*) (smem + (size_t) kCb * 8);
    int* ids = (int*) (smem + (size_t) kCb * 12);
    const int tid = threadIdx.x;
    for (int i = tid; i < kCb; i += blockDim.x) key[i] = ((uint64_t) ordered_bits(logits[i]) << 32) | (uint64_t) (0xFFFFFFFFu - (uint32_t) i);
    __syncthreads();
    for (int k = 2; k <= kCb; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < kCb; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const bool desc = (i & k) == 0;
                    const uint64_t a = key[i], b = key[ixj];
                    if ((a < b) == desc) { key[i] = b; key[ixj] = a; }
                }
            }
            __syncthreads();
        }
    const float top = from_ordered((uint32_t) (key[0] >> 32)), t = fmaxf(sp.temp, 1e-5f);
    float z = 0.f;
    for (int i = tid; i < kCb; i += blockDim.x) {
        const float val = from_ordered((uint32_t) (key[i] >> 32));
        ids[i] = (int) (0xFFFFFFFFu - (uint32_t) key[i]);
        w[i] = expf((val - top) / t);
        z += w[i];
    }
    z = bsum(z, sh);
    if (tid == 0) code_out[0] = ids[draw_sorted(w, ids, kCb, (double) z, sp, philox_uniform(seed, stream, ctr))];
}

// (E_lm[c0] + E_cb1[c1]) / sqrt(2)
__global__ void next_emb_k(const uint16_t* out_embd, const uint16_t* cb1, const int* codes, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kE) out[i] = (bf(out_embd[(size_t) codes[0] * kE + i]) + bf(cb1[(size_t) codes[1] * kE + i])) * 0.70710678118654752f;
}

} // namespace

struct TowerRunner::Impl {
    const DramaModel& m;
    const DramaGeometry& g;
    TowerSampling smp;
    SampleSpec head_sp;
    int ctx = 0, pos = 0, nsh = 0, nheads = 0, dev = 0;
    bool emu = false; // DRAMA_EMU=1: round the activations like the reference (BF16 weights are fed BF16 activations)
    cudaStream_t st = nullptr;
    float* mem = nullptr;
    uint64_t bytes = 0, kv_bytes = 0;
    float *kc = nullptr, *vc = nullptr;
    // scratch slices of `mem`
    float *d_h, *d_x, *cat, *mixed, *cur, *qr, *kr, *vr, *gate, *qbuf, *aout, *o, *g1, *u1, *act, *rb, *logits, *dump, *next, *sc;
    int* codes; // [10] codes of the output frame, then [10] of the input frame
    explicit Impl(const DramaModel& model) : m(model), g(model.geometry()) {}
    ~Impl() {
        cudaSetDevice(dev);
        cudaFree(mem);
        cudaFree(kc);
        cudaFree(vc);
        if (st) cudaStreamDestroy(st);
    }
    const float* inb(const float* x, int n) { // the activation a BF16 GEMV reads
        if (!emu) return x;
        k::round_bf16(st, x, rb, n);
        return rb;
    }
};

TowerRunner::TowerRunner(const DramaModel& model) : m_(new Impl(model)) {}
TowerRunner::~TowerRunner() = default;
int TowerRunner::pos() const { return m_->pos; }
void TowerRunner::reset() { m_->pos = 0; }

bool TowerRunner::init(int ctx, const TowerSampling& sampling, std::string& err) {
    Impl& s = *m_;
    const DramaGeometry& g = s.g;
    if (g.n_embd != kE || g.t_head != kNH || g.t_head_kv != kNKV || g.t_head_dim != kHd || g.t_ff != kFF || g.t_codebooks != kCodebooks || g.t_resid_heads != 9 ||
        g.t_layers != 13) {
        err = "tower geometry not supported by the J3 kernels";
        return false;
    }
    if (ctx < 1 || (size_t) ctx * 4 > 48 * 1024) { err = "tower ctx must be in 1..12288 (attention scores live in shared memory)"; return false; }
    s.ctx = ctx;
    s.smp = sampling;
    s.head_sp = head_spec(sampling.llama ? SamplingMode::Llama : SamplingMode::Official, sampling.temp, sampling.top_k, sampling.top_p);
    s.nsh = g.t_layers - g.t_resid_heads;
    s.nheads = g.t_resid_heads;
    s.dev = s.m.tower().dev;
    s.emu = std::getenv("DRAMA_EMU") != nullptr;
    CK(cudaSetDevice(s.dev));
    CK(cudaStreamCreateWithFlags(&s.st, cudaStreamNonBlocking));
    CK(cudaFuncSetAttribute(head_sample_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kSortSmem));
    uint64_t n = 0;
    const auto take = [&](uint64_t cnt) { const uint64_t o = n; n += (cnt + 63) / 64 * 64; return o; }; // 256-byte slices
    const uint64_t o_h = take(kE), o_x = take(kE), o_cat = take(2 * kE), o_mixed = take(kE), o_cur = take(kE), o_q = take(kNH * kHd), o_k = take(kNKV * kHd), o_v = take(kNKV * kHd),
                   o_gate = take(kNH * kHd), o_qb = take(kNH * kHd), o_ao = take(kNH * kHd), o_o = take(kE), o_g = take(kFF), o_u = take(kFF), o_act = take(kFF), o_rb = take(kFF),
                   o_lg = take((uint64_t) s.nheads * kCb), o_dump = take((uint64_t) g.t_layers * kE), o_next = take(kE), o_sc = take(2), o_codes = take(20);
    s.bytes = n * 4;
    CK(cudaMalloc((void**) &s.mem, s.bytes));
    float* b = s.mem;
    s.d_h = b + o_h; s.d_x = b + o_x; s.cat = b + o_cat; s.mixed = b + o_mixed; s.cur = b + o_cur; s.qr = b + o_q; s.kr = b + o_k; s.vr = b + o_v; s.gate = b + o_gate;
    s.qbuf = b + o_qb; s.aout = b + o_ao; s.o = b + o_o; s.g1 = b + o_g; s.u1 = b + o_u; s.act = b + o_act; s.rb = b + o_rb; s.logits = b + o_lg; s.dump = b + o_dump;
    s.next = b + o_next; s.sc = b + o_sc; s.codes = (int*) (b + o_codes);
    s.kv_bytes = (uint64_t) g.t_layers * ctx * kNKV * kHd * 4;
    CK(cudaMalloc((void**) &s.kc, s.kv_bytes));
    CK(cudaMalloc((void**) &s.vc, s.kv_bytes));
    s.pos = 0;
    return true;
}

bool TowerRunner::step(const TowerStep& t, std::string& err) {
    Impl& s = *m_;
    const DramaGeometry& g = s.g;
    const DramaTower& tw = s.m.tower();
    if (!s.mem) { err = "init() not called"; return false; }
    if (s.pos >= s.ctx) { err = "tower context full (" + std::to_string(s.ctx) + " rows)"; return false; }
    if (!t.h) { err = "tower step without h"; return false; }
    if (t.code0 >= kCb) { err = "code0 " + std::to_string(t.code0) + " out of range"; return false; }
    if (t.forced && t.code0 < 0) { err = "forced codes need code0 >= 0"; return false; }
    const bool next_speech = t.code0 >= 0, cur_speech = t.frame_in && t.frame_in[0] >= 0;
    const auto clampc = [](int v) { return std::min(std::max(v, 0), kCb - 1); };
    struct { int codes[10]; int frame[10]; float sc[2]; } in;
    std::memset(&in, 0, sizeof in);
    in.codes[0] = std::max(t.code0, 0);
    if (t.forced)
        for (int i = 0; i < s.nheads; ++i) in.codes[1 + i] = clampc(t.forced[i]);
    if (cur_speech)
        for (int i = 0; i < kCodebooks; ++i) in.frame[i] = clampc(t.frame_in[i]);
    in.sc[0] = next_speech ? 1.f : 0.f;
    in.sc[1] = cur_speech ? 1.f : 0.f;
    CK(cudaSetDevice(s.dev));
    const cudaStream_t st = s.st;
    static_assert(sizeof(in) == 22 * 4, "tower input block");
    CK(cudaMemcpyAsync(s.codes, in.codes, 40, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(s.sc, in.sc, 8, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(s.codes + 10, in.frame, 40, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(s.d_h, t.h, (size_t) kE * 4, cudaMemcpyHostToDevice, st));
    const float eps = g.t_eps;
    const size_t kvl = (size_t) s.ctx * kNKV * kHd;
    for (int il = 0; il < g.t_layers; ++il) {
        const DramaTowerLayer& L = tw.blk[(size_t) il];
        const int ci = std::max(0, il - s.nsh);
        embed_cat_k<<<1, 512, 0, st>>>((const uint16_t*) L.embd.p, (const uint16_t*) tw.embd.p, s.codes + ci, s.codes + 10, s.sc, il == 0, F(L.enorm), F(L.hnorm),
                                       il == 0 ? s.d_h : s.d_x, s.cat, eps);
        k::gemv_bf16(st, L.eh_proj.p, s.inb(s.cat, 2 * kE), s.mixed, kE, 2 * kE);
        k::rmsnorm(st, s.mixed, F(L.ln1), s.cur, kE, 1, eps);
        const float* ci_ = s.inb(s.cur, kE);
        k::gemv_bf16(st, L.q.p, ci_, s.qr, kNH * kHd, kE);
        k::gemv_bf16(st, L.k.p, ci_, s.kr, kNKV * kHd, kE);
        k::gemv_bf16(st, L.v.p, ci_, s.vr, kNKV * kHd, kE);
        k::gemv_bf16(st, L.gate.p, ci_, s.gate, kNH * kHd, kE);
        qk_prep_k<<<kNH + 2 * kNKV, kHd, 0, st>>>(s.qr, s.kr, s.vr, F(L.q_norm), F(L.k_norm), s.qbuf, s.kc + il * kvl, s.vc + il * kvl, s.pos, eps, 1e7f);
        tower_attn_k<<<kNH, kHd, (size_t) (s.pos + 1) * 4, st>>>(s.qbuf, s.gate, s.kc + il * kvl, s.vc + il * kvl, s.aout, s.pos + 1, 1.0f / std::sqrt((float) kHd));
        k::gemv_bf16(st, L.out.p, s.inb(s.aout, kNH * kHd), s.o, kE, kNH * kHd);
        k::add(st, s.mixed, s.o, kE);
        k::rmsnorm(st, s.mixed, F(L.ln2), s.cur, kE, 1, eps);
        const float* fi = s.inb(s.cur, kE);
        k::gemv_bf16(st, L.ffn_gate.p, fi, s.g1, kFF, kE);
        k::gemv_bf16(st, L.ffn_up.p, fi, s.u1, kFF, kE);
        k::silu_mul(st, s.g1, s.u1, s.act, kFF);
        k::gemv_bf16(st, L.ffn_down.p, s.inb(s.act, kFF), s.o, kE, kFF);
        k::add(st, s.mixed, s.o, kE);
        k::rmsnorm(st, s.mixed, F(L.out_norm), s.d_x, kE, 1, eps);
        if (t.layer_out) CK(cudaMemcpyAsync(s.dump + (size_t) il * kE, s.d_x, (size_t) kE * 4, cudaMemcpyDeviceToDevice, st));
        if (il >= s.nsh && next_speech) {
            const int ih = il - s.nsh;
            const uint8_t* head = (const uint8_t*) tw.head.p + (size_t) ih * kCb * kE * 2;
            k::gemv_bf16(st, head, s.inb(s.d_x, kE), s.logits + (size_t) ih * kCb, kCb, kE);
            if (!t.forced)
                head_sample_k<<<1, kSortThreads, kSortSmem, st>>>(s.logits + (size_t) ih * kCb, s.codes + ih + 1, s.head_sp, s.smp.seed, t.ctr, (uint32_t) (2 + ih));
        }
    }
    if (next_speech)
        next_emb_k<<<(kE + 255) / 256, 256, 0, st>>>((const uint16_t*) tw.out_embd.p, (const uint16_t*) tw.embd.p + (size_t) kCb * kE, s.codes, s.next);
    if (t.codes_out && next_speech) CK(cudaMemcpyAsync(t.codes_out, s.codes, (size_t) kCodebooks * 4, cudaMemcpyDeviceToHost, st));
    if (t.next_emb && next_speech) CK(cudaMemcpyAsync(t.next_emb, s.next, (size_t) kE * 4, cudaMemcpyDeviceToHost, st));
    if (t.layer_out) CK(cudaMemcpyAsync(t.layer_out, s.dump, (size_t) g.t_layers * kE * 4, cudaMemcpyDeviceToHost, st));
    if (t.head_logits && next_speech) CK(cudaMemcpyAsync(t.head_logits, s.logits, (size_t) s.nheads * kCb * 4, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());
    ++s.pos;
    return true;
}

std::vector<std::string> TowerRunner::report() const {
    const Impl& s = *m_;
    char b[256];
    std::snprintf(b, sizeof b, "tower dev %d: KV %.1f MiB (ctx %d), scratch %.2f MiB, sampling %s (temp %g, top_k %d, top_p %g), emu %d", s.dev,
                  (double) (2 * s.kv_bytes) / (1 << 20), s.ctx, (double) s.bytes / (1 << 20), s.smp.llama ? "llama" : "official", s.smp.temp, s.smp.top_k, s.smp.top_p, (int) s.emu);
    return {b};
}

} // namespace strata::drama
