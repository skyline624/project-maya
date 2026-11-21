// src/drama/drama_kernels.cu - Drama 3 backbone kernels, one row at a time (J2).  Plain and f32; correctness before speed.
#include "drama_kernels.hpp"

#include <cuda_fp16.h>

#include <cfloat>

namespace strata::drama::k {

namespace {

constexpr int kWpb = 4;  // warps per block of the one-row-per-warp kernels
constexpr int kHd = 256; // attention head dim (checked at load)
constexpr int kS = 128;  // GDN head dim (checked at load)

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
// every warp reduces the same 32 partials, so the result is in all threads and `sh` is free again on return
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
    const float t = wmax(l < nw ? sh[l] : -FLT_MAX);
    __syncthreads();
    return t;
}
__device__ __forceinline__ float bf(uint32_t b16) { return __uint_as_float(b16 << 16); } // BF16 bits -> f32
__device__ __forceinline__ float silu_f(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.f / (1.f + expf(-x)); }

__global__ void rmsnorm_k(const float* x, const float* w, float* out, int n, float eps) {
    __shared__ float sh[32];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* o = out + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = bsum(s, sh);
    const float r = 1.f / sqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = xr[i] * r * w[i];
}

// one warp per output row, 16-byte loads of 8 BF16
__global__ void gemv_bf16_k(const uint16_t* W, const float* x, float* y, int rows, int K) {
    const int row = blockIdx.x * kWpb + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= rows) return;
    const uint4* wr = (const uint4*) (W + (size_t) row * K);
    float acc = 0.f;
    for (int c = lane; c < K / 8; c += 32) {
        const uint4 v = __ldg(wr + c);
        const float4 a = *(const float4*) (x + c * 8), b = *(const float4*) (x + c * 8 + 4);
        acc += bf(v.x & 0xffffu) * a.x + bf(v.x >> 16) * a.y + bf(v.y & 0xffffu) * a.z + bf(v.y >> 16) * a.w;
        acc += bf(v.z & 0xffffu) * b.x + bf(v.z >> 16) * b.y + bf(v.w & 0xffffu) * b.z + bf(v.w >> 16) * b.w;
    }
    acc = wsum(acc);
    if (lane == 0) y[row] = acc;
}

__global__ void gemv_f32_k(const float* W, const float* x, float* y, int rows, int K) {
    const int row = blockIdx.x * kWpb + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= rows) return;
    const float4* wr = (const float4*) (W + (size_t) row * K);
    float acc = 0.f;
    for (int c = lane; c < K / 4; c += 32) {
        const float4 v = __ldg(wr + c), a = *(const float4*) (x + c * 4);
        acc += v.x * a.x + v.y * a.y + v.z * a.z + v.w * a.w;
    }
    acc = wsum(acc);
    if (lane == 0) y[row] = acc;
}

__global__ void round_bf16_k(const float* x, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t u = __float_as_uint(x[i]);
    u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    out[i] = __uint_as_float(u);
}
// one warp per block of 32
__global__ void round_q8_k(const float* x, float* out, int n) {
    const int b = blockIdx.x * kWpb + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b * 32 >= n) return;
    const float v = x[b * 32 + lane];
    const float amax = wmax(fabsf(v));
    const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
    out[b * 32 + lane] = roundf(v * id) * __half2float(__float2half(d));
}
__global__ void add_k(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
__global__ void silu_mul_k(const float* a, const float* b, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = silu_f(a[i]) * b[i];
}
__global__ void ffn_combine_k(float* x, const float* moe, const float* sh, const float* sg, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += moe[i] + sigmoid_f(sg[0]) * sh[i];
}

// one block: softmax, then n_used rounds of block argmax
__global__ void router_k(const float* logits, int* ids, float* wts, int ne, int nu) {
    extern __shared__ float p[];
    __shared__ float sh[32], rv[32], sw[16];
    __shared__ int ri[32], si[16];
    const int tid = threadIdx.x, w = tid >> 5, l = tid & 31, nw = blockDim.x >> 5;
    float m = -FLT_MAX;
    for (int e = tid; e < ne; e += blockDim.x) m = fmaxf(m, logits[e]);
    m = bmax(m, sh);
    float s = 0.f;
    for (int e = tid; e < ne; e += blockDim.x) { p[e] = expf(logits[e] - m); s += p[e]; }
    s = bsum(s, sh);
    for (int e = tid; e < ne; e += blockDim.x) p[e] /= s;
    __syncthreads();
    for (int r = 0; r < nu; ++r) {
        float bv = -1.f;
        int bi = 0x7fffffff;
        for (int e = tid; e < ne; e += blockDim.x)
            if (p[e] > bv) { bv = p[e]; bi = e; }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (l == 0) { rv[w] = bv; ri[w] = bi; }
        __syncthreads();
        if (tid == 0) {
            float v = rv[0];
            int i = ri[0];
            for (int q = 1; q < nw; ++q)
                if (rv[q] > v || (rv[q] == v && ri[q] < i)) { v = rv[q]; i = ri[q]; }
            si[r] = i;
            sw[r] = v;
            p[i] = -1.f;
        }
        __syncthreads();
    }
    if (tid == 0) {
        float sum = 0.f;
        for (int r = 0; r < nu; ++r) sum += sw[r];
        sum = fmaxf(sum, 6.103515625e-5f); // as llama.cpp's clamp
        for (int r = 0; r < nu; ++r) { ids[r] = si[r]; wts[r] = sw[r] / sum; }
    }
}

// Q8_0: 34 bytes per block of 32 = half scale + 32 int8.  2 blocks per warp iteration, one 2-byte load (2 weights) per lane.
__device__ __forceinline__ float q8_dot16(const uint8_t* blk, const float* x, int s) { // 2 weights of the block, scale applied
    const float d = __half2float(*(const __half*) blk);
    const uint32_t q = *(const uint16_t*) (blk + 2 + 2 * s);
    const float2 xv = *(const float2*) (x + 2 * s);
    return d * ((float) (int8_t) (q & 0xffu) * xv.x + (float) (int8_t) (q >> 8) * xv.y);
}

__global__ void moe_gate_up_k(const uint8_t* gate, const uint8_t* up, size_t esz, const int* ids, const float* x, float* act, int n_ff, int nb) {
    const int k = blockIdx.y, row = blockIdx.x * kWpb + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= n_ff) return;
    const size_t off = (size_t) ids[k] * esz + (size_t) row * nb * 34;
    const uint8_t *g = gate + off, *u = up + off;
    const int half = lane >> 4, s = lane & 15;
    float ag = 0.f, au = 0.f;
#pragma unroll 4
    for (int it = 0; it < nb / 2; ++it) {
        const int b = 2 * it + half;
        ag += q8_dot16(g + b * 34, x + b * 32, s);
        au += q8_dot16(u + b * 34, x + b * 32, s);
    }
    ag = wsum(ag);
    au = wsum(au);
    if (lane == 0) act[(size_t) k * n_ff + row] = silu_f(ag) * au;
}

__global__ void moe_down_k(const uint8_t* down, size_t esz, const int* ids, const float* wts, const float* act, float* out, int nb, int nu, int n_ff) {
    const int row = blockIdx.x * kWpb + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int half = lane >> 4, s = lane & 15;
    float acc = 0.f;
    for (int k = 0; k < nu; ++k) {
        const uint8_t* d = down + (size_t) ids[k] * esz + (size_t) row * nb * 34;
        const float* a = act + (size_t) k * n_ff;
        float part = 0.f;
#pragma unroll 4
        for (int it = 0; it < nb / 2; ++it) {
            const int b = 2 * it + half;
            part += q8_dot16(d + b * 34, a + b * 32, s);
        }
        acc += wts[k] * part;
    }
    acc = wsum(acc);
    if (lane == 0) out[row] = acc;
}

// blocks 0..n_head-1: q heads (norm, rope -> q); then n_kv k heads (norm, rope -> cache[pos]); then n_kv v heads (copy)
// qfull holds [q(256) | gate(256)] per head.  NEOX rope on the first rope_dim dims: pairs (i, i + rope_dim/2), theta = pos * base^(-2i/rope_dim)
__global__ void attn_prep_k(const float* qfull, const float* kvec, const float* vvec, const float* qn, const float* kn, float* q, float* kc, float* vc,
                            int pos, int n_head, int n_kv, int rope_dim, float rope_base, float eps) {
    __shared__ float sh[32], v[kHd];
    const int b = blockIdx.x, d = threadIdx.x;
    if (b >= n_head + n_kv) { // v: straight into the cache
        const int h = b - n_head - n_kv;
        vc[((size_t) pos * n_kv + h) * kHd + d] = vvec[h * kHd + d];
        return;
    }
    const bool is_q = b < n_head;
    const float* src = is_q ? qfull + (size_t) b * 2 * kHd : kvec + (size_t)(b - n_head) * kHd;
    const float* w = is_q ? qn : kn;
    float* dst = is_q ? q + (size_t) b * kHd : kc + ((size_t) pos * n_kv + (b - n_head)) * kHd;
    const float x = src[d];
    const float ss = bsum(x * x, sh);
    v[d] = x * (1.f / sqrtf(ss / (float) kHd + eps)) * w[d];
    __syncthreads();
    const int half = rope_dim / 2;
    if (d < half) {
        const float theta = (float) pos * powf(powf(rope_base, -2.0f / (float) rope_dim), (float) d); // as ggml's rope_multi
        const float c = cosf(theta), sn = sinf(theta);
        dst[d] = v[d] * c - v[d + half] * sn;
        dst[d + half] = v[d] * sn + v[d + half] * c;
    } else if (d >= rope_dim) {
        dst[d] = v[d];
    }
}

// one block per query head, 8 warps: scores to shared memory, softmax, weighted sum of V, output gate
__global__ void attn_decode_k(const float* q, const float* qfull, const float* kc, const float* vc, float* out, int n, int nh, int nkv, float scale) {
    extern __shared__ float sc[];
    __shared__ float sh[32];
    const int h = blockIdx.x, kvh = h / (nh / nkv), tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    float qr[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) qr[t] = q[h * kHd + lane * 8 + t];
    for (int j = warp; j < n; j += 8) {
        const float4* kp = (const float4*) (kc + ((size_t) j * nkv + kvh) * kHd + lane * 8);
        const float4 a = kp[0], b = kp[1];
        float s = qr[0] * a.x + qr[1] * a.y + qr[2] * a.z + qr[3] * a.w + qr[4] * b.x + qr[5] * b.y + qr[6] * b.z + qr[7] * b.w;
        s = wsum(s);
        if (lane == 0) sc[j] = s * scale;
    }
    __syncthreads();
    float m = -FLT_MAX;
    for (int j = tid; j < n; j += blockDim.x) m = fmaxf(m, sc[j]);
    m = bmax(m, sh);
    float sum = 0.f;
    for (int j = tid; j < n; j += blockDim.x) { sc[j] = expf(sc[j] - m); sum += sc[j]; }
    sum = bsum(sum, sh); // also the barrier that publishes sc
    float o = 0.f;
    for (int j = 0; j < n; ++j) o += sc[j] * vc[((size_t) j * nkv + kvh) * kHd + tid];
    out[h * kHd + tid] = o / sum * sigmoid_f(qfull[(size_t) h * 2 * kHd + kHd + tid]);
}

// causal depthwise conv over [3 previous inputs, this one], then SiLU; the state keeps the raw inputs, oldest first
__global__ void gdn_conv_k(const float* qkv, float* st, const float* w, float* out, int n_ch) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_ch) return;
    const float s0 = st[c * 3], s1 = st[c * 3 + 1], s2 = st[c * 3 + 2], cur = qkv[c];
    const float o = s0 * w[c * 4] + s1 * w[c * 4 + 1] + s2 * w[c * 4 + 2] + cur * w[c * 4 + 3];
    st[c * 3] = s1;
    st[c * 3 + 1] = s2;
    st[c * 3 + 2] = cur;
    out[c] = silu_f(o);
}

// x / sqrt(sum(x^2) + eps), per vector (ggml's build_gdn_l2_norm)
__global__ void gdn_l2_k(float* x, int dim, float eps) {
    __shared__ float sh[32];
    float* r = x + (size_t) blockIdx.x * dim;
    float s = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) s += r[i] * r[i];
    s = bsum(s, sh);
    const float inv = 1.f / sqrtf(s + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) r[i] *= inv;
}

// grid (n_head, kS / kWpb): one warp per state row j of head h.  M[j][i] = S[i][j] as in ggml's op: decay, delta = (v - M k) * beta,
// M += delta k^T, out = M q / sqrt(S).  Value head h reads q/k head h % n_group.
__global__ void gdn_delta_k(float* S, const float* qkv, const float* alpha, const float* beta, const float* dt_bias, const float* a, float* out, int n_head, int n_group) {
    const int h = blockIdx.x, j = blockIdx.y * kWpb + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int key = n_group * kS;
    const float* q = qkv + (size_t)(h % n_group) * kS;
    const float* k = qkv + key + (size_t)(h % n_group) * kS;
    const float* v = qkv + 2 * key + (size_t) h * kS;
    const float xa = alpha[h] + dt_bias[h];
    const float decay = expf((xa > 20.f ? xa : logf(1.f + expf(xa))) * a[h]);
    const float bt = sigmoid_f(beta[h]);
    float* M = S + ((size_t) h * kS + j) * kS;
    float kk[4], qq[4], m[4], dot = 0.f;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int i = lane + 32 * t;
        kk[t] = k[i];
        qq[t] = q[i];
        m[t] = M[i] * decay;
        dot += m[t] * kk[t];
    }
    dot = wsum(dot);
    const float delta = (v[j] - dot) * bt;
    float o = 0.f;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        m[t] += delta * kk[t];
        M[lane + 32 * t] = m[t];
        o += m[t] * qq[t];
    }
    o = wsum(o);
    if (lane == 0) out[(size_t) h * kS + j] = o * (1.f / sqrtf((float) kS));
}

// per head: rms(o) * w * silu(z)  (ggml's build_norm_gated: SiLU of the gate, not a sigmoid)
__global__ void gdn_outnorm_k(float* o, const float* z, const float* w, float eps) {
    __shared__ float sh[32];
    float* r = o + (size_t) blockIdx.x * kS;
    const float* zr = z + (size_t) blockIdx.x * kS;
    const float x = r[threadIdx.x];
    const float s = bsum(x * x, sh);
    r[threadIdx.x] = x * (1.f / sqrtf(s / (float) kS + eps)) * w[threadIdx.x] * silu_f(zr[threadIdx.x]);
}

int nblk(int n, int per) { return (n + per - 1) / per; }

} // namespace

void rmsnorm(cudaStream_t s, const float* x, const float* w, float* out, int n, int rows, float eps) { rmsnorm_k<<<rows, 256, 0, s>>>(x, w, out, n, eps); }
void gemv_bf16(cudaStream_t s, const void* W, const float* x, float* y, int rows, int K) { gemv_bf16_k<<<nblk(rows, kWpb), kWpb * 32, 0, s>>>((const uint16_t*) W, x, y, rows, K); }
void gemv_f32(cudaStream_t s, const void* W, const float* x, float* y, int rows, int K) { gemv_f32_k<<<nblk(rows, kWpb), kWpb * 32, 0, s>>>((const float*) W, x, y, rows, K); }
void round_bf16(cudaStream_t s, const float* x, float* out, int n) { round_bf16_k<<<nblk(n, 256), 256, 0, s>>>(x, out, n); }
void round_q8(cudaStream_t s, const float* x, float* out, int n) { round_q8_k<<<nblk(n / 32, kWpb), kWpb * 32, 0, s>>>(x, out, n); }
void add(cudaStream_t s, float* x, const float* y, int n) { add_k<<<nblk(n, 256), 256, 0, s>>>(x, y, n); }
void silu_mul(cudaStream_t s, const float* a, const float* b, float* out, int n) { silu_mul_k<<<nblk(n, 256), 256, 0, s>>>(a, b, out, n); }
void router(cudaStream_t s, const float* logits, int* ids, float* wts, int ne, int nu) { router_k<<<1, 256, (size_t) ne * sizeof(float), s>>>(logits, ids, wts, ne, nu); }
void moe_gate_up(cudaStream_t s, const void* gate, const void* up, size_t esz, const int* ids, const float* x, float* act, int nu, int n_ff, int nb) {
    moe_gate_up_k<<<dim3(nblk(n_ff, kWpb), nu), kWpb * 32, 0, s>>>((const uint8_t*) gate, (const uint8_t*) up, esz, ids, x, act, n_ff, nb);
}
void moe_down(cudaStream_t s, const void* down, size_t esz, const int* ids, const float* wts, const float* act, float* out, int n_embd, int nb, int nu, int n_ff) {
    moe_down_k<<<nblk(n_embd, kWpb), kWpb * 32, 0, s>>>((const uint8_t*) down, esz, ids, wts, act, out, nb, nu, n_ff);
}
void ffn_combine(cudaStream_t s, float* x, const float* moe, const float* sh, const float* sg, int n) { ffn_combine_k<<<nblk(n, 256), 256, 0, s>>>(x, moe, sh, sg, n); }
void attn_prep(cudaStream_t s, const float* qfull, const float* kvec, const float* vvec, const float* qn, const float* kn, float* q, float* kc, float* vc,
               int pos, int n_head, int n_kv, int rope_dim, float rope_base, float eps) {
    attn_prep_k<<<n_head + 2 * n_kv, kHd, 0, s>>>(qfull, kvec, vvec, qn, kn, q, kc, vc, pos, n_head, n_kv, rope_dim, rope_base, eps);
}
bool attn_allow_smem(size_t bytes) { return cudaFuncSetAttribute(attn_decode_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) bytes) == cudaSuccess; }
void attn_decode(cudaStream_t s, const float* q, const float* qfull, const float* kc, const float* vc, float* out, int n_keys, int n_head, int n_kv, float scale) {
    attn_decode_k<<<n_head, kHd, (size_t) n_keys * sizeof(float), s>>>(q, qfull, kc, vc, out, n_keys, n_head, n_kv, scale);
}
void gdn_conv(cudaStream_t s, const float* qkv, float* state, const float* w, float* out, int n_ch, int) { gdn_conv_k<<<nblk(n_ch, 256), 256, 0, s>>>(qkv, state, w, out, n_ch); }
void gdn_l2(cudaStream_t s, float* qk, int n_vec, int dim, float eps) { gdn_l2_k<<<n_vec, 128, 0, s>>>(qk, dim, eps); }
void gdn_delta(cudaStream_t s, float* S, const float* qkv, const float* alpha, const float* beta, const float* dt_bias, const float* a, float* out, int n_head, int n_group) {
    gdn_delta_k<<<dim3(n_head, kS / kWpb), kWpb * 32, 0, s>>>(S, qkv, alpha, beta, dt_bias, a, out, n_head, n_group);
}
void gdn_outnorm(cudaStream_t s, float* o, const float* z, const float* w, int n_head, float eps) { gdn_outnorm_k<<<n_head, kS, 0, s>>>(o, z, w, eps); }

} // namespace strata::drama::k
