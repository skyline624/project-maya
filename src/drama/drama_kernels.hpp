// src/drama/drama_kernels.hpp - Drama 3 backbone kernels, one row at a time (J2).  Internal to src/drama (includes CUDA).
//
// f32 activations everywhere; BF16 / F32 weights read as stored, Q8_0 expert blocks dequantized in the dot product.
// Row-major weights as in the GGUF: W[r][k], k contiguous.  Every launcher enqueues on `s` and checks nothing (the caller
// reads cudaGetLastError at its sync points).
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strata::drama::k {

// out[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * w   for `rows` independent vectors of n (in place allowed)
void rmsnorm(cudaStream_t s, const float* x, const float* w, float* out, int n, int rows, float eps);
// y[r] = W[r] . x  (BF16 rows: K % 8 == 0 and 16-byte aligned; F32 rows: K % 4 == 0)
void gemv_bf16(cudaStream_t s, const void* W, const float* x, float* y, int rows, int K);
void gemv_f32(cudaStream_t s, const void* W, const float* x, float* y, int rows, int K);
// DRAMA_EMU=1: the reference's activation roundings (llama.cpp feeds BF16 weights with BF16 activations and Q8_0 experts with Q8_0-quantized ones)
void round_bf16(cudaStream_t s, const float* x, float* out, int n); // round to nearest even
void round_q8(cudaStream_t s, const float* x, float* out, int n);   // blocks of 32: scale amax/127 (as f16), round(x/scale) * scale
void add(cudaStream_t s, float* x, const float* y, int n);                     // x += y
void silu_mul(cudaStream_t s, const float* a, const float* b, float* out, int n); // silu(a) * b

// softmax over all experts, top `n_used` (lowest index on ties), renormalized to sum 1
void router(cudaStream_t s, const float* logits, int* ids, float* wts, int n_expert, int n_used);
// act[k][r] = silu(gate_e . x) * (up_e . x) for the experts ids[k]; rows of nb Q8_0 blocks (nb even); esz = bytes of one expert
void moe_gate_up(cudaStream_t s, const void* gate, const void* up, size_t esz, const int* ids, const float* x, float* act, int n_used, int n_ff, int nb);
// out[r] = sum_k wts[k] * (down_ids[k] . act[k])
void moe_down(cudaStream_t s, const void* down, size_t esz, const int* ids, const float* wts, const float* act, float* out, int n_embd, int nb, int n_used, int n_ff);
void ffn_combine(cudaStream_t s, float* x, const float* moe, const float* sh, const float* sg, int n); // x += moe + sigmoid(sg) * sh

// full attention, one query row at position `pos`
void attn_prep(cudaStream_t s, const float* qfull, const float* kvec, const float* vvec, const float* qn, const float* kn, float* q, float* kc, float* vc,
               int pos, int n_head, int n_kv, int rope_dim, float rope_base, float eps);
bool attn_allow_smem(size_t bytes); // raises the current device's dynamic shared-memory limit of attn_decode (> 48 KiB scores)
void attn_decode(cudaStream_t s, const float* q, const float* qfull, const float* kc, const float* vc, float* out, int n_keys, int n_head, int n_kv, float scale);

// gated delta-net, one row.  Head pairing is tiled (value head h reads q/k head h % n_group), as ggml's op does.
void gdn_conv(cudaStream_t s, const float* qkv, float* state, const float* w, float* out, int n_ch, int kc); // kc == 4
void gdn_l2(cudaStream_t s, float* qk, int n_vec, int dim, float eps);                                      // in place, n_vec rows of dim
void gdn_delta(cudaStream_t s, float* S, const float* qkv, const float* alpha, const float* beta, const float* dt_bias, const float* a, float* out,
               int n_head, int n_group);
void gdn_outnorm(cudaStream_t s, float* o, const float* z, const float* w, int n_head, float eps);        // rms(o) * w * silu(z), in place

} // namespace strata::drama::k
