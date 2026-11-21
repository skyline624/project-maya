// include/strata/drama/drama_model.hpp - Drama 3 (Fish Audio TTS): geometry and device weights.
//
// J1 of docs/plan-drama3-maya.md: the backbone GGUF (qwen35moe, 40 layers, 256 Q8_0 experts) and the codebook tower
// (mmproj GGUF, a.gen.code.*) are read, checked and uploaded RAW (BF16 / F32 / Q8_0, no conversion), split over the
// GPUs by layer.  No forward pass yet.  Device pointers are `const void*` so this header needs no CUDA include.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "strata/artifact/gguf_reader.hpp"

namespace strata::drama {

constexpr int kEosId = 248046;         // <|im_end|>
constexpr int kSemanticBegin = 248099; // <|semantic:0|> (config.json semantic_start_token_id; not in the GGUF metadata)
constexpr int kSemanticCount = 4096;

enum class LayerKind : uint8_t { Gdn, Attn };

struct DramaGeometry {
    // backbone (qwen35moe.*)
    int n_layer = 0, n_embd = 0, n_ff_exp = 0, n_ff_shexp = 0, n_expert = 0, n_expert_used = 0;
    int n_head = 0, n_head_kv = 0, head_dim = 0, rope_dim = 0, full_attn_interval = 0;
    float rope_base = 0, rms_eps = 0;
    int ssm_conv = 0, ssm_state = 0, ssm_groups = 0, ssm_heads = 0, ssm_inner = 0;
    int64_t n_vocab = 0;
    std::vector<LayerKind> kind; // per layer, from attention.recurrent_layers
    // tower (clip.gen.audio.*, a.gen.code.*)
    int t_layers = 0, t_ff = 0, t_head = 0, t_head_kv = 0, t_head_dim = 0;
    int t_codebooks = 0;  // embd.weight [n_embd, 4096, t_codebooks]: code 0 is the semantic one
    int t_resid_heads = 0; // head.weight [n_embd, 4096, t_resid_heads]: the last t_resid_heads blocks sample one each
    float t_eps = 0;

    int conv_dim() const { return ssm_inner + 2 * ssm_groups * ssm_state; }
    // reads both files and refuses what the planned kernels cannot do, with a precise error
    static bool from_gguf(const strata::GgufFile& bb, const strata::GgufFile& tw, DramaGeometry& out, std::string& err);
    std::vector<std::string> describe() const;
};

struct DTensor {       // a tensor in a device arena, bytes as in the GGUF
    const void* p = nullptr;
    uint64_t bytes = 0;
};
// expert `e` of a fused [.., .., n_expert] tensor
inline const void* expert_ptr(const DTensor& t, int e, int n_expert) {
    return (const uint8_t*) t.p + (uint64_t) e * (t.bytes / (uint64_t) n_expert);
}

struct DramaLayer {
    LayerKind kind = LayerKind::Gdn;
    int part = 0, dev = 0;
    DTensor attn_norm, post_norm;                                        // F32 [n_embd]
    DTensor gate_inp;                                                    // F32 [n_embd, n_expert]
    DTensor shexp_gate, shexp_up, shexp_down, shexp_inp;                 // BF16; shexp_inp F32 [n_embd, 1]
    DTensor exps_gate, exps_up, exps_down;                               // Q8_0, fused over the experts
    DTensor qkv, gate, alpha, beta, a, dt_bias, conv1d, ssm_norm, ssm_out; // GDN
    DTensor q, k, v, o, q_norm, k_norm;                                  // full attention (q holds q and gate per head)
};

struct DramaTowerLayer {
    DTensor embd, eh_proj, enorm, hnorm, out_norm, ln1, ln2, q_norm, k_norm, q, gate, k, v, out, ffn_gate, ffn_up, ffn_down;
};
struct DramaTower { // part 0
    int dev = 0;
    DTensor out_embd, embd, head;
    std::vector<DramaTowerLayer> blk;
};
struct DramaHead { // last part: output_norm and the lm_head cut to rows [eos, sem_begin .. sem_begin+4095]
    int dev = 0;
    DTensor output_norm, lm_head;
    std::vector<int> ids; // vocab id of each lm_head row
};

class DramaModel {
public:
    DramaModel();
    ~DramaModel();
    DramaModel(const DramaModel&) = delete;
    DramaModel& operator=(const DramaModel&) = delete;

    // devs: one CUDA device per part; bounds: first layer of each later part (empty with 2+ devs: an even split)
    bool load(const std::string& backbone_path, const std::string& mmproj_path, std::vector<int> devs,
              std::vector<int> bounds, std::string& err);
    // first and last 4 KiB of every uploaded tensor, and of one random expert's slices per layer, read back and memcmp'd
    bool verify(std::string& err);
    std::vector<std::string> report() const; // VRAM by category per device, load time

    const DramaGeometry& geometry() const { return g_; }
    const std::vector<DramaLayer>& layers() const { return layers_; }
    const DramaTower& tower() const { return tower_; }
    const DramaHead& head() const { return head_; }
    // row of token_embd.weight (BF16 [n_embd]) in the host mmap: embeddings are gathered from there
    const uint8_t* token_embd_row(int id) const { return embd_host_ + (uint64_t) id * (uint64_t) g_.n_embd * 2; }

private:
    struct Impl;
    std::unique_ptr<Impl> m_;
    DramaGeometry g_;
    std::vector<DramaLayer> layers_;
    DramaTower tower_;
    DramaHead head_;
    const uint8_t* embd_host_ = nullptr;
};

} // namespace strata::drama
