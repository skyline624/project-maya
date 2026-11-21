// src/drama/drama_model.cu - Drama 3 geometry, tensor checks and the raw upload of the weights (J1, no forward pass).
//
// Every tensor goes to VRAM in its GGUF type, one arena per part (one cudaMalloc, 256-byte aligned sub-allocations).
// The reads of a part run in their own thread through a pinned double buffer: the DMA of chunk i overlaps the page-in
// of chunk i+1 from the mmap.
#include "strata/drama/drama_model.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>

namespace strata::drama {

namespace {

constexpr uint32_t kF32 = 0, kQ8_0 = 8, kBF16 = 30; // ggml type ids
constexpr uint64_t kAlign = 256, kChunk = 64ull << 20, kCheckBytes = 4096;
enum Cat { kExperts, kDense, kTower, kHead, kNCat };
const char* const kCatName[kNCat] = {"experts", "dense", "tower", "head"};

std::string str(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
std::string str(const char* fmt, ...) {
    char b[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    return b;
}
double gib(uint64_t b) { return (double) b / (1024.0 * 1024 * 1024); }

} // namespace

// ================================ geometry ================================

bool DramaGeometry::from_gguf(const strata::GgufFile& bb, const strata::GgufFile& tw, DramaGeometry& o, std::string& err) {
    const strata::MetaValue* arch = bb.get("general.architecture");
    if (!arch || arch->s != "qwen35moe") {
        err = "backbone architecture is '" + (arch ? arch->s : std::string("?")) + "', expected 'qwen35moe'";
        return false;
    }
    const strata::MetaValue* pt = tw.get("clip.gen.audio.projector_type");
    if (!pt || pt->s != "drama3_gen") {
        err = "mmproj clip.gen.audio.projector_type is '" + (pt ? pt->s : std::string("?")) + "', expected 'drama3_gen'";
        return false;
    }
    bool ok = true;
    const auto num = [&](const strata::GgufFile& g, const char* key) -> double {
        const strata::MetaValue* v = g.get(key);
        if (!v || !v->is_num()) {
            if (ok) err = std::string("missing metadata ") + key;
            ok = false;
            return 0;
        }
        return v->num();
    };
    o = DramaGeometry();
    const auto b = [&](const char* k) { return (int) num(bb, (std::string("qwen35moe.") + k).c_str()); };
    o.n_layer = b("block_count");
    o.n_embd = b("embedding_length");
    o.n_ff_exp = b("expert_feed_forward_length");
    o.n_ff_shexp = b("expert_shared_feed_forward_length");
    o.n_expert = b("expert_count");
    o.n_expert_used = b("expert_used_count");
    o.n_head = b("attention.head_count");
    o.n_head_kv = b("attention.head_count_kv");
    o.head_dim = b("attention.key_length");
    const int value_len = b("attention.value_length");
    o.rope_dim = b("rope.dimension_count");
    o.full_attn_interval = b("full_attention_interval");
    o.rope_base = (float) num(bb, "qwen35moe.rope.freq_base");
    o.rms_eps = (float) num(bb, "qwen35moe.attention.layer_norm_rms_epsilon");
    o.ssm_conv = b("ssm.conv_kernel");
    o.ssm_state = b("ssm.state_size");
    o.ssm_groups = b("ssm.group_count");
    o.ssm_heads = b("ssm.time_step_rank");
    o.ssm_inner = b("ssm.inner_size");
    o.t_layers = (int) num(tw, "clip.gen.audio.block_count");
    o.t_ff = (int) num(tw, "clip.gen.audio.feed_forward_length");
    o.t_head = (int) num(tw, "clip.gen.audio.attention.head_count");
    o.t_head_kv = (int) num(tw, "clip.gen.audio.attention.head_count_kv");
    o.t_eps = (float) num(tw, "clip.gen.audio.attention.layer_norm_epsilon");
    const int t_embd = (int) num(tw, "clip.gen.audio.embedding_length");
    if (!ok) return false;

    // the planned kernels' limits (MMVQ n_in % 256, experts n_ff % 32 / k <= 8 / n_expert <= 512, GDN S = 128, HD = 256)
    const auto refuse = [&](bool bad, const std::string& why) {
        if (bad && err.empty()) err = why;
    };
    refuse(o.n_embd % 256 != 0, str("n_embd %d is not a multiple of 256 (dense MMVQ kernels)", o.n_embd));
    refuse(o.n_ff_exp % 32 != 0, str("expert_feed_forward_length %d is not a multiple of 32 (expert kernels)", o.n_ff_exp));
    refuse(o.n_expert_used < 1 || o.n_expert_used > 8, str("expert_used_count %d is outside 1..8 (expert kernels)", o.n_expert_used));
    refuse(o.n_expert < 1 || o.n_expert > 512, str("expert_count %d is outside 1..512 (router)", o.n_expert));
    refuse(o.n_expert_used > o.n_expert, str("expert_used_count %d > expert_count %d", o.n_expert_used, o.n_expert));
    refuse(o.ssm_state != 128, str("ssm.state_size %d, the GDN kernels need 128", o.ssm_state));
    refuse(o.head_dim != 256 || value_len != 256, str("attention key_length %d / value_length %d, the attention kernel needs 256/256", o.head_dim, value_len));
    refuse(o.ssm_heads < 1 || o.ssm_inner != o.ssm_heads * o.ssm_state, str("ssm.inner_size %d != time_step_rank %d x state_size %d", o.ssm_inner, o.ssm_heads, o.ssm_state));
    refuse(o.ssm_groups < 1 || o.ssm_heads % o.ssm_groups != 0, str("ssm.time_step_rank %d is not a multiple of group_count %d", o.ssm_heads, o.ssm_groups));
    refuse(o.n_head_kv < 1 || o.n_head % o.n_head_kv != 0, str("head_count %d is not a multiple of head_count_kv %d", o.n_head, o.n_head_kv));
    refuse(o.rope_dim < 2 || o.rope_dim > o.head_dim || o.rope_dim % 2 != 0, str("rope.dimension_count %d invalid for head dim %d", o.rope_dim, o.head_dim));
    refuse(o.n_layer < 1, str("block_count %d", o.n_layer));
    refuse(t_embd != o.n_embd, str("tower embedding_length %d != backbone %d", t_embd, o.n_embd));
    refuse(o.t_layers < 1 || o.t_head < 1 || o.t_head_kv < 1 || o.t_head % o.t_head_kv != 0, str("tower heads %d/%d, layers %d", o.t_head, o.t_head_kv, o.t_layers));
    if (!err.empty()) return false;

    // per-layer kind
    strata::MetaValue rec;
    if (!bb.get_full("qwen35moe.attention.recurrent_layers", rec) || rec.type != strata::MetaType::ARRAY || rec.items.size() != (size_t) o.n_layer) {
        err = "qwen35moe.attention.recurrent_layers missing or not one entry per layer";
        return false;
    }
    for (int i = 0; i < o.n_layer; ++i) {
        o.kind.push_back(rec.items[(size_t) i].u ? LayerKind::Gdn : LayerKind::Attn);
        // the interval key must agree with the array: a layer is full attention every `interval`-th
        if (o.full_attn_interval > 0 && ((i + 1) % o.full_attn_interval == 0) != (o.kind.back() == LayerKind::Attn)) {
            err = str("layer %d: recurrent_layers disagrees with full_attention_interval %d", i, o.full_attn_interval);
            return false;
        }
    }
    // what only the tensors tell
    const strata::TensorInfo* ow = bb.find("output.weight");
    const strata::TensorInfo* qn = tw.find("a.gen.code.blk.0.attn_q_norm.weight");
    const strata::TensorInfo* ce = tw.find("a.gen.code.embd.weight");
    const strata::TensorInfo* ch = tw.find("a.gen.code.head.weight");
    if (!ow || ow->shape.size() != 2) { err = "output.weight missing or not 2-D"; return false; }
    if (!qn || qn->shape.size() != 1) { err = "a.gen.code.blk.0.attn_q_norm.weight missing or not 1-D"; return false; }
    if (!ce || ce->shape.size() != 3 || ce->shape[1] != (uint64_t) kSemanticCount) { err = "a.gen.code.embd.weight missing or not [n_embd, 4096, n_codebooks]"; return false; }
    if (!ch || ch->shape.size() != 3 || ch->shape[1] != (uint64_t) kSemanticCount) { err = "a.gen.code.head.weight missing or not [n_embd, 4096, n_heads]"; return false; }
    o.n_vocab = (int64_t) ow->shape[1];
    o.t_head_dim = (int) qn->shape[0];
    o.t_codebooks = (int) ce->shape[2];
    o.t_resid_heads = (int) ch->shape[2];
    if (o.t_resid_heads >= o.t_layers) { err = str("tower: %d residual heads for %d layers", o.t_resid_heads, o.t_layers); return false; }
    if (o.n_vocab < (int64_t) kSemanticBegin + kSemanticCount) { err = str("n_vocab %lld is below the semantic range end %d", (long long) o.n_vocab, kSemanticBegin + kSemanticCount); return false; }
    return true;
}

std::vector<std::string> DramaGeometry::describe() const {
    int gdn = 0;
    for (auto k : kind) gdn += k == LayerKind::Gdn;
    std::vector<std::string> v;
    v.push_back(str("backbone: %d layers (%d GDN, %d attention), n_embd %d, vocab %lld, eps %g", n_layer, gdn, n_layer - gdn, n_embd, (long long) n_vocab, rms_eps));
    v.push_back(str("moe: %d experts, top-%d, n_ff_exp %d, shared n_ff %d", n_expert, n_expert_used, n_ff_exp, n_ff_shexp));
    v.push_back(str("attention: %d/%d heads, head_dim %d, rope %d dims base %g", n_head, n_head_kv, head_dim, rope_dim, rope_base));
    v.push_back(str("gdn: conv %d, state %d, groups %d, v-heads %d, inner %d, conv_dim %d", ssm_conv, ssm_state, ssm_groups, ssm_heads, ssm_inner, conv_dim()));
    v.push_back(str("tower: %d layers (%d shared + %d residual), %d/%d heads, head_dim %d, ffn %d, %d codebooks of %d, eps %g", t_layers, t_layers - t_resid_heads, t_resid_heads, t_head, t_head_kv, t_head_dim, t_ff, t_codebooks, kSemanticCount, t_eps));
    v.push_back(str("tokens: eos %d, semantic %d..%d", kEosId, kSemanticBegin, kSemanticBegin + kSemanticCount - 1));
    return v;
}

// ================================ model ================================

struct DramaModel::Impl {
    std::unique_ptr<strata::GgufFile> bb, tw;
    struct Part {
        int dev = 0, l0 = 0, l1 = 0;
        uint8_t* base = nullptr;
        uint64_t used = 0, cat[kNCat] = {};
        size_t free_after = 0, total = 0;
    };
    std::vector<Part> parts;
    struct Job { // one contiguous host range copied to one arena offset
        int part;
        uint64_t off, bytes;
        const uint8_t* src;
        std::string name;
        int n_expert; // > 0: a fused expert tensor (its slices are verified too)
    };
    std::vector<Job> jobs;
    struct Fix { DTensor* dst; int part; uint64_t off; };
    std::vector<Fix> fixes;
    std::vector<std::pair<size_t, int>> expert_checks; // job, expert
    std::vector<std::string> notes;
    double open_s = 0, upload_s = 0;
    uint64_t uploaded = 0;

    ~Impl() {
        for (auto& p : parts)
            if (p.base) { cudaSetDevice(p.dev); cudaFree(p.base); }
    }
};

DramaModel::DramaModel() : m_(new Impl) {}
DramaModel::~DramaModel() = default;

namespace {

// Pinned double buffer for one part: the copy of chunk i runs while the host fills the other buffer with chunk i+1.
struct Staging {
    uint8_t* buf[2] = {nullptr, nullptr};
    cudaEvent_t ev[2] = {nullptr, nullptr};
    cudaStream_t st = nullptr;
    ~Staging() {
        for (int i = 0; i < 2; ++i) {
            if (ev[i]) cudaEventDestroy(ev[i]);
            if (buf[i]) cudaFreeHost(buf[i]);
        }
        if (st) cudaStreamDestroy(st);
    }
    bool init(std::string& err) {
        cudaError_t e = cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
        for (int i = 0; i < 2 && e == cudaSuccess; ++i) {
            e = cudaHostAlloc((void**) &buf[i], kChunk, cudaHostAllocDefault);
            if (e == cudaSuccess) e = cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming);
        }
        if (e != cudaSuccess) err = std::string("staging buffers: ") + cudaGetErrorString(e);
        return e == cudaSuccess;
    }
};

bool upload_part(int dev, uint8_t* base, const std::vector<std::tuple<uint64_t, uint64_t, const uint8_t*>>& ranges, std::string& err) {
    cudaError_t e = cudaSetDevice(dev);
    Staging s;
    if (e != cudaSuccess) { err = std::string("cudaSetDevice: ") + cudaGetErrorString(e); return false; }
    if (!s.init(err)) return false;
    uint64_t i = 0;
    for (const auto& [off, bytes, src] : ranges)
        for (uint64_t o = 0; o < bytes; o += kChunk, ++i) {
            const uint64_t n = std::min(kChunk, bytes - o);
            const int b = (int) (i & 1);
            // buffer b is free once the copy that last read it is done (an event never recorded is complete)
            if ((e = cudaEventSynchronize(s.ev[b])) != cudaSuccess) break;
            std::memcpy(s.buf[b], src + o, (size_t) n);
            if ((e = cudaMemcpyAsync(base + off + o, s.buf[b], (size_t) n, cudaMemcpyHostToDevice, s.st)) != cudaSuccess) break;
            if ((e = cudaEventRecord(s.ev[b], s.st)) != cudaSuccess) break;
        }
    if (e == cudaSuccess) e = cudaStreamSynchronize(s.st);
    if (e != cudaSuccess) { err = std::string("upload: ") + cudaGetErrorString(e); return false; }
    return true;
}

} // namespace

bool DramaModel::load(const std::string& backbone_path, const std::string& mmproj_path, std::vector<int> devs, std::vector<int> bounds, std::string& err) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    Impl& m = *m_;
    try {
        m.bb.reset(new strata::GgufFile(backbone_path));
        m.tw.reset(new strata::GgufFile(mmproj_path));
        const strata::GgufFile &bb = *m.bb, &tw = *m.tw;
        if (!DramaGeometry::from_gguf(bb, tw, g_, err)) return false;
        const DramaGeometry& g = g_;

        // semantic ids are not in the metadata: the tokenizer must agree with the constants
        strata::MetaValue toks;
        if (!bb.get_full("tokenizer.ggml.tokens", toks) || toks.items.size() < (size_t) kSemanticBegin + kSemanticCount) {
            err = "tokenizer.ggml.tokens missing or shorter than the semantic range end " + std::to_string(kSemanticBegin + kSemanticCount);
            return false;
        }
        const std::string &t_first = toks.items[kSemanticBegin].s, &t_last = toks.items[kSemanticBegin + kSemanticCount - 1].s, &t_eos = toks.items[kEosId].s;
        if (t_first.find("semantic:0|>") == std::string::npos || t_last.find("semantic:4095|>") == std::string::npos) {
            err = "tokens[" + std::to_string(kSemanticBegin) + "] is '" + t_first + "' and tokens[" + std::to_string(kSemanticBegin + kSemanticCount - 1) + "] is '" + t_last + "', not <|semantic:0|> .. <|semantic:4095|>";
            return false;
        }
        if (const strata::MetaValue* eos = bb.get("tokenizer.ggml.eos_token_id"); eos && eos->u != (uint64_t) kEosId) {
            err = "tokenizer.ggml.eos_token_id is " + std::to_string(eos->u) + ", expected " + std::to_string(kEosId);
            return false;
        }
        m.notes.push_back("tokens: eos '" + t_eos + "', tokens[" + std::to_string(kSemanticBegin) + "] = '" + t_first + "', last '" + t_last + "'");

        // devices and the layer split
        const int n = (int) devs.size();
        if (n < 1) { err = "no device given"; return false; }
        if (bounds.empty())
            for (int i = 1; i < n; ++i) bounds.push_back(i * g.n_layer / n);
        if ((int) bounds.size() != n - 1) { err = std::to_string(n) + " devices need " + std::to_string(n - 1) + " layer boundaries, got " + std::to_string(bounds.size()); return false; }
        for (int i = 0; i < n - 1; ++i)
            if (bounds[(size_t) i] <= (i ? bounds[(size_t) i - 1] : 0) || bounds[(size_t) i] >= g.n_layer) { err = "layer boundaries must be strictly increasing within (0, " + std::to_string(g.n_layer) + ")"; return false; }
        m.parts.resize((size_t) n);
        for (int p = 0; p < n; ++p) {
            m.parts[(size_t) p].dev = devs[(size_t) p];
            m.parts[(size_t) p].l0 = p ? bounds[(size_t) p - 1] : 0;
            m.parts[(size_t) p].l1 = p + 1 < n ? bounds[(size_t) p] : g.n_layer;
        }
        const auto part_of = [&](int il) { int p = 0; while (p < n - 1 && il >= bounds[(size_t) p]) ++p; return p; };

        // ---- plan: check every tensor, give it an arena offset ----
        uint64_t checked = 0;
        const auto chk = [&](const strata::GgufFile& f, const std::string& name, uint32_t type, std::vector<uint64_t> shape) -> const strata::TensorInfo& {
            const strata::TensorInfo* t = f.find(name);
            if (!t) throw std::runtime_error("tensor " + name + " is missing");
            if (t->type != type) throw std::runtime_error("tensor " + name + " is " + t->type_name() + ", expected " + strata::ggml_type_name(type));
            if (t->shape != shape) {
                std::string a, b;
                for (auto d : t->shape) a += (a.empty() ? "" : ",") + std::to_string(d);
                for (auto d : shape) b += (b.empty() ? "" : ",") + std::to_string(d);
                throw std::runtime_error("tensor " + name + " has shape [" + a + "], expected [" + b + "]");
            }
            int el = 0, by = 0;
            if (!strata::block_geometry(type, el, by) || t->shape[0] % (uint64_t) el) throw std::runtime_error("tensor " + name + ": row length not a whole number of blocks");
            const uint64_t bytes = t->elements() / (uint64_t) el * (uint64_t) by;
            if (f.data_start() + t->offset + bytes > f.file_size())
                throw std::runtime_error("tensor " + name + " ends past the end of the file (" + std::to_string(f.data_start() + t->offset + bytes) + " > " + std::to_string(f.file_size()) + ")");
            ++checked;
            return *t;
        };
        const auto bytes_of = [&](const strata::TensorInfo& t) {
            int el = 0, by = 0;
            strata::block_geometry(t.type, el, by);
            return t.elements() / (uint64_t) el * (uint64_t) by;
        };
        const auto alloc = [&](int part, Cat cat, uint64_t bytes) {
            Impl::Part& p = m.parts[(size_t) part];
            const uint64_t off = (p.used + kAlign - 1) / kAlign * kAlign;
            p.used = off + bytes;
            p.cat[cat] += bytes;
            return off;
        };
        const auto add = [&](const strata::GgufFile& f, int part, Cat cat, const std::string& name, uint32_t type, std::vector<uint64_t> shape, DTensor& dst, int n_expert = 0) {
            const strata::TensorInfo& t = chk(f, name, type, std::move(shape));
            const uint64_t bytes = bytes_of(t), off = alloc(part, cat, bytes);
            dst.bytes = bytes;
            m.fixes.push_back({&dst, part, off});
            m.jobs.push_back({part, off, bytes, f.tensor_data(t), name, n_expert});
        };

        const uint64_t E = (uint64_t) g.n_embd, FE = (uint64_t) g.n_ff_exp, FS = (uint64_t) g.n_ff_shexp, NE = (uint64_t) g.n_expert;
        const uint64_t HD = (uint64_t) g.head_dim, NH = (uint64_t) g.n_head, NKV = (uint64_t) g.n_head_kv;
        const uint64_t INNER = (uint64_t) g.ssm_inner, CONV = (uint64_t) g.conv_dim(), SH = (uint64_t) g.ssm_heads, SS = (uint64_t) g.ssm_state, KC = (uint64_t) g.ssm_conv;
        layers_.assign((size_t) g.n_layer, DramaLayer());
        for (int il = 0; il < g.n_layer; ++il) {
            DramaLayer& L = layers_[(size_t) il];
            const int part = part_of(il);
            L.kind = g.kind[(size_t) il];
            L.part = part;
            L.dev = devs[(size_t) part];
            const auto B = [&](const char* s) { return "blk." + std::to_string(il) + "." + s; };
            const auto D = [&](const char* s, uint32_t type, std::vector<uint64_t> shape, DTensor& dst) { add(bb, part, kDense, B(s), type, std::move(shape), dst); };
            D("attn_norm.weight", kF32, {E}, L.attn_norm);
            D("post_attention_norm.weight", kF32, {E}, L.post_norm);
            D("ffn_gate_inp.weight", kF32, {E, NE}, L.gate_inp);
            D("ffn_gate_shexp.weight", kBF16, {E, FS}, L.shexp_gate);
            D("ffn_up_shexp.weight", kBF16, {E, FS}, L.shexp_up);
            D("ffn_down_shexp.weight", kBF16, {FS, E}, L.shexp_down);
            D("ffn_gate_inp_shexp.weight", kF32, {E, 1}, L.shexp_inp);
            add(bb, part, kExperts, B("ffn_gate_exps.weight"), kQ8_0, {E, FE, NE}, L.exps_gate, g.n_expert);
            add(bb, part, kExperts, B("ffn_up_exps.weight"), kQ8_0, {E, FE, NE}, L.exps_up, g.n_expert);
            add(bb, part, kExperts, B("ffn_down_exps.weight"), kQ8_0, {FE, E, NE}, L.exps_down, g.n_expert);
            if (L.kind == LayerKind::Gdn) {
                D("attn_qkv.weight", kBF16, {E, CONV}, L.qkv);
                D("attn_gate.weight", kBF16, {E, INNER}, L.gate);
                D("ssm_alpha.weight", kBF16, {E, SH}, L.alpha);
                D("ssm_beta.weight", kBF16, {E, SH}, L.beta);
                D("ssm_a", kF32, {SH}, L.a);
                D("ssm_dt.bias", kF32, {SH}, L.dt_bias);
                D("ssm_conv1d.weight", kF32, {KC, CONV}, L.conv1d);
                D("ssm_norm.weight", kF32, {SS}, L.ssm_norm);
                D("ssm_out.weight", kBF16, {INNER, E}, L.ssm_out);
            } else {
                D("attn_q.weight", kBF16, {E, NH * HD * 2}, L.q);
                D("attn_k.weight", kBF16, {E, NKV * HD}, L.k);
                D("attn_v.weight", kBF16, {E, NKV * HD}, L.v);
                D("attn_output.weight", kBF16, {NH * HD, E}, L.o);
                D("attn_q_norm.weight", kF32, {HD}, L.q_norm);
                D("attn_k_norm.weight", kF32, {HD}, L.k_norm);
            }
        }
        // the lm_head cut to [eos, sem0..sem4095]: one row of 2*n_embd bytes each, the semantic ones contiguous
        {
            const int last = n - 1;
            add(bb, last, kHead, "output_norm.weight", kF32, {E}, head_.output_norm);
            const strata::TensorInfo& ow = chk(bb, "output.weight", kBF16, {E, (uint64_t) g.n_vocab});
            const uint64_t row = E * 2, rows = 1 + (uint64_t) kSemanticCount, off = alloc(last, kHead, rows * row);
            const uint8_t* src = bb.tensor_data(ow);
            head_.lm_head.bytes = rows * row;
            m.fixes.push_back({&head_.lm_head, last, off});
            m.jobs.push_back({last, off, row, src + (uint64_t) kEosId * row, "output.weight[eos]", 0});
            m.jobs.push_back({last, off + row, (uint64_t) kSemanticCount * row, src + (uint64_t) kSemanticBegin * row, "output.weight[semantic]", 0});
            head_.dev = devs[(size_t) last];
            head_.ids.assign(1, kEosId);
            for (int i = 0; i < kSemanticCount; ++i) head_.ids.push_back(kSemanticBegin + i);
            // token_embd stays in the host mmap
            embd_host_ = bb.tensor_data(chk(bb, "token_embd.weight", kBF16, {E, (uint64_t) g.n_vocab}));
        }
        // the tower, whole, on part 0
        {
            tower_.dev = devs[0];
            const uint64_t CB = (uint64_t) kSemanticCount, TH = (uint64_t) g.t_head, TKV = (uint64_t) g.t_head_kv, THD = (uint64_t) g.t_head_dim, TF = (uint64_t) g.t_ff;
            const auto T = [&](const char* s, uint32_t type, std::vector<uint64_t> shape, DTensor& dst) { add(tw, 0, kTower, std::string("a.gen.code.") + s, type, std::move(shape), dst); };
            T("out_embd.weight", kBF16, {E, CB}, tower_.out_embd);
            T("embd.weight", kBF16, {E, CB, (uint64_t) g.t_codebooks}, tower_.embd);
            T("head.weight", kBF16, {E, CB, (uint64_t) g.t_resid_heads}, tower_.head);
            tower_.blk.assign((size_t) g.t_layers, DramaTowerLayer());
            for (int il = 0; il < g.t_layers; ++il) {
                DramaTowerLayer& L = tower_.blk[(size_t) il];
                const auto X = [&](const char* s, uint32_t type, std::vector<uint64_t> shape, DTensor& dst) { T(("blk." + std::to_string(il) + "." + s).c_str(), type, std::move(shape), dst); };
                X("embd.weight", kBF16, {E, CB}, L.embd);
                X("eh_proj.weight", kBF16, {2 * E, E}, L.eh_proj);
                X("enorm.weight", kF32, {E}, L.enorm);
                X("hnorm.weight", kF32, {E}, L.hnorm);
                X("out_norm.weight", kF32, {E}, L.out_norm);
                X("ln1.weight", kF32, {E}, L.ln1);
                X("ln2.weight", kF32, {E}, L.ln2);
                X("attn_q_norm.weight", kF32, {THD}, L.q_norm);
                X("attn_k_norm.weight", kF32, {THD}, L.k_norm);
                X("attn_q.weight", kBF16, {E, TH * THD}, L.q);
                X("attn_gate.weight", kBF16, {E, TH * THD}, L.gate);
                X("attn_k.weight", kBF16, {E, TKV * THD}, L.k);
                X("attn_v.weight", kBF16, {E, TKV * THD}, L.v);
                X("attn_out.weight", kBF16, {TH * THD, E}, L.out);
                X("ffn_gate.weight", kBF16, {E, TF}, L.ffn_gate);
                X("ffn_up.weight", kBF16, {E, TF}, L.ffn_up);
                X("ffn_down.weight", kBF16, {TF, E}, L.ffn_down);
            }
        }
        if (checked < bb.tensors().size()) m.notes.push_back(std::to_string(bb.tensors().size() - checked) + " backbone tensors are not used");
        // one random expert per layer, for verify()
        std::mt19937 rng(1);
        for (size_t j = 0; j < m.jobs.size(); ++j)
            if (m.jobs[j].n_expert && m.jobs[j].name.find("ffn_gate_exps") != std::string::npos) {
                const int e = (int) (rng() % (uint32_t) g.n_expert);
                for (size_t k = j; k < j + 3; ++k) m.expert_checks.push_back({k, e}); // gate, up, down are consecutive
            }

        // ---- arenas (the device check comes after the plan: a dry run with CUDA_VISIBLE_DEVICES= checks every tensor without touching a GPU) ----
        int n_dev = 0;
        if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) { err = "no CUDA device"; return false; }
        for (int d : devs)
            if (d < 0 || d >= n_dev) { err = "device " + std::to_string(d) + " out of range (" + std::to_string(n_dev) + " visible)"; return false; }
        for (auto& p : m.parts) {
            size_t fr = 0, tot = 0;
            cudaError_t e = cudaSetDevice(p.dev);
            if (e == cudaSuccess) e = cudaMemGetInfo(&fr, &tot);
            if (e != cudaSuccess) { err = std::string("cudaMemGetInfo: ") + cudaGetErrorString(e); return false; }
            // + 2 staging buffers of one chunk and the CUDA context
            if (p.used + 2 * kChunk + (512ull << 20) > fr) {
                err = str("device %d: the part needs %.2f GiB (layers %d-%d) but only %.2f GiB of %.2f are free", p.dev, gib(p.used), p.l0, p.l1 - 1, gib(fr), gib(tot));
                return false;
            }
            if ((e = cudaMalloc((void**) &p.base, p.used)) != cudaSuccess) { p.base = nullptr; err = str("device %d: cudaMalloc of %.2f GiB: %s", p.dev, gib(p.used), cudaGetErrorString(e)); return false; }
        }
        for (const auto& f : m.fixes) f.dst->p = m.parts[(size_t) f.part].base + f.off;
        const auto t1 = clk::now();
        m.open_s = std::chrono::duration<double>(t1 - t0).count();

        // ---- upload: one thread per part ----
        std::vector<std::string> errs((size_t) n);
        std::vector<char> ok((size_t) n, 0);
        std::vector<std::thread> th;
        for (int p = 0; p < n; ++p)
            th.emplace_back([&, p] {
                std::vector<std::tuple<uint64_t, uint64_t, const uint8_t*>> ranges;
                for (const auto& j : m.jobs)
                    if (j.part == p) ranges.emplace_back(j.off, j.bytes, j.src);
                ok[(size_t) p] = upload_part(m.parts[(size_t) p].dev, m.parts[(size_t) p].base, ranges, errs[(size_t) p]);
            });
        for (auto& t : th) t.join();
        for (int p = 0; p < n; ++p)
            if (!ok[(size_t) p]) { err = "device " + std::to_string(m.parts[(size_t) p].dev) + ": " + errs[(size_t) p]; return false; }
        m.upload_s = std::chrono::duration<double>(clk::now() - t1).count();
        for (const auto& j : m.jobs) m.uploaded += j.bytes;
        for (auto& p : m.parts) {
            cudaSetDevice(p.dev);
            cudaMemGetInfo(&p.free_after, &p.total);
        }
    } catch (const std::exception& ex) {
        err = ex.what();
        return false;
    }
    return true;
}

bool DramaModel::verify(std::string& err) {
    Impl& m = *m_;
    std::vector<uint8_t> host(kCheckBytes);
    uint64_t regions = 0;
    const auto cmp = [&](const Impl::Job& j, uint64_t rel, uint64_t n, const char* what) {
        const Impl::Part& p = m.parts[(size_t) j.part];
        cudaError_t e = cudaSetDevice(p.dev);
        if (e == cudaSuccess) e = cudaMemcpy(host.data(), p.base + j.off + rel, (size_t) n, cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) { err = "verify " + j.name + ": " + cudaGetErrorString(e); return false; }
        if (std::memcmp(host.data(), j.src + rel, (size_t) n) != 0) { err = "verify: " + j.name + " " + what + " differs from the file (device " + std::to_string(p.dev) + ")"; return false; }
        ++regions;
        return true;
    };
    for (const auto& j : m.jobs) {
        const uint64_t n = std::min(kCheckBytes, j.bytes);
        if (!cmp(j, 0, n, "head") || !cmp(j, j.bytes - n, n, "tail")) return false;
    }
    for (const auto& [k, e] : m.expert_checks) {
        const Impl::Job& j = m.jobs[k];
        const uint64_t slice = j.bytes / (uint64_t) j.n_expert, n = std::min(kCheckBytes, slice);
        if (!cmp(j, (uint64_t) e * slice, n, "expert head") || !cmp(j, (uint64_t) e * slice + slice - n, n, "expert tail")) return false;
    }
    m.notes.push_back(str("verify: %llu regions read back from the devices, all equal to the file", (unsigned long long) regions));
    return true;
}

std::vector<std::string> DramaModel::report() const {
    const Impl& m = *m_;
    std::vector<std::string> v = m.notes;
    for (size_t p = 0; p < m.parts.size(); ++p) {
        const Impl::Part& q = m.parts[p];
        std::string s = str("dev %d layers %d-%d%s%s:", q.dev, q.l0, q.l1 - 1, p == 0 ? " +tower" : "", p + 1 == m.parts.size() ? " +head" : "");
        uint64_t sum = 0;
        for (int c = 0; c < kNCat; ++c) { s += str(" %s %.2f GiB,", kCatName[c], gib(q.cat[c])); sum += q.cat[c]; }
        v.push_back(s + str(" other %.2f GiB (alignment); arena %.2f GiB; free %.2f / total %.2f GiB", gib(q.used - sum), gib(q.used), gib(q.free_after), gib(q.total)));
    }
    v.push_back(str("load: %.2f GiB uploaded in %.1f s (%.2f GiB/s), %.1f s to open and check the files", gib(m.uploaded), m.upload_s, m.upload_s > 0 ? gib(m.uploaded) / m.upload_s : 0.0, m.open_s));
    return v;
}

} // namespace strata::drama
