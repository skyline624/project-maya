// src/drama/drama_sampling.hpp - Drama 3 token sampling, shared by the host (main token, parity checks) and the device (codebook heads).  Internal to src/drama.
//
// Two policies (docs/plan-drama3-maya.md 9.4): `official` follows SGLang (sglang_omni/models/fishaudio_drama3/sampling*.py), `llama` follows
// llama.cpp (mtmd-helper-gen.cpp drama3_gen_audio_pipeline::draw / sample_main, models/qwen3tts-gen.cpp code_gen::do_sampling).
// Candidates are sorted by (logit descending, id ascending); the kept set is a prefix of that order, then the draw is proportional to the weights.
#pragma once

#include <cstdint>

#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

#ifdef __CUDACC__
#define DRAMA_HD __host__ __device__
#else
#define DRAMA_HD
#endif

namespace strata::drama {

enum class SamplingMode { Official, Llama };

constexpr int kRasWindow = 16;
constexpr int kResidualCodes = 1024; // codes the codec's residual codebooks hold; the heads are 4096 wide

struct SampleSpec {
    float temp = 1.f;
    int top_k = 0;            // 0: no limit
    float top_p = 1.f;
    bool crossing = true;     // keep the token that crosses top_p (SGLang main token, llama.cpp both); false: drop it (SGLang codebook heads)
    bool renorm_topk = false; // top_p measured on the probabilities renormalized over the top-k set (llama.cpp's codebook sampler); false: over all candidates
    int legal = 0;            // > 0: only ids < legal can be drawn (SGLang reference sampling)
};

// main token: temperature 1, top_p 0.9, no top_k
inline SampleSpec main_spec(float temp = 1.f, float top_p = 0.9f) {
    SampleSpec s;
    s.temp = temp;
    s.top_p = top_p;
    return s;
}
// repetition-aware redraw (RAS) of the main token
inline SampleSpec ras_spec() {
    SampleSpec s;
    s.top_k = 30;
    s.top_p = 0.95f;
    return s;
}
inline SampleSpec head_spec(SamplingMode m, float temp, int top_k, float top_p) {
    SampleSpec s;
    s.temp = temp;
    s.top_k = top_k;
    s.top_p = top_p;
    if (m == SamplingMode::Llama) {
        s.crossing = true;
        s.renorm_topk = true;
    } else {
        s.crossing = false;
        s.legal = kResidualCodes;
    }
    return s;
}

// Philox4x32-10 uniform in [0, 1), keyed by (seed, stream, counter): no state, the same on host and device
DRAMA_HD inline float philox_uniform(uint64_t seed, uint32_t stream, uint32_t ctr) {
    uint32_t c0 = ctr, c1 = stream, c2 = 0, c3 = 0, k0 = (uint32_t) seed, k1 = (uint32_t) (seed >> 32);
    for (int r = 0; r < 10; ++r) {
        const uint64_t p0 = (uint64_t) 0xD2511F53u * c0, p1 = (uint64_t) 0xCD9E8D57u * c2;
        const uint32_t n0 = (uint32_t) (p1 >> 32) ^ c1 ^ k0, n1 = (uint32_t) p1, n2 = (uint32_t) (p0 >> 32) ^ c3 ^ k1, n3 = (uint32_t) p0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

// w[i] = exp((logit_i - logit_0) / temp) of the n sorted candidates, z = sum of all w.  Length of the kept prefix.
DRAMA_HD inline int kept_prefix(const float* w, int n, double z, const SampleSpec& sp) {
    const int lim = (sp.top_k > 0 && sp.top_k < n) ? sp.top_k : n;
    double denom = z;
    if (sp.renorm_topk) {
        denom = 0;
        for (int i = 0; i < lim; ++i) denom += w[i];
    }
    const double cap = (double) sp.top_p * denom;
    double cum = 0;
    int m = 0;
    for (; m < lim; ++m) {
        const double before = cum;
        cum += w[m];
        if (m > 0 && !(sp.crossing ? before <= cap : cum <= cap)) break;
    }
    return m;
}

// index (in the sorted order) of the drawn candidate; ids[i] is the id of sorted candidate i, u a uniform in [0, 1)
DRAMA_HD inline int draw_sorted(const float* w, const int* ids, int n, double z, const SampleSpec& sp, float u) {
    const int m = kept_prefix(w, n, z, sp);
    double tot = 0;
    for (int i = 0; i < m; ++i)
        if (sp.legal <= 0 || ids[i] < sp.legal) tot += w[i];
    if (tot <= 0) { // nothing legal kept: the best legal candidate
        for (int i = 0; i < n; ++i)
            if (sp.legal <= 0 || ids[i] < sp.legal) return i;
        return 0;
    }
    double r = (double) u * tot;
    int last = 0;
    for (int i = 0; i < m; ++i) {
        if (sp.legal > 0 && ids[i] >= sp.legal) continue;
        last = i;
        r -= w[i];
        if (r < 0) return i;
    }
    return last;
}

// ---- host side ---------------------------------------------------------------------------------------------------------------------------

struct Sorted {
    std::vector<float> w;
    std::vector<int> ids;
    double z = 0;
};

// candidates (logits[i], ids[i]) sorted by (logit descending, id ascending); weights at the temperature of `sp`
inline Sorted sort_candidates(const std::vector<float>& logits, const std::vector<int>& ids, float temp) {
    const size_t n = logits.size();
    std::vector<size_t> o(n);
    for (size_t i = 0; i < n; ++i) o[i] = i;
    std::sort(o.begin(), o.end(), [&](size_t a, size_t b) { return logits[a] != logits[b] ? logits[a] > logits[b] : ids[a] < ids[b]; });
    Sorted s;
    s.w.resize(n);
    s.ids.resize(n);
    const float top = n ? logits[o[0]] : 0.f, t = temp > 1e-5f ? temp : 1e-5f;
    for (size_t i = 0; i < n; ++i) {
        s.ids[i] = ids[o[i]];
        s.w[i] = std::exp((logits[o[i]] - top) / t);
        s.z += s.w[i];
    }
    return s;
}

// ids a draw with `sp` can return
inline std::vector<int> nucleus_ids(const Sorted& s, const SampleSpec& sp) {
    const int n = (int) s.w.size(), m = kept_prefix(s.w.data(), n, s.z, sp);
    std::vector<int> out;
    for (int i = 0; i < m; ++i)
        if (sp.legal <= 0 || s.ids[(size_t) i] < sp.legal) out.push_back(s.ids[(size_t) i]);
    if (out.empty())
        for (int i = 0; i < n; ++i)
            if (sp.legal <= 0 || s.ids[(size_t) i] < sp.legal) { out.push_back(s.ids[(size_t) i]); break; }
    return out;
}

// Main token of a frame.  Logits are the 4097 of the backbone head: position 0 = eos, position j >= 1 = <|semantic:j-1|>.
// `official` draws among eos and all 4096 semantic codes, `llama` leaves code 0 out (the reference never samples it).  A semantic first draw that
// repeats one of the last 16 main tokens is redrawn with RAS (temperature 1, top_k 30, top_p 0.95).  Draws are keyed by (seed, stream, frame).
class MainSampler {
public:
    MainSampler(SamplingMode m, uint64_t seed, float temp = 1.f, float top_p = 0.9f) : mode_(m), seed_(seed), temp_(temp), top_p_(top_p) {}

    Sorted candidates(const float* lg, float temp) const {
        std::vector<float> v;
        std::vector<int> id;
        v.reserve(4097);
        id.reserve(4097);
        for (int j = 0; j < 4097; ++j) {
            if (j == 1 && mode_ == SamplingMode::Llama) continue;
            v.push_back(lg[j]);
            id.push_back(j);
        }
        return sort_candidates(v, id, temp);
    }
    // position of the drawn token; call push() with it unless it is eos
    int sample(const float* lg, uint32_t frame) const {
        Sorted s = candidates(lg, temp_);
        const int n = (int) s.w.size();
        const int first = s.ids[(size_t) draw_sorted(s.w.data(), s.ids.data(), n, s.z, main_spec(temp_, top_p_), philox_uniform(seed_, 0, frame))];
        if (first == 0 || !repeats(first)) return first;
        Sorted r = candidates(lg, 1.f);
        return r.ids[(size_t) draw_sorted(r.w.data(), r.ids.data(), n, r.z, ras_spec(), philox_uniform(seed_, 1, frame))];
    }
    std::vector<int> nucleus(const float* lg, bool ras) const {
        return ras ? nucleus_ids(candidates(lg, 1.f), ras_spec()) : nucleus_ids(candidates(lg, temp_), main_spec(temp_, top_p_));
    }
    bool repeats(int pos) const { return std::find(hist_.begin(), hist_.end(), pos) != hist_.end(); }
    void push(int pos) {
        hist_.push_back(pos);
        if ((int) hist_.size() > kRasWindow) hist_.pop_front();
    }
    void reset() { hist_.clear(); }

private:
    SamplingMode mode_;
    uint64_t seed_;
    float temp_, top_p_;
    std::deque<int> hist_;
};

} // namespace strata::drama
