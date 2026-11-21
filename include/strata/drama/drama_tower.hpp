// include/strata/drama/drama_tower.hpp - Drama 3 codebook tower, one backbone row at a time (J3 of docs/plan-drama3-maya.md).
//
// The tower (13 layers: 4 shared, then 9 residual, each followed by a codebook head) runs on the device of part 0, in f32, with its own KV cache.
// For every backbone row it takes the backbone's h, the main code the row's output frame starts with, and (reference rows) the codes of the row's
// input frame; it gives the 9 other codes of the output frame (sampled on the device, or imposed) and the next backbone input row.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "strata/drama/drama_model.hpp"

namespace strata::drama {

constexpr int kCodebooks = 10; // codes per frame: c0 is the semantic index (0..4095), c1..c9 come from the heads

struct TowerSampling {
    bool llama = false; // false: SGLang's rules (`official`); true: llama.cpp's (`llama`), see drama_sampling.hpp
    float temp = 1.f;
    int top_k = 30;
    float top_p = 0.9f;
    uint64_t seed = 1;
};

struct TowerStep {
    const float* h = nullptr;            // n_embd f32: the backbone's h of this row
    int code0 = -1;                      // main code (0..4095) of the output frame; -1: the next row is text, the tower only fills its KV
    const int32_t* frame_in = nullptr;   // kCodebooks codes of this row's input frame (a reference frame); null for a text row
    const int32_t* forced = nullptr;     // t_resid_heads codes imposed on the heads (codes 1..9); null: sampled (needs code0 >= 0)
    uint32_t ctr = 0;                    // draw counter (the frame index): sampling is a pure function of (seed, ctr, head)
    int32_t* codes_out = nullptr;        // kCodebooks codes of the output frame (code0 >= 0)
    float* next_emb = nullptr;           // n_embd: backbone input of the next row, (E_lm[c0] + E_cb1[c1]) / sqrt(2) (code0 >= 0)
    float* layer_out = nullptr;          // t_layers * n_embd: tower_out of every layer (llama.cpp's tensor of that name)
    float* head_logits = nullptr;        // t_resid_heads * 4096: the logits of every head (code0 >= 0)
};

class TowerRunner {
public:
    explicit TowerRunner(const DramaModel& model);
    ~TowerRunner();
    TowerRunner(const TowerRunner&) = delete;
    TowerRunner& operator=(const TowerRunner&) = delete;

    bool init(int ctx, const TowerSampling& sampling, std::string& err); // KV cache of `ctx` rows per layer
    void reset();                                                          // position 0 (the KV cache needs no clearing)
    int pos() const;
    bool save(DeviceSnapshot& snap, std::string& err);                     // the KV rows below pos() and pos (prefix cache, as DramaRunner::save)
    bool restore(const DeviceSnapshot& snap, std::string& err);
    bool step(const TowerStep& in, std::string& err);                      // runs one row at pos(), then pos()++
    std::vector<std::string> report() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_;
};

} // namespace strata::drama
