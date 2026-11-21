// include/strata/drama/drama_forward.hpp - Drama 3 backbone forward pass, one row at a time (J2 of docs/plan-drama3-maya.md).
//
// A row is a token id (BF16 row of token_embd, gathered from the host mmap) or an f32 embedding row.  It goes through the 40
// layers over the parts of a DramaModel (the residual hops between devices at the part boundary), then the final RMSNorm (`h`)
// and the compact lm_head: logits for ids [eos, sem0 .. sem4095].  All activations and the KV / recurrent states are f32.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "strata/drama/drama_model.hpp"

namespace strata::drama {

class DramaRunner {
public:
    explicit DramaRunner(const DramaModel& model);
    ~DramaRunner();
    DramaRunner(const DramaRunner&) = delete;
    DramaRunner& operator=(const DramaRunner&) = delete;

    // allocates the scratch and the per-layer state on each layer's device: a KV cache of `ctx` rows per attention layer, the GDN matrices
    bool init(int ctx, std::string& err);
    void reset(); // position 0, recurrent states zeroed (the KV cache needs no clearing: only rows < pos are read)
    int pos() const;
    int n_out() const; // logits per row (the lm_head rows)

    // the prefix cache: copies the recurrent states and the KV rows below pos() into `snap` (device memory), and puts them back (position included).
    // Rows past the saved position are overwritten by the next steps, as after reset().
    bool save(DeviceSnapshot& snap, std::string& err);
    bool restore(const DeviceSnapshot& snap, std::string& err);

    // runs one row at position pos(), then pos()++.  tok >= 0: a token id; else `emb` (n_embd floats).
    // h (n_embd, after output_norm) and logits (n_out) go to host buffers; either may be null.
    bool step(int tok, const float* emb, float* h, float* logits, std::string& err);

    std::vector<std::string> report() const; // state and scratch bytes per device, free VRAM now

private:
    struct Impl;
    std::unique_ptr<Impl> m_;
};

} // namespace strata::drama
