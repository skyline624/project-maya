// include/strata/drama/expert_pool.hpp - the expert pool of Drama 3 (J5 of docs/plan-drama3-maya.md, 5).
//
// Most of the MoE experts live in VRAM, the rest in pinned host RAM and come over PCIe when the router picks them.  Per layer there are S slots of one expert
// each (gate | up | down Q8_0, 3 342 336 bytes, one blob); the expert kernels read an expert at `slot * stride`, so the kernels are the resident ones.
// With S == n_expert, slot == expert id, nothing is ever moved and the runner needs no host round trip.  The design follows Maya's GLM pool (LFU with aging,
// next-layer prefetch) but is sized for one sequence: one copy stream per device, a few hundred lines.  Internal to the runner; no CUDA in this header.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "strata/drama/drama_model.hpp"

namespace strata::drama {

struct ExpertOptions {
    int64_t vram_mib = -1;       // experts per device in MiB: < 0 all that fit after what is already allocated and the reserve; the pool never holds more than n_expert per layer
    int64_t reserve_mib = 1536;  // VRAM left free per device when vram_mib < 0
    bool prefetch = true;        // predict the next layer's experts from its router on the current hidden state
    std::string profile;         // usage counters (written by profile_save) that decide the initial fill and seed the LFU
    std::string profile_save;    // written by ExpertPool::save_profile
};

struct PoolStats {
    uint64_t hits = 0, misses = 0;           // experts the router picked: already in a slot / copied on demand
    uint64_t pf_issued = 0, pf_useful = 0;   // prefetch copies / prefetched experts the router then picked
    uint64_t pf_skipped = 0;                 // predicted but not worth evicting for
    uint64_t bytes = 0;                      // H2D bytes (demand + prefetch)
    double sync_ms = 0;                      // host time blocked on the per-layer router read-back
    uint64_t steps = 0;                      // backbone rows run (filled in by the runner)
};

class ExpertPool {
public:
    explicit ExpertPool(const DramaModel& model);
    ~ExpertPool();
    ExpertPool(const ExpertPool&) = delete;
    ExpertPool& operator=(const ExpertPool&) = delete;

    // Allocates the slots of every device (after the runners' own allocations: the default budget is what is free then), fills them, and builds the pinned host tier
    // when some layer has fewer slots than experts.  A failed init leaves nothing allocated.
    bool init(const ExpertOptions& opt, std::string& err);
    bool verify(std::string& err); // one expert per layer read back from its slot and compared with the file

    int slots(int il) const;
    bool direct(int il) const { return slots(il) >= n_expert_; } // slot == expert id: ids can be used as they are
    bool counting() const;                                       // a profile is to be saved: the runner must read the ids back even when everything is resident
    // layer il's slot blobs: expert in slot s: gate at gate + s * stride, up at up + s * stride, down at down + s * stride
    struct View { const void *gate, *up, *down; size_t stride; };
    View view(int il) const;
    size_t expert_bytes() const { return esz_; } // one of gate, up, down

    // The router of layer il picked `ids` (host copy of the n ints): slots[k] is where ids[k] is, the missing ones are enqueued on the device's copy stream and
    // `compute` (a cudaStream_t of that device) waits for them.  Never evicts an expert of this call.
    bool acquire(int il, const int* ids, int n, int* slots, void* compute, std::string& err);
    // Layer il's router, run on a hidden state before it was due, predicts `ids`: starts their copies now (a wrong guess costs one slot of a cold expert).
    void prefetch(int il, const int* ids, int n);
    void note_sync(double ms);

    bool save_profile(std::string& err) const; // total usage per layer and expert, every acquire() since init
    PoolStats stats() const;
    void reset_stats();
    std::vector<std::string> report() const; // slots, VRAM and pinned RAM per device, the raw H2D speed measured at init

private:
    struct Impl;
    std::unique_ptr<Impl> m_;
    int n_expert_ = 0;
    size_t esz_ = 0;
};

} // namespace strata::drama
