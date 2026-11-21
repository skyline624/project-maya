// src/drama/expert_pool.cu - the expert pool of Drama 3 (J5).  See expert_pool.hpp.
//
// Per layer: S slots of one expert blob [gate | up | down].  Slot choice = LFU with aging (counters halved every 32768 events of the layer, ties: the least
// recently used slot goes first), counters kept for every expert, resident or not.  The host tier is one pinned copy of every expert of the parts that need it
// (a resident expert is duplicated there: nothing is ever copied back, an evicted expert is already in RAM).
#include "strata/drama/expert_pool.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <thread>
#include <tuple>

namespace strata::drama {

namespace {

constexpr uint64_t kAgeEvery = 32768;          // as the GLM pool: counters halved after this many expert uses in a layer
constexpr uint64_t kMaxPinned = 36ull << 30;   // pinned host RAM cap (the server has 125 GiB)
constexpr int kProbeCopies = 64;
double gib(uint64_t b) { return (double) b / (1024.0 * 1024 * 1024); }

} // namespace

struct ExpertPool::Impl {
    const DramaModel& m;
    const DramaGeometry& g;
    ExpertOptions opt;
    size_t esz = 0, blob = 0;
    int ne = 0;
    struct Layer {
        int part = 0, S = 0;
        uint8_t* slots = nullptr;        // S blobs on the device
        const uint8_t* host = nullptr;   // n_expert blobs, pinned (null: S == n_expert)
        std::vector<int> slot_of, expert_of;
        std::vector<uint32_t> cnt;       // aged use counters per expert
        std::vector<uint64_t> total;     // raw use counters per expert (the profile)
        std::vector<uint64_t> last, used, prot; // per slot: tick of last use, hard exclusion (this acquire's experts), soft exclusion (predicted by the prefetch)
        std::vector<uint8_t> pref;       // per slot: copied by the prefetch and not used yet
        uint64_t tick = 0, events = 0;
        cudaEvent_t ev = nullptr;
        bool pending = false;            // copies enqueued that `ev` marks the end of
    };
    struct Part {
        int dev = 0, l0 = 0, l1 = 0, S = 0;
        uint8_t* arena = nullptr;
        cudaStream_t cp = nullptr;
        uint8_t* pinned = nullptr;       // the host tier of the part's layers (one block), null if S == n_expert
        uint64_t pinned_bytes = 0, arena_bytes = 0;
        double probe_gbs = 0;
        std::string err;
        bool ok = true;
    };
    std::vector<Layer> L;
    std::vector<Part> P;
    PoolStats st;

    explicit Impl(const DramaModel& model) : m(model), g(model.geometry()) {}
    ~Impl() {
        for (auto& l : L) if (l.ev) cudaEventDestroy(l.ev);
        for (auto& p : P) {
            cudaSetDevice(p.dev);
            if (p.cp) cudaStreamDestroy(p.cp);
            if (p.arena) cudaFree(p.arena);
            if (p.pinned) cudaFreeHost(p.pinned);
        }
    }

    // the slot to give up in layer l: not one of this call's experts (used == hard), not a predicted one if another will do, then the lowest counter, then the oldest
    int victim(const Layer& l, uint64_t hard, uint64_t soft) const {
        int best = -1;
        int64_t bc = 0;
        uint64_t ba = 0;
        int bp = 0;
        for (int s = 0; s < l.S; ++s) {
            if (l.used[(size_t) s] == hard) continue;
            const int e = l.expert_of[(size_t) s];
            const int64_t c = e < 0 ? -1 : (int64_t) l.cnt[(size_t) e];
            const int p = l.prot[(size_t) s] == soft ? 1 : 0;
            if (best < 0 || std::make_tuple(p, c, l.last[(size_t) s]) < std::make_tuple(bp, bc, ba)) { best = s; bp = p; bc = c; ba = l.last[(size_t) s]; }
        }
        return best;
    }
    void assign(Layer& l, int s, int e) {
        const int old = l.expert_of[(size_t) s];
        if (old >= 0) l.slot_of[(size_t) old] = -1;
        l.expert_of[(size_t) s] = e;
        l.slot_of[(size_t) e] = s;
        l.pref[(size_t) s] = 0;
    }
    void copy_in(Layer& l, const Part& p, int s, int e) {
        cudaMemcpyAsync(l.slots + (size_t) s * blob, l.host + (size_t) e * blob, blob, cudaMemcpyHostToDevice, p.cp);
        st.bytes += blob;
    }
    void age(Layer& l) {
        for (auto& c : l.cnt) c >>= 1;
        l.events = 0;
    }
};

ExpertPool::ExpertPool(const DramaModel& model) : m_(new Impl(model)) {}
ExpertPool::~ExpertPool() = default;
int ExpertPool::slots(int il) const { return m_->L[(size_t) il].S; }
bool ExpertPool::counting() const { return !m_->opt.profile_save.empty(); }
ExpertPool::View ExpertPool::view(int il) const {
    const uint8_t* b = m_->L[(size_t) il].slots;
    return {b, b + m_->esz, b + 2 * m_->esz, m_->blob};
}
PoolStats ExpertPool::stats() const { return m_->st; }
void ExpertPool::reset_stats() { m_->st = PoolStats(); }
void ExpertPool::note_sync(double ms) { m_->st.sync_ms += ms; }

// profile: "# drama-expert-profile 1 <n_layer> <n_expert>" then one line of n_expert counts per layer
static bool read_profile(const std::string& path, int nl, int ne, std::vector<std::vector<uint64_t>>& out, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) { err = "cannot open " + path; return false; }
    int v = 0, l = 0, e = 0;
    bool ok = std::fscanf(f, "# drama-expert-profile %d %d %d", &v, &l, &e) == 3 && v == 1;
    if (!ok) err = path + ": not a drama-expert-profile 1 file";
    else if (l != nl || e != ne) { err = path + ": profile of " + std::to_string(l) + " layers x " + std::to_string(e) + " experts, the model has " + std::to_string(nl) + " x " + std::to_string(ne); ok = false; }
    out.assign((size_t) nl, std::vector<uint64_t>((size_t) ne, 0));
    for (int i = 0; ok && i < nl; ++i)
        for (int j = 0; ok && j < ne; ++j) {
            unsigned long long c = 0;
            if (std::fscanf(f, "%llu", &c) != 1) { err = path + ": truncated at layer " + std::to_string(i); ok = false; }
            out[(size_t) i][(size_t) j] = c;
        }
    std::fclose(f);
    return ok;
}

bool ExpertPool::save_profile(std::string& err) const {
    const Impl& s = *m_;
    if (s.opt.profile_save.empty()) return true;
    FILE* f = std::fopen(s.opt.profile_save.c_str(), "w");
    if (!f) { err = "cannot write " + s.opt.profile_save; return false; }
    std::fprintf(f, "# drama-expert-profile 1 %d %d\n", (int) s.L.size(), s.ne);
    for (const auto& l : s.L) {
        for (int e = 0; e < s.ne; ++e) std::fprintf(f, e ? " %llu" : "%llu", (unsigned long long) l.total[(size_t) e]);
        std::fputc('\n', f);
    }
    std::fclose(f);
    return true;
}

bool ExpertPool::init(const ExpertOptions& opt, std::string& err) {
    Impl& s = *m_;
    s.opt = opt;
    const DramaGeometry& g = s.g;
    const auto& layers = s.m.layers();
    s.ne = n_expert_ = g.n_expert;
    const uint64_t eb = layers[0].exps_gate.bytes / (uint64_t) g.n_expert;
    for (const auto& l : layers)
        if (l.exps_gate.bytes != l.exps_up.bytes || l.exps_gate.bytes != l.exps_down.bytes || l.exps_gate.bytes / (uint64_t) g.n_expert != eb || !l.exps_src[0] || !l.exps_src[1] || !l.exps_src[2]) {
            err = "expert tensors do not have one size per expert";
            return false;
        }
    s.esz = esz_ = (size_t) eb;
    s.blob = 3 * s.esz;
    int np = 0;
    for (const auto& l : layers) np = std::max(np, l.part + 1);
    s.P.assign((size_t) np, Impl::Part());
    s.L.assign(layers.size(), Impl::Layer());
    for (size_t il = 0; il < layers.size(); ++il) {
        Impl::Part& p = s.P[(size_t) layers[il].part];
        if (p.l1 == 0) { p.dev = layers[il].dev; p.l0 = (int) il; }
        p.l1 = (int) il + 1;
        s.L[il].part = layers[il].part;
    }
    // slots per layer on each device
    uint64_t pinned_total = 0;
    for (auto& p : s.P) {
        const int nl = p.l1 - p.l0;
        size_t fr = 0, tot = 0;
        if (cudaSetDevice(p.dev) != cudaSuccess || cudaMemGetInfo(&fr, &tot) != cudaSuccess) { err = "cudaMemGetInfo failed on device " + std::to_string(p.dev); return false; }
        const int64_t budget = opt.vram_mib >= 0 ? opt.vram_mib * (1 << 20) : std::max<int64_t>(0, (int64_t) fr - opt.reserve_mib * (1 << 20));
        p.S = (int) std::min<int64_t>(g.n_expert, budget / ((int64_t) nl * (int64_t) s.blob));
        if (p.S < g.n_expert_used) {
            err = "device " + std::to_string(p.dev) + ": a budget of " + std::to_string(budget >> 20) + " MiB gives " + std::to_string(p.S) + " slots per layer, at least " + std::to_string(g.n_expert_used) + " are needed";
            return false;
        }
        p.arena_bytes = (uint64_t) nl * (uint64_t) p.S * s.blob;
        if (p.S < g.n_expert) pinned_total += p.pinned_bytes = (uint64_t) nl * (uint64_t) g.n_expert * s.blob;
    }
    if (pinned_total > kMaxPinned) { err = "the host tier needs " + std::to_string(pinned_total >> 20) + " MiB of pinned RAM, the cap is " + std::to_string(kMaxPinned >> 20); return false; }
    std::vector<std::vector<uint64_t>> prof;
    if (!opt.profile.empty() && !read_profile(opt.profile, g.n_layer, g.n_expert, prof, err)) return false;

    // allocations
    for (auto& p : s.P) {
        p.arena = nullptr;
        p.pinned = nullptr;
        cudaError_t e = cudaSetDevice(p.dev);
        if (e == cudaSuccess) e = cudaStreamCreateWithFlags(&p.cp, cudaStreamNonBlocking);
        if (e == cudaSuccess) e = cudaMalloc((void**) &p.arena, p.arena_bytes);
        if (e != cudaSuccess) p.arena = nullptr;
        if (e == cudaSuccess && p.pinned_bytes) e = cudaHostAlloc((void**) &p.pinned, p.pinned_bytes, cudaHostAllocDefault);
        if (e != cudaSuccess) p.pinned = nullptr;
        if (e != cudaSuccess) {
            err = "device " + std::to_string(p.dev) + ": pool of " + std::to_string(p.arena_bytes >> 20) + " MiB + " + std::to_string(p.pinned_bytes >> 20) + " MiB pinned: " + cudaGetErrorString(e);
            return false;
        }
        for (int il = p.l0; il < p.l1; ++il) {
            Impl::Layer& l = s.L[(size_t) il];
            l.S = p.S;
            l.slots = p.arena + (uint64_t) (il - p.l0) * (uint64_t) p.S * s.blob;
            if (p.pinned) l.host = p.pinned + (uint64_t) (il - p.l0) * (uint64_t) g.n_expert * s.blob;
            l.slot_of.assign((size_t) g.n_expert, -1);
            l.expert_of.assign((size_t) p.S, -1);
            l.cnt.assign((size_t) g.n_expert, 0);
            l.total.assign((size_t) g.n_expert, 0);
            l.last.assign((size_t) p.S, 0);
            l.used.assign((size_t) p.S, 0);
            l.prot.assign((size_t) p.S, 0);
            l.pref.assign((size_t) p.S, 0);
            if (cudaEventCreateWithFlags(&l.ev, cudaEventDisableTiming) != cudaSuccess) { err = "cudaEventCreate failed"; return false; }
        }
    }

    // fill: one thread per device.  The blob of every expert is packed from the mmap into pinned memory (the host tier, or one reusable staging block), then the
    // chosen experts are copied to their slots.  S == n_expert: slot = expert id.
    for (auto& p : s.P)
        p.ok = true;
    std::vector<std::thread> th;
    for (size_t pi = 0; pi < s.P.size(); ++pi)
        th.emplace_back([&, pi] {
            Impl::Part& p = s.P[pi];
            const int NE = g.n_expert;
            cudaError_t e = cudaSetDevice(p.dev);
            uint8_t* stage = nullptr;
            if (e == cudaSuccess && !p.pinned) e = cudaHostAlloc((void**) &stage, (size_t) NE * s.blob, cudaHostAllocDefault);
            for (int il = p.l0; il < p.l1 && e == cudaSuccess; ++il) {
                Impl::Layer& l = s.L[(size_t) il];
                const DramaLayer& D = layers[(size_t) il];
                uint8_t* h = p.pinned ? (uint8_t*) l.host : stage;
                for (int x = 0; x < NE; ++x)
                    for (int k = 0; k < 3; ++k) std::memcpy(h + (size_t) x * s.blob + (size_t) k * s.esz, D.exps_src[k] + (size_t) x * s.esz, s.esz);
                std::vector<int> pick((size_t) l.S);
                for (int i = 0; i < l.S; ++i) pick[(size_t) i] = i;
                if (!prof.empty() && l.S < NE) {
                    std::vector<int> order((size_t) NE);
                    for (int i = 0; i < NE; ++i) order[(size_t) i] = i;
                    const auto& pr = prof[(size_t) il];
                    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return pr[(size_t) a] > pr[(size_t) b]; });
                    for (int i = 0; i < l.S; ++i) pick[(size_t) i] = order[(size_t) i];
                }
                if (!prof.empty()) { // seed the LFU: the hottest expert of the layer starts at 4096 uses
                    const uint64_t mx = std::max<uint64_t>(1, *std::max_element(prof[(size_t) il].begin(), prof[(size_t) il].end()));
                    for (int x = 0; x < NE; ++x) l.cnt[(size_t) x] = (uint32_t) (prof[(size_t) il][(size_t) x] * 4096 / mx);
                }
                if (il == p.l0 && l.S < NE) { // raw speed of the slot copies, measured on this layer's slots before they are filled
                    cudaEvent_t a, b;
                    cudaEventCreate(&a);
                    cudaEventCreate(&b);
                    const int n = std::min(kProbeCopies, l.S);
                    cudaEventRecord(a, p.cp);
                    for (int i = 0; i < n; ++i) cudaMemcpyAsync(l.slots + (size_t) i * s.blob, h + (size_t) i * s.blob, s.blob, cudaMemcpyHostToDevice, p.cp);
                    cudaEventRecord(b, p.cp);
                    cudaEventSynchronize(b);
                    float ms = 0;
                    cudaEventElapsedTime(&ms, a, b);
                    p.probe_gbs = ms > 0 ? (double) n * (double) s.blob / (ms * 1e-3) / 1e9 : 0;
                    cudaEventDestroy(a);
                    cudaEventDestroy(b);
                }
                for (int i = 0; i < l.S && e == cudaSuccess; ++i) {
                    s.assign(l, i, pick[(size_t) i]);
                    e = cudaMemcpyAsync(l.slots + (size_t) i * s.blob, h + (size_t) pick[(size_t) i] * s.blob, s.blob, cudaMemcpyHostToDevice, p.cp);
                }
                if (e == cudaSuccess) e = cudaStreamSynchronize(p.cp); // the staging block is reused by the next layer
            }
            if (stage) cudaFreeHost(stage);
            if (e != cudaSuccess) { p.ok = false; p.err = std::string("fill: ") + cudaGetErrorString(e); }
        });
    for (auto& t : th) t.join();
    for (auto& p : s.P)
        if (!p.ok) { err = "device " + std::to_string(p.dev) + ": " + p.err; return false; }
    s.st = PoolStats();
    return true;
}

bool ExpertPool::verify(std::string& err) {
    Impl& s = *m_;
    std::vector<uint8_t> h(s.blob);
    for (size_t il = 0; il < s.L.size(); ++il) {
        const Impl::Layer& l = s.L[il];
        const DramaLayer& D = s.m.layers()[il];
        for (int sl : {0, l.S - 1}) {
            const int e = l.expert_of[(size_t) sl];
            cudaError_t c = cudaSetDevice(D.dev);
            if (c == cudaSuccess) c = cudaMemcpy(h.data(), l.slots + (size_t) sl * s.blob, s.blob, cudaMemcpyDeviceToHost);
            if (c != cudaSuccess) { err = std::string("pool verify: ") + cudaGetErrorString(c); return false; }
            for (int k = 0; k < 3; ++k)
                if (std::memcmp(h.data() + (size_t) k * s.esz, D.exps_src[k] + (size_t) e * s.esz, s.esz) != 0) {
                    err = "pool verify: layer " + std::to_string(il) + " slot " + std::to_string(sl) + " (expert " + std::to_string(e) + ") differs from the file";
                    return false;
                }
        }
    }
    return true;
}

bool ExpertPool::acquire(int il, const int* ids, int n, int* slots, void* compute, std::string& err) {
    Impl& s = *m_;
    Impl::Layer& l = s.L[(size_t) il];
    Impl::Part& p = s.P[(size_t) l.part];
    cudaSetDevice(p.dev);
    const uint64_t tick = ++l.tick;
    if (n > l.S) { err = "layer " + std::to_string(il) + ": " + std::to_string(n) + " experts for " + std::to_string(l.S) + " slots"; return false; }
    for (int k = 0; k < n; ++k) {
        if (ids[k] < 0 || ids[k] >= s.ne) { err = "layer " + std::to_string(il) + ": expert id " + std::to_string(ids[k]) + " out of range"; return false; }
        ++l.total[(size_t) ids[k]];
        ++l.cnt[(size_t) ids[k]];
        if (++l.events >= kAgeEvery) s.age(l);
    }
    if (l.S >= s.ne) { // everything resident (only reached to count the use)
        for (int k = 0; k < n; ++k) slots[k] = ids[k];
        s.st.hits += (uint64_t) n;
        return true;
    }
    for (int k = 0; k < n; ++k) {
        const int sl = l.slot_of[(size_t) ids[k]];
        slots[k] = sl;
        if (sl < 0) continue;
        l.used[(size_t) sl] = tick;
        l.last[(size_t) sl] = tick;
        ++s.st.hits;
        if (l.pref[(size_t) sl]) { ++s.st.pf_useful; l.pref[(size_t) sl] = 0; }
    }
    bool issued = false;
    for (int k = 0; k < n; ++k) {
        if (slots[k] >= 0) continue;
        const int dup = l.slot_of[(size_t) ids[k]]; // the same expert twice in one call
        if (dup >= 0) { slots[k] = dup; continue; }
        const int v = s.victim(l, tick, tick);
        if (v < 0) { err = "layer " + std::to_string(il) + ": no slot to evict"; return false; }
        s.assign(l, v, ids[k]);
        l.used[(size_t) v] = tick;
        l.last[(size_t) v] = tick;
        slots[k] = v;
        s.copy_in(l, p, v, ids[k]);
        ++s.st.misses;
        issued = true;
    }
    if (issued) { cudaEventRecord(l.ev, p.cp); l.pending = true; }
    if (l.pending) { cudaStreamWaitEvent((cudaStream_t) compute, l.ev, 0); l.pending = false; }
    return true;
}

void ExpertPool::prefetch(int il, const int* ids, int n) {
    Impl& s = *m_;
    Impl::Layer& l = s.L[(size_t) il];
    if (l.S >= s.ne) return;
    Impl::Part& p = s.P[(size_t) l.part];
    cudaSetDevice(p.dev);
    const uint64_t next = l.tick + 1; // the tick of the acquire this prefetch is for
    for (int k = 0; k < n; ++k)
        if (const int sl = l.slot_of[(size_t) ids[k]]; sl >= 0) l.prot[(size_t) sl] = next;
    bool issued = false;
    for (int k = 0; k < n; ++k) {
        const int e = ids[k];
        if (e < 0 || e >= s.ne || l.slot_of[(size_t) e] >= 0) continue;
        const int v = s.victim(l, ~0ull, next);
        if (v < 0 || l.prot[(size_t) v] == next) { ++s.st.pf_skipped; continue; } // only predicted experts left
        const int old = l.expert_of[(size_t) v];
        if (old >= 0 && l.cnt[(size_t) old] > l.cnt[(size_t) e]) { ++s.st.pf_skipped; continue; } // the resident one is worth more than the guess
        s.assign(l, v, e);
        l.prot[(size_t) v] = next;
        l.pref[(size_t) v] = 1;
        l.last[(size_t) v] = l.tick;
        s.copy_in(l, p, v, e);
        ++s.st.pf_issued;
        issued = true;
    }
    if (issued) { cudaEventRecord(l.ev, p.cp); l.pending = true; }
}

std::vector<std::string> ExpertPool::report() const {
    const Impl& s = *m_;
    std::vector<std::string> v;
    for (const auto& p : s.P) {
        size_t fr = 0, tot = 0;
        cudaSetDevice(p.dev);
        cudaMemGetInfo(&fr, &tot);
        char b[512];
        std::snprintf(b, sizeof b, "expert pool dev %d layers %d-%d: %d slots per layer of %d (%.1f %%), VRAM %.2f GiB, pinned %.2f GiB%s; free now %.2f / %.2f GiB", p.dev, p.l0, p.l1 - 1, p.S, s.ne,
                      100.0 * p.S / s.ne, gib(p.arena_bytes), gib(p.pinned_bytes), p.probe_gbs > 0 ? (", raw H2D " + std::to_string(p.probe_gbs).substr(0, 5) + " GB/s").c_str() : "", gib(fr), gib(tot));
        v.push_back(b);
    }
    return v;
}

} // namespace strata::drama
