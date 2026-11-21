// src/drama/drama_main.cpp - maya-drama: Drama 3 on a Maya fork.  J1: --check loads, verifies and reports the weights.
//   maya-drama --check --model <backbone.gguf> --mmproj <mmproj.gguf> [--devs 0,1] [--layer-split 20] [--verify]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "strata/drama/drama_model.hpp"

namespace {

int usage() {
    std::puts("usage: maya-drama --check --model <backbone.gguf> --mmproj <mmproj.gguf> [--devs 0,1] [--layer-split 20] [--verify]\n"
              "  --check        load both GGUFs onto the GPUs, report VRAM per device, print OK (exit 0) or ERROR <reason> (exit 1)\n"
              "  --devs         CUDA devices, one per part (default 0,1)\n"
              "  --layer-split  first layer of each later part, comma separated (default: an even split)\n"
              "  --verify       read back the first and last 4 KiB of every tensor (and of one expert per layer) and compare with the file");
    return 2;
}

std::vector<int> ints(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');) v.push_back(std::atoi(t.c_str()));
    return v;
}

int fail(const std::string& why) {
    std::printf("ERROR %s\n", why.c_str());
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    bool check = false, verify = false;
    std::string model, mmproj;
    std::vector<int> devs = {0, 1}, split;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto val = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--check") check = true;
        else if (a == "--verify") verify = true;
        else if (a == "--model") model = val();
        else if (a == "--mmproj") mmproj = val();
        else if (a == "--devs") devs = ints(val());
        else if (a == "--layer-split") split = ints(val());
        else return usage();
    }
    if (!check) return usage();
    if (model.empty() || mmproj.empty()) return fail("--check needs --model and --mmproj");

    strata::drama::DramaModel m;
    std::string err;
    if (!m.load(model, mmproj, devs, split, err)) return fail(err);
    for (const auto& l : m.geometry().describe()) std::printf("INFO %s\n", l.c_str());
    if (verify && !m.verify(err)) return fail(err);
    for (const auto& l : m.report()) std::printf("INFO %s\n", l.c_str());
    std::puts("OK");
    return 0;
}
