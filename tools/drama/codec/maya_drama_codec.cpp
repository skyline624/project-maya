// maya-drama-codec: the Drama 3 / Fish S2 DAC codec of llama.cpp (libmtmd public API), as a command line tool.
//   maya-drama-codec decode --mmproj mm.gguf --codes in.codes --out out.wav [--chunk 64] [--gpu]
//   maya-drama-codec encode --mmproj mm.gguf --wav ref.wav --out ref.codes [--gpu]
// .codes = "DRC1" u32 n_frames u32 n_cb(10), per frame i32 c0 (semantic index 0..4095, not a token id), c1..c9 (0..1023); little-endian.
// decode: PCM mono 16-bit at the codec rate (44100 Hz), the frames go through the codec by pieces of --chunk with its state kept
//         (mtmd_gen_audio_process GEN_WAV, as drama3_gen_audio_pipeline::decode_wav does with 64).
// encode: the reference clip goes through the speaker encoder path of mtmd_helper_gen_audio (mtmd_helper_bitmap_init_from_file:
//         decode + mono + resample to the codec rate; mtmd_encode_chunk; mtmd_gen_state_get_ref_codes). Truncation to 30 s is llama.cpp's.
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

static constexpr uint32_t N_CB = 10;

[[noreturn]] static void die(const std::string & msg) { fprintf(stderr, "maya-drama-codec: %s\n", msg.c_str()); exit(1); }

static mtmd_context * init_mtmd(const std::string & mmproj, bool gpu) {
    mtmd_context_params p = mtmd_context_params_default();
    p.use_gpu = gpu;
    p.n_threads = 12;
    p.warmup = false;
    p.print_timings = false;
    mtmd_context * m = mtmd_init_from_file(mmproj.c_str(), nullptr, p);  // no text model needed for the codec
    if (!m) die("cannot load mmproj " + mmproj);
    const mtmd_gen_audio_info info = mtmd_gen_audio_get_info(m);
    if (info.type != MTMD_GEN_AUDIO_TYPE_DRAMA3) die("mmproj is not a Drama 3 generator");
    return m;
}

static void put(std::ofstream & f, const void * p, size_t n) { f.write((const char *) p, (std::streamsize) n); }

static int decode(std::map<std::string, std::string> & a) {
    for (const char * k : { "mmproj", "codes", "out" }) if (!a.count(k)) die(std::string("missing --") + k);
    const size_t chunk = a.count("chunk") ? (size_t) atoll(a["chunk"].c_str()) : 64;
    if (chunk == 0) die("--chunk must be > 0");

    std::ifstream in(a["codes"], std::ios::binary);
    char magic[4]; uint32_t hdr[2] = { 0, 0 };
    in.read(magic, 4); in.read((char *) hdr, 8);
    if (!in || memcmp(magic, "DRC1", 4) != 0 || hdr[1] != N_CB) die("bad .codes header: " + a["codes"]);
    const size_t n_frames = hdr[0];
    std::vector<int32_t> codes(n_frames * N_CB);
    in.read((char *) codes.data(), (std::streamsize) codes.size() * 4);
    if (!in) die("truncated .codes: " + a["codes"]);

    mtmd_context * mctx = init_mtmd(a["mmproj"], a.count("gpu") > 0);
    const int32_t rate = mtmd_gen_audio_get_info(mctx).sample_rate;
    mtmd_gen_state * st = mtmd_gen_state_init(mctx, mtmd_gen_state_params_default());
    if (!st) die("cannot create the gen state");

    std::vector<float> pcm;
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t done = 0; done < n_frames;) {
        const size_t n = std::min(chunk, n_frames - done);
        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
        inp.codes = codes.data() + done * N_CB;
        inp.n_codes = n * N_CB;
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, st, &inp, &out) != 0) die("codec decode failed at frame " + std::to_string(done));
        pcm.insert(pcm.end(), out.audio, out.audio + out.n_samples);
        done += n;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // 16-bit WAV as write_wav16 of mtmd-helper-gen
    std::ofstream f(a["out"], std::ios::binary);
    if (!f) die("cannot write " + a["out"]);
    const uint32_t data_sz = (uint32_t) (pcm.size() * 2), riff_sz = 36 + data_sz, fmt_sz = 16, rate32 = (uint32_t) rate, byte_rate = rate32 * 2;
    const uint16_t fmt = 1, ch = 1, align = 2, bits = 16;
    put(f, "RIFF", 4); put(f, &riff_sz, 4); put(f, "WAVE", 4); put(f, "fmt ", 4); put(f, &fmt_sz, 4);
    put(f, &fmt, 2); put(f, &ch, 2); put(f, &rate32, 4); put(f, &byte_rate, 4); put(f, &align, 2); put(f, &bits, 2);
    put(f, "data", 4); put(f, &data_sz, 4);
    std::vector<int16_t> s16(pcm.size());
    for (size_t i = 0; i < pcm.size(); i++) s16[i] = (int16_t) std::lround(std::max(-1.0f, std::min(1.0f, pcm[i])) * 32767.0f);
    put(f, s16.data(), s16.size() * 2);
    if (!f) die("write failed");
    fprintf(stderr, "%zu frames in pieces of %zu -> %zu samples at %d Hz (%.2f s of audio) in %.1f s\n", n_frames, chunk, pcm.size(), rate,
            (double) pcm.size() / rate, s);
    mtmd_gen_state_free(st);
    mtmd_free(mctx);
    return 0;
}

static int encode(std::map<std::string, std::string> & a) {
    for (const char * k : { "mmproj", "wav", "out" }) if (!a.count(k)) die(std::string("missing --") + k);
    mtmd_context * mctx = init_mtmd(a["mmproj"], a.count("gpu") > 0);
    if (!mtmd_support_audio(mctx)) die("mmproj has no speaker encoder");

    mtmd_helper_bitmap_wrapper w = mtmd_helper_bitmap_init_from_file(mctx, a["wav"].c_str(), false, mtmd_helper_init_opt_default());
    if (!w.bitmap) die("cannot read the audio file " + a["wav"]);
    const std::string marker = mtmd_get_marker(mctx);
    mtmd_input_text text{ marker.c_str(), marker.size(), false, true };
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bmp = w.bitmap;
    if (mtmd_tokenize(mctx, chunks, &text, &bmp, 1) != 0) die("tokenize failed");
    const mtmd_input_chunk * audio = nullptr;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks) && !audio; i++) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_AUDIO) audio = c;
    }
    if (!audio) die("no audio chunk");
    const auto t0 = std::chrono::steady_clock::now();
    if (mtmd_encode_chunk(mctx, audio) != 0) die("speaker encode failed");
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    mtmd_gen_state * st = mtmd_gen_state_init(mctx, mtmd_gen_state_params_default());
    if (!st || mtmd_gen_state_set_ref(mctx, st) != 0) die("the encoder gave no reference codes");
    size_t n_codes = 0;
    const int32_t * codes = mtmd_gen_state_get_ref_codes(st, &n_codes);
    if (!codes || n_codes == 0 || n_codes % N_CB != 0) die("bad reference codes");

    std::ofstream f(a["out"], std::ios::binary);
    if (!f) die("cannot write " + a["out"]);
    const uint32_t h[2] = { (uint32_t) (n_codes / N_CB), N_CB };
    put(f, "DRC1", 4); put(f, h, 8); put(f, codes, n_codes * 4);
    if (!f) die("write failed");
    fprintf(stderr, "%s: %u frames (%.2f s of audio) in %.1f s\n", a["wav"].c_str(), h[0], h[0] * 2048.0 / mtmd_gen_audio_get_info(mctx).sample_rate, s);
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(w.bitmap);
    mtmd_gen_state_free(st);
    mtmd_free(mctx);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) die("usage: maya-drama-codec decode|encode [options]");
    std::map<std::string, std::string> a;
    for (int i = 2; i < argc; i++) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) die("unexpected argument " + k);
        k = k.substr(2);
        if (k == "gpu") { a[k] = "1"; continue; }
        if (i + 1 >= argc) die("missing value for --" + k);
        a[k] = argv[++i];
    }
    const std::string cmd = argv[1];
    if (cmd == "decode") return decode(a);
    if (cmd == "encode") return encode(a);
    die("unknown subcommand " + cmd);
}
