// include/strata/drama/drama_serve.hpp - maya-drama --serve: the engine behind serve/drama_server.py (J6 of docs/plan-drama3-maya.md).
//
// The model is loaded once, then one JSON object per line is read on stdin and one is answered on stdout (every other stdout line starts with "INFO ").
//   {"op":"gen","id":7,"tokens":[...],"ref_at":66,"ref_codes":"/p/ref.codes","prefix_rows":301,"voice":"key","seed":1,"max_frames":1024,"out":"/p/g.codes"}
//     tokens: the prompt's token ids; ref_at: index in `tokens` where the reference frames go (-1: no reference), their codes read from the DRC1 file ref_codes, each frame
//     becoming one embedding row; prefix_rows: the first rows (the voice, identical in every prompt of `voice`) prefilled once, saved on the devices and restored by later
//     requests of the same voice key (LRU); out: DRC1 file the frames are written to.
//   -> {"id":7,"ok":true,"frames":617,"stop":"eos","prefix":"hit","rows":418,"prefix_ms":1.2,"prefill_ms":2054.3,"gen_ms":13100.2,"backbone_ms":..,"tower_ms":..}
//      or {"id":7,"ok":false,"error":"..."}
//   {"op":"quit"}
// First line after loading: {"event":"ready","ctx":4096,"voices":4,"sampling":"official"}.
#pragma once

namespace strata::drama {

// true when argv holds --serve
bool serve_requested(int argc, char** argv);
// parses the options (--model --mmproj [--devs 0,1] [--layer-split N] [--ctx 4096] [--voices 4] [--sampling official|llama]), loads, serves until "quit" or the end of stdin
int serve_main(int argc, char** argv);

} // namespace strata::drama
