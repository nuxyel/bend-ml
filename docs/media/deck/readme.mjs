// The README deck: bend-ml v1.2.2 and every release before it, measured again on 2026-10-11 next to PyTorch,
// NumPy, llama.cpp, llm.c, candle and dfdx. Every number comes from numbers.json ("evidence").
//   node docs/media/deck.mjs docs/media/deck/readme.mjs docs/media/deck/frames
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const title = "bend-ml v1.2.2 · evidence of 2026-10-11";
const here = path.dirname(fileURLToPath(import.meta.url));
const N = JSON.parse(fs.readFileSync(path.join(here, "..", "numbers.json"), "utf8"));
const E = N.evidence;
const FOOT = "Intel Core Ultra 7 155H, 32 GB · measured on 2026-10-11";

export default ({ frame, hbars, vbars, lines, stats, table, C }) => {
  const H = E.gpt2_ms_per_token_history, P = E.gpt2_ms_per_token_peers, M = E.mnist_ms_per_batch, TC = E.gpt2_thread_curve;
  const ms = (v) => (v >= 100 ? `${Math.round(v)} ms` : v >= 10 ? `${v.toFixed(1)} ms` : `${v.toFixed(2)} ms`);
  const x = (a, b) => `${(a / b).toFixed(1)}×`;
  const b22 = H["v1.2.2 (Bend 2.0.36)"];
  const slides = [];

  slides.push({ name: "title", html: frame({
    kicker: "bend-ml v1.2.2",
    title: `machine learning in Bend 2, <span class="v">measured</span>`,
    body: `<div style="height:40px"></div>` + stats([
      { big: "27", label: "laws proved", sub: "checked again by the Lean kernel" },
      { big: "7", label: "releases measured", sub: "v0.2 to v1.2.2, one night" },
      { big: "6", label: "reference systems", sub: "PyTorch · NumPy · llama.cpp · llm.c · candle · dfdx" },
    ], 110) + `<div class="p" style="text-align:center;margin-top:60px">Every release of bend-ml, next to the systems people use, on the same work.</div>`,
    foot: FOOT,
  }) });

  slides.push({ name: "same-work", html: frame({
    kicker: "does it compute the same thing?",
    title: `same tokens, same loss, <span class="g">everywhere</span>`,
    body: table(["workload", "systems that agree", "what they agree on"], [
      ["GPT-2 small, 33 tokens", "bend-ml v0.3–v1.2.2 · PyTorch · NumPy · llama.cpp · llm.c", { t: "same 33 token ids", color: C.grn, bold: true }],
      ["MNIST, 600 SGD steps", "bend-ml v0.3–v1.2.2 · PyTorch · NumPy · dfdx · candle", { t: "loss 0.520477", color: C.grn, bold: true }],
      ["GPT-2 logits, 11 prompts", "bend-ml v0.3 to v1.2.1", { t: `within ${E.gpt2_worst_logit_diff_by_release["v1.2.1"].toExponential(1)}`, color: C.b01 }],
      ["", "bend-ml v1.2.2 (four partial sums)", { t: `within ${E.gpt2_worst_logit_diff_by_release["v1.2.2"].toExponential(1)}`, color: C.vio, bold: true }],
    ], { size: 24, widths: [380, 860, 300] }),
    takeaway: `the speed numbers that follow compare <span class="g">the same work</span>`,
    foot: "logits are of order 100 · " + FOOT,
  }) });

  const rels = ["v0.3.0", "v1.0.0", "v1.1.0", "v1.2.0", "v1.2.1", "v1.2.2"];
  const subs = { "v0.3.0": "flat Array", "v1.0.0": "first stable", "v1.1.0": "row bands", "v1.2.0": "faster loading", "v1.2.1": "one-thread fix", "v1.2.2": "4 partial sums" };
  slides.push({ name: "history", html: frame({
    kicker: "GPT-2 small · ms per generated token, every release",
    title: `seven releases, <span class="v">one night</span>`,
    body: vbars({ w: 1580, h: 520, max: 200, fmt: (v) => `${Math.round(v)}`,
      legend: [["1 thread", "bend"], ["16 threads", "bend2"]],
      groups: [{ label: "v0.2", sub: "lists", bars: [{ v: H["v0.2.0"][1], kind: "bend" }, { v: H["v0.2.0"][16], kind: "bend2" }] },
        ...rels.map((r) => ({ label: r.replace(/\.0$/, ""), sub: subs[r], bend: r === "v1.2.2", bars: [{ v: H[r][1], kind: "bend" }, { v: H[r][16], kind: "bend2" }] }))] }),
    takeaway: `lists → Array: ${x(H["v0.2.0"][1], H["v0.3.0"][1])} · since then ${Math.round(H["v0.3.0"][1])} → ${Math.round(H["v1.2.2"][1])} ms on one thread, ${Math.round(H["v0.3.0"][16])} → ${Math.round(H["v1.2.2"][16])} on 16`,
    foot: `v0.2 is off the chart (hatched): ${H["v0.2.0"][1]} ms · ` + FOOT,
  }) });

  const prow = (t) => [
    { label: "PyTorch 2.14", sub: "eager, KV cache", v: P.PyTorch[t] },
    { label: "NumPy 2.5", sub: "OpenBLAS, KV cache", v: P.NumPy[t] },
    { label: "llama.cpp", sub: "F32", v: P["llama.cpp"][t] },
    { label: "bend-ml v1.2.2", sub: "Bend 2.0.36", v: P["bend-ml v1.2.2"][t], kind: "bend" },
    { label: "llm.c", sub: "no KV cache", v: P["llm.c"][t], note: t === 1 ? "reruns the whole window" : "" },
  ];
  slides.push({ name: "vs-references", html: frame({
    kicker: "GPT-2 small · ms per generated token · lower is better",
    title: `where Bend stands`,
    body: `<div style="display:flex;gap:60px">` + hbars({ w: 820, rowH: 66, labelW: 330, max: 200, fmt: ms, title: "1 thread", items: prow(1) })
      + hbars({ w: 820, rowH: 66, labelW: 330, max: 200, fmt: ms, title: "16 threads", items: prow(16) }) + `</div>`,
    takeaway: `PyTorch is ${x(P["bend-ml v1.2.2"][1], P.PyTorch[1])} ahead on one thread and ${x(P["bend-ml v1.2.2"][16], P.PyTorch[16])} on 16`,
    foot: "llm.c has no key/value cache: a different algorithm, shown for scale · " + FOOT,
  }) });

  const th = TC.threads;
  slides.push({ name: "threads", html: frame({
    kicker: "GPT-2 small · ms per generated token by thread count",
    title: `more cores: <span class="v">Bend keeps going</span>`,
    body: `<div style="display:flex;gap:30px">` +
      lines({ w: 820, h: 500, xs: th, xlog: true, ymax: 600, yfmt: (v) => `${v}`, xlabel: "threads", ylabel: "ms", series: [
        { name: "bend-ml", kind: "bend", ys: TC["bend-ml v1.2.2"] }, { name: "llm.c", kind: "dark", ys: TC["llm.c (no KV cache)"] }] }) +
      lines({ w: 820, h: 500, xs: th, xlog: true, ymax: 48, yfmt: (v) => `${v}`, xlabel: "threads", ylabel: "ms", series: [
        { name: "PyTorch", kind: "dark", ys: TC.PyTorch }, { name: "llama.cpp", kind: "other", ys: TC["llama.cpp"] }] }) + `</div>`,
    takeaway: `Bend keeps gaining up to all 22 threads`,
    foot: "two scales, two charts · " + FOOT,
  }) });

  slides.push({ name: "mnist", html: frame({
    kicker: "MNIST MLP 784-128-10 · one SGD step on a batch of 100 · 16 threads",
    title: `training: slower, and <span class="v">lighter</span>`,
    body: `<div style="display:flex;gap:60px">` +
      hbars({ w: 820, rowH: 64, labelW: 300, max: 14, fmt: (v) => ms(v), title: "ms per batch", items: [
        { label: "NumPy", v: M.NumPy[16] }, { label: "dfdx", sub: "Rust, shapes in types", v: M.dfdx[16] },
        { label: "PyTorch", v: M.PyTorch[16] }, { label: "candle", sub: "Rust", v: M.candle[16] },
        { label: "bend-ml v1.2.2", v: M["bend-ml v1.2.2"][16], kind: "bend" }] }) +
      hbars({ w: 820, rowH: 64, labelW: 300, max: 650, fmt: (v) => `${v} MB`, title: "peak resident memory", items: [
        { label: "NumPy", v: E.mnist_peak_mb_16t.NumPy }, { label: "dfdx", sub: "Rust, shapes in types", v: E.mnist_peak_mb_16t.dfdx },
        { label: "PyTorch", v: E.mnist_peak_mb_16t.PyTorch }, { label: "candle", sub: "Rust", v: E.mnist_peak_mb_16t.candle },
        { label: "bend-ml v1.2.2", v: E.mnist_peak_mb_16t["bend-ml v1.2.2"], kind: "bend" }] }) + `</div>`,
    takeaway: `behind on speed, the smallest footprint of all`,
    foot: `one thread: Bend ${M["bend-ml v1.2.2"][1]} ms, NumPy ${M.NumPy[1]} ms · ` + FOOT,
  }) });

  return slides;
};
