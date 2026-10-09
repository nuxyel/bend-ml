// Builds the README figures as SVG from numbers.json, in Bend's visual language (bend-lang.com):
// warm paper, one monospace family (embedded, so GitHub renders it), violet for Bend, grey for
// the rest, hatched bars for values off the chart.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const N = JSON.parse(fs.readFileSync(path.join(here, "numbers.json"), "utf8"));
const out = (name, svg) => fs.writeFileSync(path.join(here, name), svg);
const b64 = (f) => fs.readFileSync(path.join(here, "fonts", f)).toString("base64");
const FONT = `@font-face{font-family:"Bend ML Mono";font-weight:400;src:url(data:font/woff2;base64,${b64("BendMLMono-Regular.woff2")}) format("woff2")}
@font-face{font-family:"Bend ML Mono";font-weight:700;src:url(data:font/woff2;base64,${b64("BendMLMono-Bold.woff2")}) format("woff2")}`;

const C = { bg: "#f2eee7", bg2: "#e9e4db", line: "#e4dfd5", rule: "#d9d3c7", b1: "#a9a59d", b00: "#87847d", b01: "#69665f", b02: "#4d4a44",
  vio: "#8b83b5", grn: "#7e9a5e", red: "#c46a60", ora: "#c4845c", bar: "#cdc7bc" };
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const T = (x, y, s, { size = 15, fill = C.b01, weight = 400, anchor = "start", extra = "" } = {}) =>
  `<text x="${x}" y="${y}" font-size="${size}" fill="${fill}" font-weight="${weight}" text-anchor="${anchor}" ${extra}>${esc(s)}</text>`;
const svg = (w, h, title, body) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="${esc(title)}">
<title>${esc(title)}</title>
<style>${FONT}
text{font-family:"Bend ML Mono",ui-monospace,Menlo,"SF Mono",Consolas,"Liberation Mono",monospace}</style>
<defs><pattern id="hatch" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)"><rect width="9" height="9" fill="#efebe4"/><rect width="4" height="9" fill="#e2dcd1"/></pattern></defs>
<rect width="${w}" height="${h}" rx="14" fill="${C.bg}"/>
${body}
</svg>
`;
const fmt = (v) => (v < 0.1 ? `${Math.round(v * 1000)} ms` : v >= 10 ? `${v.toFixed(0)} s` : v >= 1 ? `${+v.toFixed(1)} s` : `${+v.toFixed(2)} s`);

// one Bend-style vertical bar chart: bars = [{label, sub, v, kind: "bend"|"other"|"over"}]
function bars(x0, y0, w, h, title, max, items) {
  const n = items.length, gap = 18, bw = (w - gap * (n - 1)) / n;
  let s = T(x0 + w / 2, y0, title, { size: 17, fill: C.b02, weight: 700, anchor: "middle" });
  const base = y0 + 40 + h;
  s += `<line x1="${x0 - 6}" y1="${base}" x2="${x0 + w + 6}" y2="${base}" stroke="${C.rule}" stroke-width="1"/>`;
  items.forEach((it, i) => {
    const x = x0 + i * (bw + gap), over = it.v > max, bh = over ? h : Math.max(2, (it.v / max) * h);
    const fill = over ? "url(#hatch)" : it.kind === "bend" ? C.vio : C.bar;
    s += `<rect x="${x.toFixed(1)}" y="${(base - bh).toFixed(1)}" width="${bw.toFixed(1)}" height="${bh.toFixed(1)}" fill="${fill}"/>`;
    if (over) s += `<path d="M${x} ${base - h + 14} l${bw / 4} -8 l${bw / 4} 8 l${bw / 4} -8 l${bw / 4} 8" fill="none" stroke="${C.bg}" stroke-width="5"/>`;
    const col = it.kind === "bend" ? C.vio : C.b01;
    s += T(x + bw / 2, base - bh - 10, fmt(it.v), { size: 15, fill: col, weight: it.kind === "bend" ? 700 : 400, anchor: "middle" });
    s += T(x + bw / 2, base + 24, it.label, { size: 14, fill: it.kind === "bend" ? C.vio : C.b00, weight: it.kind === "bend" ? 700 : 400, anchor: "middle" });
    if (it.sub) s += T(x + bw / 2, base + 42, it.sub, { size: 13, fill: C.b1, anchor: "middle" });
  });
  return s;
}

const m = N.mnist_epoch_seconds, g = N.gpt2_seconds_per_token, R = N.ratios;
{
  let b = bars(60, 54, 400, 250, "MNIST · one training epoch", 8, [
    { label: "v0.2", sub: "lists", v: m.bend_v1_lists, kind: "over" },
    { label: "v0.3", sub: "Array", v: m.bend_v2_array, kind: "bend" },
    { label: "PyTorch", sub: "16 threads", v: m.pytorch, kind: "other" },
  ]);
  b += bars(560, 54, 400, 250, "GPT-2 small · time per token", 0.12, [
    { label: "v0.2", sub: "lists", v: g.bend_v1_lists, kind: "over" },
    { label: "v0.3", sub: "Array", v: g.bend_v2_array, kind: "bend" },
    { label: "PyTorch", sub: "1 thread", v: g.pytorch_1_thread, kind: "other" },
    { label: "PyTorch", sub: "16 threads", v: g.pytorch_16_threads, kind: "other" },
  ]);
  b += T(510, 432, "seconds, lower is better · same machine · hatched bars are off the chart", { size: 14, fill: C.b1, anchor: "middle" });
  b += T(510, 462, `v0.2 → v0.3: ${Math.round(m.bend_v1_lists / m.bend_v2_array)}× and ${Math.round(g.bend_v1_lists / g.bend_v2_array)}× faster. PyTorch is still ~${R.mnist_pytorch_ahead_max}× and ~${R.gpt2_pytorch_ahead_min}–${R.gpt2_pytorch_ahead_max}× ahead.`, { size: 15, fill: C.ora, anchor: "middle" });
  out("benchmarks.svg", svg(1020, 490, "Benchmarks: MNIST epoch and GPT-2 time per token, Bend v0.2, Bend v0.3 and PyTorch", b));
}

{
  const rows = [
    [`${N.list_vs_array_speedup.value}×`, "faster", C.vio, "lists → one flat Array", "one thread, same matrix product"],
    ["1.8×", "faster", C.vio, "parallel row blocks", "the MNIST product, 8 threads"],
    [`${N.gpu.compute_bound_speedup}×`, "faster", C.vio, "GPU on a flat numeric loop", "16384 leaves, compute-bound"],
    [`${N.gpu.memory_bound_slowdown_min}–${N.gpu.memory_bound_slowdown_max}×`, "slower", C.red, "GPU on our matrix kernels", "memory-bound: pointer chasing"],
    ["10×", "slower", C.red, "copying the matrix per task", "the copy costs more than the arithmetic"],
  ];
  let b = T(50, 58, "what moved the needle", { size: 17, fill: C.b02, weight: 700 }) + T(970, 58, "each line is an experiment in NOTES.md", { size: 13, fill: C.b1, anchor: "end" });
  rows.forEach(([f, word, col, what, why], i) => {
    const y = 112 + i * 54;
    b += `<line x1="50" y1="${y - 32}" x2="970" y2="${y - 32}" stroke="${C.line}"/>`;
    b += T(50, y, f, { size: 26, fill: col, weight: 700 }) + T(178, y, word, { size: 15, fill: col });
    b += T(300, y, what, { size: 17, fill: C.b02 }) + T(970, y, why, { size: 14, fill: C.b00, anchor: "end" });
  });
  out("findings.svg", svg(1020, 112 + rows.length * 54 - 10, "What moved the needle: each measured change and its effect", b));
}

{
  const W = 1020, H = 560, bh = 92;
  const P = {
    nat: [40, 226, 236], bpe: [356, 64, 300], ten: [356, 178, 300], aut: [356, 292, 300], arr: [356, 406, 300],
    gpt: [736, 150, 244], mni: [736, 330, 244],
  };
  const box = (k, name, sub, marks) => {
    const [x, y, w] = P[k];
    let s = `<rect x="${x}" y="${y}" width="${w}" height="${bh}" rx="6" fill="${C.bg}" stroke="${C.rule}"/>`;
    s += T(x + 18, y + 30, name, { size: 17, fill: C.b02, weight: 700 }) + T(x + 18, y + 54, sub, { size: 13, fill: C.b00 });
    let mx = x + 18;
    for (const [t, col] of marks) { s += T(mx, y + 78, t, { size: 13, fill: col, weight: 700 }); mx += t.length * 8 + 18; }
    return s;
  };
  const R2 = (k) => [P[k][0] + P[k][2], P[k][1] + bh / 2], L2 = (k) => [P[k][0], P[k][1] + bh / 2];
  let b = `<defs><marker id="ar" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L8 4L0 8z" fill="${C.b1}"/></marker></defs>`;
  for (const [a, c] of [["nat", "bpe"], ["nat", "ten"], ["nat", "aut"], ["bpe", "gpt"], ["ten", "gpt"], ["arr", "gpt"], ["arr", "mni"], ["ten", "mni"]]) {
    const [x1, y1] = R2(a), [x2, y2] = L2(c), mx = (x1 + x2) / 2;
    b += `<path d="M${x1} ${y1} C${mx} ${y1} ${mx} ${y2} ${x2 - 2} ${y2}" fill="none" stroke="${C.b1}" stroke-width="1.2" marker-end="url(#ar)"/>`;
  }
  b += `<path d="M${P.ten[0] + 150} ${P.ten[1] + bh} L${P.aut[0] + 150} ${P.aut[1] - 2}" stroke="${C.b1}" stroke-width="1.2" marker-end="url(#ar)"/>`;
  b += T(40, 40, `${N.project.packages} packages on BendHub · ${N.project.laws} laws proved`, { size: 17, fill: C.b02, weight: 700 });
  b += box("nat", "nat-lemmas", "Nat and List lemmas", [["15 laws", C.grn]]);
  b += box("bpe", "bpe-tokenizer", "byte-level BPE, GPT-2 style", [["5 laws", C.grn], ["fuzzed", C.b00]]);
  b += box("ten", "tensor", "Mat<r,c> over lists", [["2 laws", C.grn], ["vs PyTorch", C.b00]]);
  b += box("aut", "autograd", "reverse mode = forward mode", [["1 law", C.grn], ["gradcheck", C.b00]]);
  b += box("arr", "tensor-array", "Mat<r,c> over a flat Array", [["cap_ok", C.grn], ["vs PyTorch", C.b00], ["Array.new", C.ora]]);
  b += box("gpt", "GPT-2 demo", "same tokens as PyTorch", [["vs tiktoken", C.b00]]);
  b += box("mni", "MNIST demo", "typed training step", [["vs PyTorch", C.b00]]);
  b += T(40, H - 26, "proved by the kernel", { size: 13, fill: C.grn, weight: 700 }) + T(230, H - 26, "tested against a reference", { size: 13, fill: C.b00, weight: 700 }) + T(470, H - 26, "trusted", { size: 13, fill: C.ora, weight: 700 }) + T(980, H - 26, "arrows: imports", { size: 13, fill: C.b1, anchor: "end" });
  out("diagram.svg", svg(W, H, "The five packages, how they import each other, and what is proved, tested or trusted", b));
}

{
  const W = 1200, H = 360;
  let cells = "";
  const cs = 22, gp = 6, gx = W / 2 - (3 * cs + 2 * gp) / 2, gy = 52;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) cells += `<rect x="${gx + c * (cs + gp)}" y="${gy + r * (cs + gp)}" width="${cs}" height="${cs}" rx="4" fill="${r === 1 && c === 2 ? C.vio : C.bar}"/>`;
  let b = cells;
  // the wordmark and Bend's violet block cursor, centred together (the font advance is 0.6 em)
  const fs = 92, tw = 7 * 0.6 * fs, cw = 0.5 * fs, gapc = 0.12 * fs, x0 = W / 2 - (tw + gapc + cw) / 2;
  b += T(x0, 214, "bend-ml", { size: fs, fill: C.b02, weight: 700 });
  b += `<rect x="${(x0 + tw + gapc).toFixed(1)}" y="${214 - 0.78 * fs}" width="${cw}" height="${0.86 * fs}" fill="${C.vio}"/>`;
  b += T(W / 2, 270, "machine learning in Bend 2", { size: 26, fill: C.b00, anchor: "middle" });
  b += T(W / 2, 310, "shape errors are type errors · laws checked by the kernel", { size: 18, fill: C.b1, anchor: "middle" });
  out("banner.svg", svg(W, H, "bend-ml: machine learning in Bend 2. Shape errors are type errors; laws checked by the kernel.", b));
}
console.log("figures ok");
