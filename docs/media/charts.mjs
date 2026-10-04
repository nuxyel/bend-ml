// Builds the README charts, the package diagram and the banner as SVG from numbers.json.
// Every figure sits on its own dark card, so it reads the same on GitHub's light and dark themes.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const N = JSON.parse(fs.readFileSync(path.join(here, "numbers.json"), "utf8"));
const out = (name, svg) => fs.writeFileSync(path.join(here, name), svg);

const C = { bg: "#0b0f14", panel: "#121820", border: "#2a3441", fg: "#e6edf3", dim: "#8b98a8",
  red: "#ff6b6b", green: "#3ddc84", blue: "#5aa9ff", amber: "#f0b429", purple: "#b794f6", cyan: "#4dd0e1" };
const SANS = "Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif";
const MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const fmt = (v) => (v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v >= 0.1 ? v.toFixed(2).replace(/0$/, "") : (v * 1000).toFixed(0) + " ms").replace(/(\d)$/, "$1");
const sec = (v) => (v < 0.1 ? `${Math.round(v * 1000)} ms` : v >= 10 ? `${v.toFixed(0)} s` : v >= 1 ? `${v.toFixed(1)} s` : `${String(+v.toFixed(2))} s`);

const card = (w, h, body, title, subtitle) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="${esc(title)}">
<title>${esc(title)}</title>
<rect x="0.5" y="0.5" width="${w - 1}" height="${h - 1}" rx="16" fill="${C.panel}" stroke="${C.border}"/>
<text x="28" y="40" font-family="${SANS}" font-size="20" font-weight="700" fill="${C.fg}">${esc(title)}</text>
<text x="28" y="64" font-family="${SANS}" font-size="14" fill="${C.dim}">${esc(subtitle)}</text>
${body}
</svg>
`;

// horizontal bars on a log scale: rows = [{label, value, color, note}]
function logBars({ title, subtitle, rows, unitNote, caption }) {
  const W = 900, left = 250, right = 120, top = 92, rowH = 52, barH = 26;
  const H = top + rows.length * rowH + (caption ? 70 : 34);
  const vals = rows.map((r) => r.value);
  const lo = Math.log10(Math.min(...vals) / 3), hi = Math.log10(Math.max(...vals));
  const x = (v) => left + ((Math.log10(v) - lo) / (hi - lo)) * (W - left - right);
  let body = "";
  rows.forEach((r, i) => {
    const y = top + i * rowH;
    const bw = Math.max(6, x(r.value) - left);
    body += `<text x="28" y="${y + 18}" font-family="${SANS}" font-size="15" fill="${C.fg}">${esc(r.label)}</text>`;
    body += `<rect x="${left}" y="${y}" width="${bw.toFixed(1)}" height="${barH}" rx="6" fill="${r.color}"/>`;
    body += `<text x="${(left + bw + 10).toFixed(1)}" y="${y + 19}" font-family="${MONO}" font-size="15" font-weight="700" fill="${r.color}">${esc(r.text ?? sec(r.value))}</text>`;
  });
  body += `<text x="${W - 28}" y="${H - (caption ? 52 : 14)}" text-anchor="end" font-family="${SANS}" font-size="12" fill="${C.dim}">${esc(unitNote)}</text>`;
  if (caption) body += `<text x="28" y="${H - 22}" font-family="${SANS}" font-size="14" fill="${C.amber}">${esc(caption)}</text>`;
  return card(W, H, body, title, subtitle);
}

const m = N.mnist_epoch_seconds, g = N.gpt2_seconds_per_token;
out("chart-mnist.svg", logBars({
  title: "MNIST: one training epoch (784-128-10 MLP)",
  subtitle: "Same data, same initial weights, same batches: identical loss and hits in Bend and PyTorch",
  rows: [
    { label: "Bend v1 (lists)", value: m.bend_v1_lists, color: C.red },
    { label: "Bend v2 (flat Array, typed)", value: m.bend_v2_array, color: C.blue },
    { label: "PyTorch (16 threads, CPU)", value: m.pytorch, color: C.green },
  ],
  unitNote: "log scale",
  caption: `v1 → v2: ${Math.round(m.bend_v1_lists / m.bend_v2_array)}x faster. PyTorch is still ~${N.ratios.mnist_pytorch_ahead_min}–${N.ratios.mnist_pytorch_ahead_max}x ahead.`,
}));
out("chart-gpt2.svg", logBars({
  title: "GPT-2 small (124 M): time per generated token",
  subtitle: "Greedy decoding with a key/value cache; Bend generates the same tokens as PyTorch",
  rows: [
    { label: "Bend v1 (lists)", value: g.bend_v1_lists, color: C.red },
    { label: "Bend v2 (flat Array)", value: g.bend_v2_array, color: C.blue },
    { label: "PyTorch, 1 thread", value: g.pytorch_1_thread, color: C.green },
    { label: "PyTorch, 16 threads", value: g.pytorch_16_threads, color: C.green },
  ],
  unitNote: "log scale; PyTorch time is one forward pass",
  caption: `v1 → v2: ${Math.round(g.bend_v1_lists / g.bend_v2_array)}x faster. PyTorch is still ~${N.ratios.gpt2_pytorch_ahead_min}–${N.ratios.gpt2_pytorch_ahead_max}x ahead.`,
}));

// what moved the needle: multiplicative effects, log axis centred on 1x
{
  const W = 960, top = 92, rowH = 50, mid = 590, half = 230;
  const rows = [
    { label: "Lists → flat Array (1 thread)", f: N.list_vs_array_speedup.value, color: C.green },
    { label: "Parallel row blocks (MNIST product)", f: 1.8, color: C.green },
    { label: "GPU, compute-bound flat loop", f: N.gpu.compute_bound_speedup, color: C.green },
    { label: "GPU, our memory-bound kernels", f: 1 / N.gpu.memory_bound_slowdown_min, color: C.red, text: `${N.gpu.memory_bound_slowdown_min}–${N.gpu.memory_bound_slowdown_max}x slower` },
    { label: "Copying the matrix per task (matvec)", f: 1 / 10, color: C.red, text: "10x slower" },
  ];
  const span = Math.log10(60);
  const H = top + rows.length * rowH + 50;
  let body = `<line x1="${mid}" y1="${top - 12}" x2="${mid}" y2="${top + rows.length * rowH - 8}" stroke="${C.dim}" stroke-width="1" stroke-dasharray="3 4"/>`;
  body += `<text x="${mid + 6}" y="${top + rows.length * rowH + 6}" font-family="${SANS}" font-size="12" fill="${C.dim}">same speed (1x)</text>`;
  rows.forEach((r, i) => {
    const y = top + i * rowH, d = (Math.log10(r.f) / span) * half;
    const bx = d >= 0 ? mid : mid + d, bw = Math.max(6, Math.abs(d));
    body += `<text x="28" y="${y + 18}" font-family="${SANS}" font-size="15" fill="${C.fg}">${esc(r.label)}</text>`;
    body += `<rect x="${bx.toFixed(1)}" y="${y}" width="${bw.toFixed(1)}" height="26" rx="6" fill="${r.color}"/>`;
    const tx = d >= 0 ? bx + bw + 10 : bx - 10;
    body += `<text x="${tx.toFixed(1)}" y="${y + 19}" text-anchor="${d >= 0 ? "start" : "end"}" font-family="${MONO}" font-size="15" font-weight="700" fill="${r.color}">${esc(r.text ?? r.f + "x faster")}</text>`;
  });
  body += `<text x="28" y="${H - 18}" font-family="${SANS}" font-size="14" fill="${C.amber}">Measured, not guessed: each bar is an experiment in NOTES.md. Log scale; right is faster.</text>`;
  out("chart-findings.svg", card(W, H, body, "What moved the needle (and what did not)", "Effect of each change on run time, measured on this machine"));
}
console.log("charts ok");

// ---- package diagram ----
{
  const W = 960, H = 546;
  const box = (x, y, w, h, name, sub, chips, accent = C.green) => {
    let s = `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="${C.bg}" stroke="${C.border}"/>`;
    s += `<rect x="${x}" y="${y}" width="6" height="${h}" rx="3" fill="${accent}"/>`;
    s += `<text x="${x + 22}" y="${y + 28}" font-family="${MONO}" font-size="16" font-weight="700" fill="${C.fg}">${esc(name)}</text>`;
    s += `<text x="${x + 22}" y="${y + 50}" font-family="${SANS}" font-size="13" fill="${C.dim}">${esc(sub)}</text>`;
    let cx = x + 22;
    for (const [t, col] of chips) {
      const cw = t.length * 7.2 + 16;
      s += `<rect x="${cx}" y="${y + h - 26}" width="${cw}" height="18" rx="9" fill="${col}" fill-opacity="0.16" stroke="${col}" stroke-opacity="0.6"/>`;
      s += `<text x="${cx + cw / 2}" y="${y + h - 13}" text-anchor="middle" font-family="${SANS}" font-size="11" font-weight="600" fill="${col}">${esc(t)}</text>`;
      cx += cw + 6;
    }
    return s;
  };
  const edge = (x1, y1, x2, y2, col = C.dim) => {
    const mx = (x1 + x2) / 2;
    return `<path d="M${x1} ${y1} C${mx} ${y1} ${mx} ${y2} ${x2} ${y2}" fill="none" stroke="${col}" stroke-width="1.6" stroke-opacity="0.75" marker-end="url(#a)"/>`;
  };
  const bh = 82;
  const P = N.project;
  const pos = {
    nat: [24, 246, 250], bpe: [320, 86, 370], ten: [320, 186, 370], aut: [320, 286, 370], arr: [320, 386, 370],
    gpt: [724, 166, 212], mni: [724, 336, 212],
  };
  let body = `<defs><marker id="a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="${C.dim}"/></marker></defs>`;
  const R = (k) => [pos[k][0] + pos[k][2], pos[k][1] + bh / 2], L = (k) => [pos[k][0], pos[k][1] + bh / 2];
  for (const [a, b] of [["nat", "bpe"], ["nat", "ten"], ["nat", "aut"], ["ten", "aut"], ["bpe", "gpt"], ["ten", "gpt"], ["arr", "gpt"], ["arr", "mni"], ["ten", "mni"]]) {
    const [x1, y1] = R(a), [x2, y2] = L(b);
    if (a === "ten" && b === "aut") { body += `<path d="M${pos.ten[0] + 185} ${pos.ten[1] + bh} L${pos.aut[0] + 185} ${pos.aut[1]}" stroke="${C.dim}" stroke-width="1.6" stroke-opacity="0.75" marker-end="url(#a)"/>`; continue; }
    body += edge(x1, y1, x2, y2);
  }
  const bx = (k, ...rest) => box(pos[k][0], pos[k][1], pos[k][2], k === "gpt" || k === "mni" ? bh + 10 : bh, ...rest);
  body += bx("nat", "nat-lemmas", "Nat and List lemmas Base lacks", [["15 laws proved", C.green]]);
  body += bx("bpe", "bpe-tokenizer", "byte-level BPE, GPT-2 style", [["5 laws proved", C.green], ["fuzzed vs Python", C.blue]]);
  body += bx("ten", "tensor", "Mat<r,c> over lists, shape in the type", [["2 laws proved", C.green], ["vs PyTorch", C.blue]]);
  body += bx("aut", "autograd", "reverse mode = forward mode", [["1 law proved", C.green], ["gradient checking", C.blue]]);
  body += bx("arr", "tensor-array", "Mat<r,c> over a flat Array, ~49x faster", [["cap_ok proved", C.green], ["vs PyTorch", C.blue], ["Array.new trusted", C.amber]]);
  body += bx("gpt", "GPT-2 small demo", "same tokens as PyTorch", [["vs PyTorch + tiktoken", C.blue]], C.blue);
  body += bx("mni", "MNIST demo", "typed training step", [["identical to PyTorch", C.blue]], C.blue);
  // legend
  const ly = H - 22;
  const leg = [["proved by Bend's kernel (--verdict)", C.green], ["tested against a reference", C.blue], ["trusted (F32 primitives, Array.new)", C.amber]];
  let lx = 28;
  for (const [t, col] of leg) {
    body += `<circle cx="${lx + 6}" cy="${ly - 4}" r="6" fill="${col}"/><text x="${lx + 20}" y="${ly}" font-family="${SANS}" font-size="13" fill="${C.dim}">${esc(t)}</text>`;
    lx += 52 + t.length * 6.4;
  }
  out("diagram.svg", card(W, H, body, `${P.packages} packages, ${P.laws} proved laws`, "Who imports whom, and what is proved, tested or trusted"));
}

// ---- banner ----
{
  const W = 1200, H = 300;
  let cells = "";
  const cols = 14, rows = 5, size = 22, gap = 5, gx = W - 36 - cols * (size + gap), gy = 46;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const bad = c >= 9 && c < 12 && r >= 1 && r < 4; // a block that does not match
    const t = (c * 7 + r * 13) % 11;
    const col = bad ? C.red : t < 6 ? C.blue : t < 9 ? C.cyan : C.green;
    const op = bad ? 0.95 : 0.18 + ((c * 5 + r * 3) % 7) * 0.07;
    cells += `<rect x="${gx + c * (size + gap)}" y="${gy + r * (size + gap)}" width="${size}" height="${size}" rx="5" fill="${col}" fill-opacity="${op.toFixed(2)}"/>`;
  }
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="bend-ml: machine learning in Bend 2, where a shape error does not compile">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0d1420"/><stop offset="1" stop-color="#0b0f14"/></linearGradient>
<linearGradient id="t" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="${C.blue}"/><stop offset="1" stop-color="${C.green}"/></linearGradient></defs>
<rect width="${W}" height="${H}" rx="18" fill="url(#g)" stroke="${C.border}"/>
${cells}
<text x="48" y="128" font-family="${MONO}" font-size="76" font-weight="800" fill="url(#t)">bend-ml</text>
<text x="52" y="176" font-family="${SANS}" font-size="26" fill="${C.fg}">Machine learning in Bend 2,</text>
<text x="52" y="210" font-family="${SANS}" font-size="26" fill="${C.fg}">where a <tspan fill="${C.red}" font-weight="700">shape error</tspan> does not compile.</text>
<text x="52" y="256" font-family="${MONO}" font-size="16" fill="${C.dim}">${N.project.packages} packages · ${N.project.laws} proved laws · MNIST + GPT-2 · checked against PyTorch</text>
</svg>
`;
  out("banner.svg", svg);
}
console.log("diagram + banner ok");
