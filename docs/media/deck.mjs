// Renders a deck of 1920x1080 frames (WebP; needs ffmpeg) in bend-ml's visual identity (the same paper, type and colours as the
// launch video): a slides module describes each frame with the helpers below, and every frame becomes a PNG.
//
//   node deck.mjs SLIDES.mjs OUT_DIR        (SLIDES.mjs default-exports (helpers) => [{ name, html }])
//
// The README deck is deck/readme.mjs (its numbers come from numbers.json). Colours: violet for Bend, warm grey for
// the reference systems, hatched bars for values off the chart, every bar labelled with its value.
import puppeteer from "puppeteer-core";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const browserPath = process.env.BRAVE || "/usr/bin/brave";
export const C = { bg: "#f2eee7", bg2: "#e9e4db", line: "#e4dfd5", rule: "#d9d3c7", b1: "#a9a59d", b00: "#87847d", b01: "#69665f",
  b02: "#4d4a44", vio: "#8b83b5", vioL: "#ddd8e7", grn: "#7e9a5e", red: "#c46a60", ora: "#c4845c", bar: "#cdc7bc", barD: "#b5ae9f" };
export const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

const HATCH = `<pattern id="hatch" width="12" height="12" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)"><rect width="12" height="12" fill="#efebe4"/><rect width="5" height="12" fill="#e2dcd1"/></pattern>
<pattern id="hatchv" width="12" height="12" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)"><rect width="12" height="12" fill="#e6e2ef"/><rect width="5" height="12" fill="#cfc9e1"/></pattern>`;
const T = (x, y, s, { size = 22, fill = C.b01, weight = 400, anchor = "start" } = {}) =>
  `<text x="${x}" y="${y}" font-size="${size}" fill="${fill}" font-weight="${weight}" text-anchor="${anchor}">${esc(s)}</text>`;

/** Horizontal bars, one row per item: { label, sub, v, kind: "bend" | "other", note }. Values above max are hatched. */
export function hbars({ w = 1500, rowH = 58, items, max, fmt = (v) => `${v}`, labelW = 470, title }) {
  const h = items.length * rowH + (title ? 50 : 0) + 10;
  const x0 = labelW, bw = w - labelW - 190;
  let s = title ? T(0, 30, title, { size: 24, fill: C.b02, weight: 700 }) : "";
  const top = title ? 50 : 0;
  items.forEach((it, i) => {
    const y = top + i * rowH, bend = it.kind === "bend", over = it.v > max;
    const len = over ? bw : Math.max(3, (it.v / max) * bw);
    s += T(x0 - 24, y + rowH / 2 + (it.sub ? -2 : 8), it.label, { size: 23, fill: bend ? C.vio : C.b02, weight: bend ? 700 : 400, anchor: "end" });
    if (it.sub) s += T(x0 - 24, y + rowH / 2 + 22, it.sub, { size: 17, fill: C.b1, anchor: "end" });
    s += `<rect x="${x0}" y="${y + 9}" width="${len.toFixed(1)}" height="${rowH - 18}" rx="4" fill="${over ? (bend ? "url(#hatchv)" : "url(#hatch)") : bend ? C.vio : C.bar}"/>`;
    s += T(x0 + len + 14, y + rowH / 2 + 8, fmt(it.v), { size: 23, fill: bend ? C.vio : C.b01, weight: bend ? 700 : 400 });
    if (it.note) s += T(x0 + len + 14 + fmt(it.v).length * 14 + 18, y + rowH / 2 + 8, it.note, { size: 17, fill: C.b1 });
  });
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" style="overflow:visible"><defs>${HATCH}</defs>${s}</svg>`;
}

/** Grouped vertical bars: groups [{ label, sub, bars: [{ v, kind, tag }] }]; a legend names the bar kinds in order. */
export function vbars({ w = 1600, h = 520, groups, max, fmt = (v) => `${v}`, legend = [] }) {
  const n = groups.length, per = groups[0].bars.length, gap = 46, gw = (w - gap * (n - 1)) / n, inner = 8;
  const bwid = (gw - inner * (per - 1)) / per, base = h - 80;
  let s = `<line x1="0" y1="${base}" x2="${w}" y2="${base}" stroke="${C.rule}" stroke-width="2"/>`;
  groups.forEach((g, gi) => {
    const gx = gi * (gw + gap);
    g.bars.forEach((b, bi) => {
      const x = gx + bi * (bwid + inner), over = b.v > max, bh = over ? base - 40 : Math.max(3, (b.v / max) * (base - 40));
      const bend = b.kind === "bend", light = b.kind === "bend2";
      const fill = over ? (bend || light ? "url(#hatchv)" : "url(#hatch)") : bend ? C.vio : light ? "#b7b0d6" : b.kind === "other2" ? C.barD : C.bar;
      s += `<rect x="${x.toFixed(1)}" y="${(base - bh).toFixed(1)}" width="${bwid.toFixed(1)}" height="${bh.toFixed(1)}" rx="4" fill="${fill}"/>`;
      s += T(x + bwid / 2, base - bh - 12, fmt(b.v), { size: 20, fill: bend || light ? C.vio : C.b01, weight: bend ? 700 : 400, anchor: "middle" });
    });
    s += T(gx + gw / 2, base + 36, g.label, { size: 23, fill: g.bend ? C.vio : C.b02, weight: g.bend ? 700 : 400, anchor: "middle" });
    if (g.sub) s += T(gx + gw / 2, base + 64, g.sub, { size: 17, fill: C.b1, anchor: "middle" });
  });
  let lx = 0;
  for (const [name, kind] of legend) {
    const fill = kind === "bend" ? C.vio : kind === "bend2" ? "#b7b0d6" : kind === "other2" ? C.barD : C.bar;
    s += `<rect x="${lx}" y="-40" width="22" height="22" rx="4" fill="${fill}"/>` + T(lx + 32, -22, name, { size: 19, fill: C.b01 });
    lx += 32 + name.length * 12 + 40;
  }
  return `<svg width="${w}" height="${h}" viewBox="0 -50 ${w} ${h + 50}" style="overflow:visible"><defs>${HATCH}</defs>${s}</svg>`;
}

/** Lines over x values: series [{ name, kind, ys }] (null = no point); direct labels at the right end. */
export function lines({ w = 1500, h = 560, xs, series, ymax, ymin = 0, yfmt = (v) => `${v}`, xlabel = "", ylabel = "", xfmt = (v) => `${v}`, xlog = false }) {
  const L = 110, R = 330, B = 70, Tp = 20, pw = w - L - R, ph = h - B - Tp;
  const fx = (x) => L + (xlog ? (Math.log(x) - Math.log(xs[0])) / (Math.log(xs[xs.length - 1]) - Math.log(xs[0])) : (x - xs[0]) / (xs[xs.length - 1] - xs[0])) * pw;
  const fy = (y) => Tp + ph - ((Math.min(y, ymax) - ymin) / (ymax - ymin)) * ph;
  let s = "";
  for (let k = 0; k <= 4; k++) {
    const v = ymin + ((ymax - ymin) * k) / 4, y = fy(v);
    s += `<line x1="${L}" y1="${y}" x2="${L + pw}" y2="${y}" stroke="${C.line}" stroke-width="1.5"/>` + T(L - 16, y + 7, yfmt(v), { size: 18, fill: C.b1, anchor: "end" });
  }
  for (const x of xs) s += T(fx(x), Tp + ph + 34, xfmt(x), { size: 19, fill: C.b00, anchor: "middle" });
  if (xlabel) s += T(L + pw / 2, h + 8, xlabel, { size: 19, fill: C.b1, anchor: "middle" });
  if (ylabel) s += T(L - 16, Tp - 4 - 14, ylabel, { size: 18, fill: C.b1, anchor: "end" });
  const ends = [];
  for (const se of series) {
    const bend = se.kind === "bend", col = bend ? C.vio : se.kind === "dark" ? C.b00 : C.barD;
    const pts = xs.map((x, i) => (se.ys[i] == null ? null : [fx(x), fy(se.ys[i])])).filter(Boolean);
    s += `<polyline points="${pts.map((p) => p.map((v) => v.toFixed(1)).join(",")).join(" ")}" fill="none" stroke="${col}" stroke-width="${bend ? 5 : 3}" stroke-linejoin="round" stroke-linecap="round"/>`;
    for (const [x, y] of pts) s += `<circle cx="${x}" cy="${y}" r="${bend ? 7 : 5.5}" fill="${col}" stroke="${C.bg}" stroke-width="2"/>`;
    const last = pts[pts.length - 1];
    ends.push({ y: last[1], name: se.name, col: bend ? C.vio : C.b01, bold: bend, x: last[0] });
  }
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 30) ends[i].y = ends[i - 1].y + 30;
  for (const e of ends) s += T(L + pw + 22, e.y + 7, e.name, { size: 20, fill: e.col, weight: e.bold ? 700 : 400 });
  return `<svg width="${w}" height="${h + 20}" viewBox="0 0 ${w} ${h + 20}" style="overflow:visible">${s}</svg>`;
}

/** A row of big numbers: [{ big, label, sub, color }]. */
export const stats = (items, size = 92) => `<div style="display:flex;gap:90px;justify-content:center">${items.map((it) => `<div style="text-align:center">
<div style="font-size:${size}px;font-weight:700;color:${it.color || C.vio};line-height:1">${esc(it.big)}</div>
<div style="font-size:26px;color:${C.b02};margin-top:18px;white-space:nowrap">${esc(it.label)}</div>${it.sub ? `<div style="font-size:19px;color:${C.b1};margin-top:8px">${esc(it.sub)}</div>` : ""}</div>`).join("")}</div>`;

/** A simple table: header row, then rows; cells may be { t, color, bold }. */
export const table = (head, rows, { size = 21, widths = [] } = {}) => `<table style="border-collapse:collapse;font-size:${size}px;margin:0 auto">
<tr>${head.map((h, i) => `<th style="white-space:nowrap;text-align:${i ? "right" : "left"};color:${C.b1};font-weight:400;padding:6px 22px;border-bottom:2px solid ${C.rule};${widths[i] ? `width:${widths[i]}px` : ""}">${esc(h)}</th>`).join("")}</tr>
${rows.map((r) => `<tr>${r.map((c, i) => { const o = typeof c === "object" && c ? c : { t: c }; return `<td style="white-space:nowrap;text-align:${i ? "right" : "left"};padding:7px 22px;border-bottom:1px solid ${C.line};color:${o.color || (i ? C.b01 : C.b02)};font-weight:${o.bold ? 700 : 400}">${esc(o.t ?? "")}</td>`; }).join("")}</tr>`).join("\n")}</table>`;

/** One frame: a kicker, a title, the body, a takeaway line and a footnote. */
export function frame({ kicker = "", title = "", body = "", takeaway = "", foot = "", page = "", align = "center" }) {
  return `<div class="f">
${kicker ? `<div class="kick">${esc(kicker)}</div>` : ""}${title ? `<div class="title">${title}</div>` : ""}
<div class="body" style="justify-content:${align}">${body}</div>
${takeaway ? `<div class="take">${takeaway}</div>` : ""}${foot ? `<div class="foot">${esc(foot)}</div>` : ""}${page ? `<div class="page">${esc(page)}</div>` : ""}
</div>`;
}

const CSS = `
html,body{width:1920px;height:1080px;overflow:hidden}
body{background:${C.bg}}
.f{position:absolute;inset:0;padding:96px 150px 70px;display:flex;flex-direction:column;
  background:radial-gradient(ellipse at center, transparent 58%, rgba(77,74,68,.13) 100%), ${C.bg}}
.f::after{content:"";position:absolute;inset:0;pointer-events:none;opacity:.05;mix-blend-mode:multiply;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='240' height='240'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 .3  0 0 0 0 .28  0 0 0 0 .25  0 0 0 1 0'/></filter><rect width='240' height='240' filter='url(%23n)'/></svg>")}
.kick{font-size:22px;color:${C.b1};letter-spacing:.02em}
.kick::before{content:"$ ";color:${C.vio}}
.title{font-size:54px;font-weight:700;color:${C.b02};margin-top:14px;line-height:1.15}
.title .v{color:${C.vio}} .title .g{color:${C.grn}} .title .o{color:${C.ora}}
.body{flex:1;display:flex;flex-direction:column;align-items:center;margin-top:30px;gap:26px}
.take{font-size:33px;font-weight:700;color:${C.ora};text-align:center;margin-top:18px}
.take .g{color:${C.grn}} .take .v{color:${C.vio}}
.foot{font-size:19px;color:${C.b00};text-align:center;margin-top:14px;line-height:1.5}
.page{position:absolute;right:60px;bottom:34px;font-size:17px;color:${C.b1}}
.p{font-size:26px;color:${C.b01};line-height:1.55;max-width:1560px}
.p b{color:${C.b02}} .p .v{color:${C.vio};font-weight:700} .p .g{color:${C.grn};font-weight:700} .p .o{color:${C.ora};font-weight:700} .p .r{color:${C.red};font-weight:700}
.mono{background:${C.bg2};border-radius:6px;padding:22px 30px;font-size:24px;line-height:1.6;white-space:pre;color:${C.b01}}
`;

export async function render(slidesFile, outDir) {
  const mod = await import(pathToFileURL(path.resolve(slidesFile)).href);
  const slides = await mod.default({ frame, hbars, vbars, lines, stats, table, C, esc });
  fs.mkdirSync(outDir, { recursive: true });
  const theme = pathToFileURL(path.join(here, "theme.css")).href;
  const browser = await puppeteer.launch({ executablePath: browserPath, headless: "new", args: ["--no-sandbox", "--allow-file-access-from-files"] });
  const files = [];
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });
    for (const [i, s] of slides.entries()) {
      const html = `<!doctype html><meta charset="utf-8"><link rel="stylesheet" href="${theme}"><style>${CSS}</style><body>${s.html}</body>`;
      const tmp = path.join(here, ".tmp-deck.html");
      fs.writeFileSync(tmp, html);
      await page.goto(pathToFileURL(tmp).href);
      await page.evaluate(() => document.fonts.ready);
      // a PNG screenshot, kept as WebP (about 60 KB instead of 1 MB: the paper grain does not compress as PNG)
      const name = `${String(i + 1).padStart(2, "0")}-${s.name}.webp`, png = path.join(outDir, ".frame.png");
      await page.screenshot({ path: png });
      execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-i", png, "-c:v", "libwebp", "-quality", "90", "-compression_level", "6", path.join(outDir, name)]);
      fs.rmSync(png);
      files.push(name);
      console.log("frame", name);
    }
  } finally {
    await browser.close();
    fs.rmSync(path.join(here, ".tmp-deck.html"), { force: true });
  }
  // a page that shows every frame in order, to read the deck as a document
  const title = mod.title || "deck";
  fs.writeFileSync(path.join(outDir, "index.html"), `<!doctype html><meta charset="utf-8"><title>${esc(title)}</title>
<style>body{background:#e9e4db;margin:0;padding:40px 0;font-family:ui-monospace,monospace;color:#69665f}img{display:block;width:min(1600px,96vw);margin:0 auto 40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,.12)}h1{text-align:center;font-size:20px;font-weight:400}</style>
<h1>${esc(title)}</h1>${files.map((f) => `<img src="${f}" alt="${esc(f)}">`).join("\n")}`);
  return files;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const [slides, out] = process.argv.slice(2);
  if (!slides || !out) { console.error("usage: node deck.mjs SLIDES.mjs OUT_DIR"); process.exit(2); }
  await render(slides, out);
}
