// Renders the real command outputs (outputs/*.txt) as Bend-style code blocks on paper cards,
// at 2x, into shots/. Usage: node shoot.mjs [name ...]
import puppeteer from "puppeteer-core";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { blockHTML, esc } from "./term.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const read = (f) => fs.readFileSync(path.join(here, "outputs", f), "utf8").replace(/\n+$/, "");
export const browserPath = process.env.BRAVE || "/usr/bin/brave";

const gpt2 = () => {
  const ls = read("gpt2.txt").split("\n");
  const ids = ls.filter((l) => /^\s*id /.test(l)).map((l) => l.match(/id (\d+)/)[1]);
  const text = ls.find((l) => l.startsWith("text:"));
  return [ls[0], `generated ids: ${ids.join(" ")}`, text].join("\n");
};

// name, command label, text, lines to mark, caption and its class, width
const SHOTS = [
  { name: "shape-error", cmd: "bend tensor/tests/bad_matmul.bend", text: () => read("bad_matmul.txt").split("\n").slice(0, 4).join("\n"), mark: [2, 3],
    cap: "(2×3) · (4×5): the inner dimensions differ, so the program is rejected before it runs", capCls: "r", width: 860 },
  { name: "reshape-error", cmd: "bend tensor/tests/bad_reshape.bend", text: () => read("bad_reshape.txt").split("\n").slice(0, 4).join("\n"), mark: [2, 3],
    cap: "reshape from 2×6 to 5×3 needs a proof that 12 = 15, which does not exist", capCls: "r", width: 860 },
  { name: "law-roundtrip", cmd: "python reference/list_laws.py && bend bpe/main.bend --verdict", text: () => read("law_roundtrip.txt") + "\n\n" + read("verdict.txt"),
    cap: "the statement the kernel checked: decoding what was encoded gives the bytes back", capCls: "g", width: 980 },
  { name: "gpt2", cmd: 'gpt2_fast "The capital of France is" 8', text: gpt2,
    cap: "GPT-2 small in Bend: the same 8 tokens as PyTorch, about 0.1 s each", capCls: "g", width: 980 },
];

export async function shoot(only) {
  const browser = await puppeteer.launch({ executablePath: browserPath, headless: "new", args: ["--no-sandbox", "--allow-file-access-from-files"] });
  try {
    for (const s of SHOTS) {
      if (only && !only.includes(s.name)) continue;
      const page = await browser.newPage();
      await page.setViewport({ width: s.width + 120, height: 900, deviceScaleFactor: 2 });
      const html = `<!doctype html><meta charset="utf-8"><link rel="stylesheet" href="${pathToFileURL(path.join(here, "theme.css"))}">
<body style="background:transparent;padding:30px"><div id="w" class="card" style="width:${s.width}px">${blockHTML({ cmd: s.cmd, text: s.text(), mark: s.mark || [] })}<div class="cap ${s.capCls || ""}">${esc(s.cap)}</div></div>`;
      const tmp = path.join(here, ".tmp-shot.html");
      fs.writeFileSync(tmp, html);
      await page.goto(pathToFileURL(tmp).href);
      await page.evaluate(() => document.fonts.ready);
      await (await page.$("#w")).screenshot({ path: path.join(here, "shots", s.name + ".png"), omitBackground: true });
      await page.close();
      console.log("shot", s.name);
    }
  } finally {
    await browser.close();
    fs.rmSync(path.join(here, ".tmp-shot.html"), { force: true });
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) await shoot(process.argv.slice(2).length ? process.argv.slice(2) : null);
