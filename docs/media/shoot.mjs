// Renders docs/media/outputs/*.txt as terminal-window PNGs (2x) in docs/media/shots/.
import puppeteer from "puppeteer-core";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const read = (f) => fs.readFileSync(path.join(here, "outputs", f), "utf8");

export const browserPath = process.env.BRAVE || "/usr/bin/brave";

// name, window title, the command shown after the prompt, output file, optional line filter
const SHOTS = [
  { name: "shape-error", title: "tensor/tests/bad_matmul.bend", cmd: "bend tensor/tests/bad_matmul.bend", file: "bad_matmul.txt", highlight: [2, 3], width: 1000 },
  { name: "reshape-error", title: "tensor/tests/bad_reshape.bend", cmd: "bend tensor/tests/bad_reshape.bend", file: "bad_reshape.txt", highlight: [2, 3], width: 1000 },
  { name: "verdict", title: "bpe/main.bend", cmd: "bend bpe/main.bend --verdict", file: "verdict.txt", width: 760 },
  { name: "law-roundtrip", title: "reference/list_laws.py", cmd: "python reference/list_laws.py", file: "law_roundtrip.txt", width: 900 },
  { name: "gpt2", title: "demos/gpt2/fast.bend", cmd: 'gpt2_fast "The capital of France is" 8', file: "gpt2.txt", width: 900 },
  { name: "mnist", title: "demos/mnist/fast.bend", cmd: "mnist_fast 1 0 0.1", file: "mnist.txt", width: 900 },
  { name: "check", title: "make check-full", cmd: "make check-full", file: "check_full.txt", width: 980, tail: 22 },
];

export async function shoot(only) {
  const { terminalHTML } = await import("./term.js");
  const browser = await puppeteer.launch({ executablePath: browserPath, headless: "new", args: ["--no-sandbox", "--allow-file-access-from-files"] });
  try {
    for (const s of SHOTS) {
      if (only && !only.includes(s.name)) continue;
      const f = path.join(here, "outputs", s.file);
      if (!fs.existsSync(f)) { console.log("skip", s.name, "(no output file)"); continue; }
      let text = read(s.file);
      if (s.tail) text = text.replace(/\n+$/, "").split("\n").slice(-s.tail).join("\n");
      const page = await browser.newPage();
      await page.setViewport({ width: s.width + 80, height: 900, deviceScaleFactor: 2 });
      const html = `<!doctype html><meta charset="utf-8"><link rel="stylesheet" href="${pathToFileURL(path.join(here, "theme.css"))}"><body style="padding:40px;background:#0b0f14"><div id="w" style="width:${s.width}px">${terminalHTML({ ...s, text })}</div>`;
      const tmp = path.join(here, ".tmp-shot.html");
      fs.writeFileSync(tmp, html);
      await page.goto(pathToFileURL(tmp).href);
      await page.evaluate(() => document.fonts.ready);
      const el = await page.$("#w");
      await el.screenshot({ path: path.join(here, "shots", s.name + ".png"), omitBackground: true });
      await page.close();
      console.log("shot", s.name);
    }
  } finally {
    await browser.close();
    fs.rmSync(path.join(here, ".tmp-shot.html"), { force: true });
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) await shoot(process.argv.slice(2).length ? process.argv.slice(2) : null);
