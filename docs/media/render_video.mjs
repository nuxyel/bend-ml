// Renders video/scene.html frame by frame with Brave and encodes it with ffmpeg.
//   node render_video.mjs                  full video -> bend-ml.mp4 (not committed; attached to the release)
//   node render_video.mjs --stills 1,3,12  PNG stills of those seconds -> frames/still-<t>.png (for review)
//   node render_video.mjs --poster         poster.png (the hook, after the error)
import puppeteer from "puppeteer-core";
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
// frames are rendered at 60 fps and each pair is blended into one 30 fps frame (motion blur)
const FPS = 60, OUT_FPS = 30, W = 1920, H = 1080;
const browserPath = process.env.BRAVE || "/usr/bin/brave";
const args = process.argv.slice(2);
const mime = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".json": "application/json", ".woff2": "font/woff2", ".txt": "text/plain", ".png": "image/png", ".svg": "image/svg+xml" };

const server = http.createServer((req, res) => {
  const p = path.join(here, decodeURIComponent(req.url.split("?")[0]));
  if (!p.startsWith(here) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "content-type": mime[path.extname(p)] || "application/octet-stream" });
  fs.createReadStream(p).pipe(res);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;

const browser = await puppeteer.launch({ executablePath: browserPath, headless: "new", args: ["--no-sandbox", "--hide-scrollbars", "--force-device-scale-factor=1"] });
const page = await browser.newPage();
await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });
page.on("pageerror", (e) => console.error("page error:", e.message));
page.on("console", (m) => { if (m.type() === "error") console.error("console:", m.text()); });
await page.goto(`http://127.0.0.1:${port}/video/scene.html`);
await page.waitForFunction("window.__ready === true", { timeout: 30000 });
const DURATION = await page.evaluate("window.DURATION");
const frame = async (t, opts) => { await page.evaluate((x) => window.render(x), t); return page.screenshot(opts); };

try {
  if (args[0] === "--stills") {
    fs.mkdirSync(path.join(here, "frames"), { recursive: true });
    for (const t of args[1].split(",").map(Number)) {
      fs.writeFileSync(path.join(here, "frames", `still-${t}.png`), await frame(t, { type: "png" }));
      console.log("still", t);
    }
  } else if (args[0] === "--poster") {
    fs.writeFileSync(path.join(here, "poster.png"), await frame(6.0, { type: "png" }));
    console.log("poster.png");
  } else {
    const out = path.join(here, "bend-ml.mp4");
    const ff = spawn("ffmpeg", ["-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", String(FPS), "-c:v", "mjpeg", "-i", "-",
      "-vf", `tmix=frames=2,fps=${OUT_FPS},scale=in_range=full:out_range=tv,format=yuv420p`, "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-color_range", "tv", "-movflags", "+faststart", out], { stdio: ["pipe", "inherit", "inherit"] });
    const done = new Promise((r) => ff.on("close", r));
    const total = Math.round(DURATION * FPS);
    for (let f = 0; f < total; f++) {
      const buf = await frame(f / FPS, { type: "jpeg", quality: 95 });
      if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once("drain", r));
      if (f % 300 === 0) console.log(`frame ${f}/${total}`);
    }
    ff.stdin.end();
    await done;
    console.log("wrote", out);
  }
} finally {
  await browser.close();
  server.close();
}
