// Turns the text a command printed into a styled terminal window. Shared by the README
// screenshots (shoot.mjs) and the video scene, so both look the same.
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const span = (cls, s) => `<span class="${cls}">${esc(s)}</span>`;

export function colorize(line) {
  if (/^SOME PROOFS FAIL/.test(line)) return span("c-red b", line);
  if (/^Error:/.test(line)) return span("c-red b", line);
  let m;
  if ((m = line.match(/^(- expected\s*:)(.*)$/))) return span("c-blue b", m[1]) + span("c-blue", m[2]);
  if ((m = line.match(/^(- observed\s*:)(.*)$/))) return span("c-red b", m[1]) + span("c-red", m[2]);
  if (/^Location:/.test(line)) return span("c-dim", line);
  if (/^\d+>\|/.test(line)) return span("c-amber", line);
  if (/^\s*\|\s*\^+/.test(line)) return span("c-red b", line);
  if (/^\d+ \|/.test(line)) return span("c-dim", line);
  if (/ALL PROOFS CHECK/.test(line)) return span("c-green b", line);
  if (/^TOTAL:/.test(line)) return span("c-green b", line);
  if ((m = line.match(/^(ok\s+)(.*)$/))) return span("c-green b", m[1]) + span("c-dim", m[2]);
  if ((m = line.match(/^(ERROR\s+)(.*)$/))) return span("c-red b", m[1]) + esc(m[2]);
  if (/^law /.test(line)) return span("c-purple b", line);
  if (/^\s+#/.test(line)) return span("c-dim", line);
  if (/^\s+for /.test(line)) return span("c-cyan", line);
  if (/^\s+\{.*\}$/.test(line)) return span("c-green", line);
  if ((m = line.match(/^(text:)(.*)$/))) return span("c-green b", m[1]) + span("b", m[2]);
  if (/^\s*id \d+/.test(line)) return span("c-dim", line);
  if (/^(weights loaded|prompt ids)/.test(line)) return span("c-blue", line);
  if (/^epoch /.test(line)) return span("c-green", line);
  return esc(line);
}

export function terminalHTML({ title, cmd, text, highlight = [] }) {
  const lines = text.replace(/\n+$/, "").split("\n").map((l, i) => {
    const html = colorize(l);
    return highlight.includes(i) ? `<span class="hl">${html}</span>` : html;
  });
  const prompt = cmd ? `<span class="prompt">$</span> <span class="cmd">${esc(cmd)}</span>\n` : "";
  return `<div class="term"><div class="bar"><span class="dot" style="background:#ff5f56"></span><span class="dot" style="background:#ffbd2e"></span><span class="dot" style="background:#27c93f"></span><span class="title">${esc(title || "bend-ml")}</span></div><div class="body">${prompt}${lines.join("\n")}</div></div>`;
}
