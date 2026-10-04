// Turns the text a command printed into a Bend-style code block (flat panel, muted label,
// Bend's syntax colours). Shared by the README figures (shoot.mjs) and the video.
export const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const span = (cls, s) => `<span class="${cls}">${esc(s)}</span>`;

export function colorize(line) {
  let m;
  if (/^SOME PROOFS FAIL/.test(line)) return span("r b", line);
  if (/^Error:/.test(line)) return span("r", line);
  if ((m = line.match(/^(- expected\s*:)(.*)$/))) return span("c", m[1]) + span("i b", m[2]);
  if ((m = line.match(/^(- observed\s*:)(.*)$/))) return span("c", m[1]) + span("r b", m[2]);
  if (/^Location:/.test(line)) return span("c", line);
  if (/^\d+>\|/.test(line)) return span("i", line);
  if (/^\s*\|\s*\^+/.test(line)) return span("r", line);
  if (/^\d+ \|/.test(line)) return span("c", line);
  if (/ALL PROOFS CHECK/.test(line)) return span("g b", line);
  if (/^TOTAL:/.test(line)) return span("g b", line);
  if ((m = line.match(/^(ok\s+)(.*)$/))) return span("g", m[1]) + span("c", m[2]);
  if ((m = line.match(/^(ERROR\s+)(.*)$/))) return span("r b", m[1]) + esc(m[2]);
  if ((m = line.match(/^(law )(.*)$/))) return span("k", m[1]) + span("i b", m[2]);
  if (/^\s+#/.test(line)) return span("c", line);
  if ((m = line.match(/^(\s+)(for )(.*)$/))) return esc(m[1]) + span("k", m[2]) + esc(m[3]);
  if (/^\s+\{.*\}$/.test(line)) return span("v", line);
  if ((m = line.match(/^(text:)(.*)$/))) return span("c", m[1]) + span("i b", m[2]);
  if (/^\s*id \d+/.test(line)) return span("c", line);
  if (/^(weights loaded|prompt ids)/.test(line)) return span("c", line);
  if (/^epoch /.test(line)) return span("i", line);
  return esc(line);
}

// label: the command, shown muted above the block; lines: the output
export function blockHTML({ cmd, text, mark = [] }) {
  const body = text.replace(/\n+$/, "").split("\n").map((l, i) => (mark.includes(i) ? `<span class="mark">${colorize(l)}</span>` : colorize(l))).join("\n");
  const lbl = cmd ? `<div class="lbl"><span class="p">$</span> ${esc(cmd)}</div>` : "";
  return `${lbl}<div class="blk">${body}</div>`;
}
