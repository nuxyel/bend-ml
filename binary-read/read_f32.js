// read_f32: n little-endian F32 numbers at a byte offset of a file, into an Array<F32>. The JS
// lane keeps an Array<F32> as a JS array of numbers, so the bytes go through a buffer read in
// chunks and are viewed as float32 before they are stored, as the C lane's cells hold them. A last
// partial number writes only its bytes over the cell's bits, as the C lane's pread does.

function read_f32(file, offset, n, a) {
  const sys = io_sys();
  const len = Math.min(n, a.length) * 4;
  const b = new Uint8Array(Math.max(Math.min(len, 1 << 24), 4));
  const v = new DataView(b.buffer);
  let got = 0;
  while (got < len) {
    let fill = 0;
    const step = Math.min(len - got, b.length);
    while (fill < step) {
      const k = Number(sys.pread(file, sys.ptr(b, fill), step - fill,
        BigInt(offset + got + fill)));
      if (k < 0) {
        return io_tup(file, io_tup(a, io_fail(sys.errno())));
      }
      if (k === 0) {
        break;
      }
      fill += k;
    }
    const at = got >> 2;
    for (let i = 0; i + 4 <= fill; i += 4) {
      a[at + (i >> 2)] = v.getFloat32(i, true);
    }
    const part = fill & 3;
    if (part > 0) {
      const t = new DataView(new ArrayBuffer(4));
      const j = at + (fill >> 2);
      t.setFloat32(0, a[j], true);
      for (let i = 0; i < part; i++) {
        t.setUint8(i, b[fill - part + i]);
      }
      a[j] = t.getFloat32(0, true);
    }
    got += fill;
    if (fill < step) {
      break;
    }
  }
  return io_tup(file, io_tup(a, io_done(got)));
}

io_eff(CID(read_f32), read_f32);
