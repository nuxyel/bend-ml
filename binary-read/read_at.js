// read_at: a file's size as 64 bits, and little-endian F32 or BF16 numbers at a 64-bit byte offset, into an
// Array<F32>. A size or an offset crosses as two U32s, hi and lo: hi * 2^32 + lo. The JS lane keeps an Array<F32> as a
// JS array of numbers, so the bytes go through a buffer read in chunks and are viewed as float32 before they are
// stored, as the C lane's cells hold them. JS numbers hold every integer below 2^53 exactly, which covers the offsets.

// the size as (hi, lo); fstat answers it as a number, exact below 2^53
function size64(file) {
  try {
    const size = require("fs").fstatSync(file).size;
    return io_tup(file, io_done(io_tup(Math.floor(size / 4294967296), size % 4294967296)));
  } catch (e) {
    return io_tup(file, io_fail(Math.abs(e.errno ?? 5)));
  }
}

// reads up to `len` bytes at byte `off` in chunks of up to 16 MiB, handing each chunk to `put(b, at, fill)` (`at` its
// first byte's place in the read, a multiple of the chunk); answers the bytes read, or a negative errno
function read_at_chunks(file, off, len, put) {
  const sys = io_sys();
  const b = new Uint8Array(Math.max(Math.min(len, 1 << 24), 4));
  let got = 0;
  while (got < len) {
    let fill = 0;
    const step = Math.min(len - got, b.length);
    while (fill < step) {
      const k = Number(sys.pread(file, sys.ptr(b, fill), step - fill, BigInt(off + got + fill)));
      if (k < 0) {
        return -sys.errno();
      }
      if (k === 0) {
        break;
      }
      fill += k;
    }
    put(b, got, fill);
    got += fill;
    if (fill < step) {
      break;
    }
  }
  return got;
}

// the _into effects write from cell `dst` on (the _at effects from cell 0)
function read_f32_into(file, hi, lo, n, a, dst) {
  const off = (hi >>> 0) * 4294967296 + (lo >>> 0);
  const len = Math.min(n >>> 0, Math.max(a.length - (dst >>> 0), 0)) * 4;
  const got = read_at_chunks(file, off, len, (b, at, fill) => {
    const v = new DataView(b.buffer);
    const j0 = (dst >>> 0) + at / 4;
    for (let i = 0; i + 4 <= fill; i += 4) {
      a[j0 + i / 4] = v.getFloat32(i, true);
    }
    // a last partial number writes only its bytes over the cell's bits, as read_f32 does
    const part = fill & 3;
    if (part > 0) {
      const t = new DataView(new ArrayBuffer(4));
      const j = j0 + (fill >> 2);
      t.setFloat32(0, a[j], true);
      for (let i = 0; i < part; i++) {
        t.setUint8(i, b[fill - part + i]);
      }
      a[j] = t.getFloat32(0, true);
    }
  });
  return io_tup(file, io_tup(a, got < 0 ? io_fail(-got) : io_done(Math.floor(got / 4))));
}

// each whole BF16 number b becomes the float32 with bits b << 16; a last odd byte is dropped
function read_bf16_into(file, hi, lo, n, a, dst) {
  const off = (hi >>> 0) * 4294967296 + (lo >>> 0);
  const len = Math.min(n >>> 0, Math.max(a.length - (dst >>> 0), 0)) * 2;
  const t = new DataView(new ArrayBuffer(4));
  const got = read_at_chunks(file, off, len, (b, at, fill) => {
    const j0 = (dst >>> 0) + at / 2;
    for (let i = 0; i + 2 <= fill; i += 2) {
      t.setUint32(0, (b[i] | (b[i + 1] << 8)) << 16 >>> 0);
      a[j0 + i / 2] = t.getFloat32(0);
    }
  });
  return io_tup(file, io_tup(a, got < 0 ? io_fail(-got) : io_done(Math.floor(got / 2))));
}

function read_f32_at(file, hi, lo, n, a) {
  return read_f32_into(file, hi, lo, n, a, 0);
}

function read_bf16_at(file, hi, lo, n, a) {
  return read_bf16_into(file, hi, lo, n, a, 0);
}

io_eff(CID(size64), size64);
io_eff(CID(read_f32_at), read_f32_at);
io_eff(CID(read_bf16_at), read_bf16_at);
io_eff(CID(read_f32_into), read_f32_into);
io_eff(CID(read_bf16_into), read_bf16_into);
