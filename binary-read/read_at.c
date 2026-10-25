// read_at: a file's size as 64 bits, and little-endian F32 or BF16 numbers at a 64-bit byte offset, into an
// Array<F32>. Bend (2.0.39) has no 64-bit integer, so a size or an offset crosses as two U32s, hi and lo: hi * 2^32 + lo.
// The compiled program keeps an Array<F32> as one block of 32-bit cells (a BUF block of class c holds 2^c cells, cell i
// at byte 4i), so on a little-endian host an F32 file's bytes are the cells' bits, and a BF16 number b becomes the
// cell b << 16. The block does not move while the helper thread writes it, and the array is affine, so no one else
// reads it until the effect answers. Only the effects a program uses are compiled (the #if lines), as in Base.

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "read_at writes the file's bytes as they are: it needs a little-endian host"
#endif

#include <sys/stat.h>

#ifdef CID(size64)

// the size goes in w->size; the host's off_t is 64 bits, so no file is too large
static void size64_call(IoWork* w) {
  struct stat st;
  int n = fstat((int)w->hand, &st);
  io_sys_end(w, n);
  w->size = n == 0 ? (u64)st.st_size : 0;
}

static Term size64_pack(Env e, IoWork* w) {
  Term hi = (Term)(u32)(w->size >> 32);
  Term lo = (Term)(u32)w->size;
  return io_tup(e, io_hand(w->hand), io_res(e, w, io_tup(e, hi, lo)));
}

Term size64_run(Env e, Term* f, IoWork* w) {
  w->hand = (intptr_t)io_hand_v(f[0]);
  return io_work(w, size64_call, size64_pack);
}

static void __attribute__((constructor)) size64_use(void) {
  io_eff(CID(size64), size64_run);
}

#endif

#if defined(CID(read_f32_at)) || defined(CID(read_bf16_at)) || defined(CID(read_f32_into)) \
  || defined(CID(read_bf16_into))

// the array goes back with the number of whole numbers read (w->size, set by the call), or the error
static Term read_at_pack(Env e, IoWork* w) {
  Term r = io_res(e, w, (Term)(u32)w->size);
  return io_tup(e, io_hand(w->hand), io_tup(e, w->item, r));
}

// f = file, hi, lo, n, array, then the first cell to write (`at`, 0 for the _at effects); `width` bytes a number.
// w->made is the offset, w->size the bytes to read (n clipped to the cells from `at` to the array's end), w->data the
// cell `at`.
static Term read_at_start(Env e, Term* f, IoWork* w, u64 width, u64 at, IoCall call) {
  Term a = f[4];
  w->hand = (intptr_t)io_hand_v(f[0]);
  w->item = a;
  w->code = 0;
  if (term_tag(a) != TAG_BUF) {
    Term r = io_fail(e, EINVAL, NULL);
    return io_tup(e, io_hand(w->hand), io_tup(e, a, r));
  }
  u64 cells = 1ull << blk_cls(a);
  cells     = at < cells ? cells - at : 0;
  u64 n     = (u64)(u32)f[3];
  w->made = (intptr_t)(((u64)(u32)f[1] << 32) | (u64)(u32)f[2]);
  w->size = (n < cells ? n : cells) * width;
  w->data = (char*)blk_ptr(e.mem, blk_loc(e.mem, a), 0) + 4 * at;  // read only when a cell from `at` exists
  return io_work(w, call, read_at_pack);
}

// pread until `want` bytes are at `to` or the file ends (a short count) or fails (w->code); answers the bytes read
static u64 read_at_fill(IoWork* w, char* to, u64 want, u64 at) {
  u64 got = 0;
  while (got < want) {
    u64     step = want - got < (1ull << 30) ? want - got : (1ull << 30);
    ssize_t n    = pread((int)w->hand, to + got, step, (off_t)(at + got));
    if (n <= 0) {
      io_sys_end(w, n);
      break;
    }
    got += (u64)n;
  }
  return got;
}

#endif

#if defined(CID(read_f32_at)) || defined(CID(read_f32_into))

// the bytes go straight into the cells, as read_f32's do
static void read_f32_at_call(IoWork* w) {
  w->size = read_at_fill(w, w->data, w->size, (u64)w->made) / 4;
}

#endif

#ifdef CID(read_f32_at)

Term read_f32_at_run(Env e, Term* f, IoWork* w) {
  return read_at_start(e, f, w, 4, 0, read_f32_at_call);
}

static void __attribute__((constructor)) read_f32_at_use(void) {
  io_eff(CID(read_f32_at), read_f32_at_run);
}

#endif

#ifdef CID(read_f32_into)

Term read_f32_into_run(Env e, Term* f, IoWork* w) {
  return read_at_start(e, f, w, 4, (u64)(u32)f[5], read_f32_at_call);
}

static void __attribute__((constructor)) read_f32_into_use(void) {
  io_eff(CID(read_f32_into), read_f32_into_run);
}

#endif

#if defined(CID(read_bf16_at)) || defined(CID(read_bf16_into))

// the bytes go through a buffer of up to 16 MiB, and each whole BF16 number b becomes the cell b << 16 (exact: the
// F32 with b's sign, exponent and top mantissa bits, and zeros below). Only the last block can end on an odd byte,
// and got / 2 drops that byte.
static void read_bf16_at_call(IoWork* w) {
  u64       want = w->size;
  u64       got  = 0;
  u64       cap  = want < (1ull << 24) ? want : (1ull << 24);
  uint8_t*  b    = malloc(cap > 0 ? cap : 1);
  uint32_t* c    = (uint32_t*)w->data;
  if (b == NULL) {
    w->code = ENOMEM;
    w->size = 0;
    return;
  }
  while (got < want) {
    u64 step = want - got < cap ? want - got : cap;
    u64 fill = read_at_fill(w, (char*)b, step, (u64)w->made + got);
    u64 at   = got / 2;
    for (u64 i = 0; i + 2 <= fill; i += 2) {
      c[at + i / 2] = ((uint32_t)b[i] | (uint32_t)b[i + 1] << 8) << 16;
    }
    got += fill;
    if (fill < step) {
      break;
    }
  }
  free(b);
  w->size = got / 2;
}

#endif

#ifdef CID(read_bf16_at)

Term read_bf16_at_run(Env e, Term* f, IoWork* w) {
  return read_at_start(e, f, w, 2, 0, read_bf16_at_call);
}

static void __attribute__((constructor)) read_bf16_at_use(void) {
  io_eff(CID(read_bf16_at), read_bf16_at_run);
}

#endif

#ifdef CID(read_bf16_into)

Term read_bf16_into_run(Env e, Term* f, IoWork* w) {
  return read_at_start(e, f, w, 2, (u64)(u32)f[5], read_bf16_at_call);
}

static void __attribute__((constructor)) read_bf16_into_use(void) {
  io_eff(CID(read_bf16_into), read_bf16_into_run);
}

#endif
