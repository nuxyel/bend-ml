// read_f32: n little-endian F32 numbers at a byte offset of a file, written by pread straight into
// an Array<F32>. The compiled program keeps an Array<F32> as one block of 32-bit cells (a BUF
// block of class c holds 2^c cells, cell i at byte 4i), so on a little-endian host the file's
// bytes are the cells' bits. The block does not move while the helper thread writes it, and the
// array is affine, so no one else reads it until the effect answers.

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "read_f32 writes the file's bytes as they are: it needs a little-endian host"
#endif

// w->size holds the bytes to read until the call, then the bytes read; w->made is the offset
static void read_f32_call(IoWork* w) {
  u64 want = w->size;
  u64 got  = 0;
  while (got < want) {
    u64     step = want - got < (1ull << 30) ? want - got : (1ull << 30);
    ssize_t n    = pread((int)w->hand, w->data + got, step,
      (off_t)w->made + (off_t)got);
    if (n <= 0) {
      io_sys_end(w, n);
      break;
    }
    got += (u64)n;
  }
  w->size = got;
}

// w->data points into the array's block, so there is nothing to free
static Term read_f32_pack(Env e, IoWork* w) {
  Term r = io_res(e, w, (Term)(u32)w->size);
  return io_tup(e, io_hand(w->hand), io_tup(e, w->item, r));
}

Term read_f32_run(Env e, Term* f, IoWork* w) {
  Term a = f[3];
  w->hand = (intptr_t)io_hand_v(f[0]);
  w->item = a;
  w->code = 0;
  if (term_tag(a) != TAG_BUF) {
    Term r = io_fail(e, EINVAL, NULL);
    return io_tup(e, io_hand(w->hand), io_tup(e, a, r));
  }
  u64 cells = 1ull << blk_cls(a);
  u64 n     = (u64)(u32)f[2];
  w->made = (intptr_t)(u32)f[1];
  w->size = (n < cells ? n : cells) * 4;
  w->data = (char*)blk_ptr(e.mem, blk_loc(e.mem, a), 0);
  return io_work(w, read_f32_call, read_f32_pack);
}

static void __attribute__((constructor)) read_f32_use(void) {
  io_eff(CID(read_f32), read_f32_run);
}
