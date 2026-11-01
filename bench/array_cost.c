// The C twin of bench/array_cost.bend: the same seven operations as plain loops over one malloc'd block, with the
// same values and the same order of F32 additions, so each prints what the Bend program prints. Built as Bend builds
// its C (clang -std=c11 -O3). A compiler barrier after each rep keeps every rep's loads and stores.
//
//   ./array_cost_c <op> <depth> <reps>     op: get fold set new tolist round btolist
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KEEP(p) __asm__ volatile("" : : "r"(p) : "memory")

typedef struct Cell { float v; struct Cell* next; } Cell;

// a list built front to back in an arena (the least a cons list can cost: one store pair a cell)
static Cell* to_list(const float* a, uint64_t n, Cell* arena) {
  for (uint64_t i = 0; i < n; i++) {
    arena[i].v = a[i];
    arena[i].next = i + 1 < n ? &arena[i + 1] : NULL;
  }
  return n ? arena : NULL;
}

static float sum_list(const Cell* l, float acc) {
  for (; l; l = l->next) acc = acc + l->v;
  return acc;
}

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  const char* op = argv[1];
  uint64_t d = strtoull(argv[2], 0, 10), k = strtoull(argv[3], 0, 10), n = 1ull << d;
  if (!strcmp(op, "get") || !strcmp(op, "set")) {
    uint32_t* a = malloc(n * 4);
    for (uint64_t i = 0; i < n; i++) a[i] = (uint32_t)i + 3;
    KEEP(a);
    if (!strcmp(op, "get")) {
      uint32_t s = 0;
      for (uint64_t r = 0; r < k; r++) {
        for (uint64_t i = 0; i < n; i++) s += a[i];
        KEEP(a);
      }
      printf("%u\n", s);
    } else {
      uint32_t x = 5;
      for (uint64_t r = 0; r < k; r++, x++) {
        for (uint64_t i = 0; i < n; i++) a[i] = (uint32_t)i + x;
        KEEP(a);
      }
      printf("%u\n", a[0] + a[n - 1]);
    }
    return 0;
  }
  float* a = malloc(n * 4);
  for (uint64_t i = 0; i < n; i++) a[i] = (float)(i & 1023);
  KEEP(a);
  if (!strcmp(op, "fold")) {
    float s = 0.0f;
    for (uint64_t r = 0; r < k; r++) {
      s = s * 0.5f;
      for (uint64_t i = 0; i < n; i++) s = s + a[i];
      KEEP(a);
    }
    printf("%.9g\n", s);
  } else if (!strcmp(op, "new") || !strcmp(op, "newrt")) {
    float acc = 1.0f;
    for (uint64_t r = 0; r < k; r++) {
      for (uint64_t i = 0; i < n; i++) a[i] = acc;
      KEEP(a);
      acc = a[7] + 1.0f;
    }
    printf("%.9g\n", acc);
  } else if (!strcmp(op, "tolist") || !strcmp(op, "round")) {
    Cell* arena = malloc(n * sizeof(Cell));
    int round = !strcmp(op, "round");
    float s = 0.0f;
    Cell* l = to_list(a, n, arena);
    for (uint64_t r = 0; r < k; r++) {
      if (!round) s = sum_list(l, s * 0.5f);
      else for (Cell* c = l; c; c = c->next) a[c - arena] = c->v;
      KEEP(a);
      l = to_list(a, n, arena);
      KEEP(arena);
    }
    if (round) s = sum_list(l, 0.0f);
    printf("%.9g\n", s);
  } else if (!strcmp(op, "btolist")) {
    Cell* arena = malloc(n * sizeof(Cell));
    float acc = 1.0f;
    for (uint64_t r = 0; r < k; r++) {
      for (uint64_t i = 0; i < n; i++) a[i] = acc;
      KEEP(a);
      Cell* l = to_list(a, n, arena);
      KEEP(arena);
      acc = sum_list(l, 0.0f) * 0.000001f;
    }
    printf("%.9g\n", acc);
  } else {
    return 2;
  }
  return 0;
}
