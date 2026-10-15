// The same work as read_at_bytes.bend in C: pread 1 MiB blocks of w.bin and add up the bytes.
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
int main(void) {
  struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
  int fd = open("w.bin", O_RDONLY); unsigned char *buf = malloc(1 << 20); uint32_t s = 0;
  for (off_t off = 0; off < 384L << 20; off += 1 << 20) {
    ssize_t n = pread(fd, buf, 1 << 20, off);
    for (ssize_t i = 0; i < n; i++) s += buf[i];
  }
  close(fd); clock_gettime(CLOCK_MONOTONIC, &b);
  printf("sum %u  ms %ld\n", s, (b.tv_sec - a.tv_sec) * 1000 + (b.tv_nsec - a.tv_nsec) / 1000000);
}
