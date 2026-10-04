# Running Bend's `!` calls on the GPU without sudo

Bend honors the `CUDA_HOME` environment variable (default `/usr/local/cuda`) and needs only
`include/cuda.h`, `include/nvrtc.h` and `lib64/libnvrtc.so` from it; the driver (`libcuda`) comes from the
system. NVIDIA publishes redistributable archives that need no installer or sudo:

```bash
J=https://developer.download.nvidia.com/compute/cuda/redist
mkdir -p ~/.local/cuda12 /tmp/cudadl && cd /tmp/cudadl
curl -sSfO $J/cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.9.79-archive.tar.xz
curl -sSfO $J/cuda_nvrtc/linux-x86_64/cuda_nvrtc-linux-x86_64-12.9.86-archive.tar.xz
for f in *.tar.xz; do tar -xf $f -C ~/.local/cuda12 --strip-components=1; done
ln -sfn lib ~/.local/cuda12/lib64
```

Then, in each shell that builds or runs a GPU program:

```bash
export CUDA_HOME=$HOME/.local/cuda12 LD_LIBRARY_PATH=$HOME/.local/cuda12/lib
bend prog.bend -o prog     # also writes prog.gpu, which must stay next to the binary
./prog                     # `f!(x)` calls run on the GPU; `./prog --gpu off` forces the CPU
```

Checked on 2026-10-04 with driver 610.57.04 and an RTX 4050 Laptop (6 GB): the guide's `pow2!(26n)` builds, runs on the GPU and returns the right value.
