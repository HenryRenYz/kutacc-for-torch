# kutacc-for-torch

`kutacc-for-torch` registers a CPU `torch.distributed` backend named `kutacc`.
It maps PyTorch operations to the generic, asynchronous Kutacc shared-memory
collectives. C++ users can include
`kutacc_for_torch/process_group_kutacc.hpp` and call
`kutacc_torch::create_process_group_kutacc` to obtain a `c10d::Backend`.
The installed `_C.so` exports this factory and the C++ AllGatherV extension;
it can be linked by C++ applications together with LibTorch and libpython.

Supported operations:

- `broadcast`
- `all_gather` (list, including rank-ragged inputs)
- `all_gather_into_tensor`
- `all_to_all` (list, including pair-ragged tensors)
- `all_to_all_single` (equal or explicit split sizes)
- `barrier`
- `kutacc_for_torch.all_gatherv_into_tensor` as a ragged single-output extension

This maps every generic copy collective added on Kutacc's `feat/more-comms`
branch. The older BF16/topology-specific AllReduce and ReduceScatter APIs are
not part of this backend version.

All operations return a native asynchronous `Work` when PyTorch is called with
`async_op=True`. Tensor metadata is copied during submission and tensor objects
are retained by `Work` until completion.

## Tensor support

The backend is CPU-only and intentionally does not maintain a dtype whitelist.
It passes `Tensor.element_size()` to Kutacc, which transports the logical tensor
as bytes. This covers bool, integer, floating-point, bfloat16, complex, Float8,
and future strided CPU dtypes as long as communicating tensors have compatible
element widths and logical shapes.

Contiguous, transposed, channels-last, and positive-stride sliced tensors are
packed and restored according to their logical element order. Negative strides,
expanded/zero-stride dimensions, sparse layouts, and overlapping output views
are rejected.

## Build

The build integrates kutacc and kupl: either build them from the pinned
submodules in `third_party/` (vendored, default when no pre-installed
prefixes are found), or link against existing installs discovered through
`KUTACC_HOME`/`KUPL_HOME` (system mode). Both modes produce a self-contained
wheel: the required shared libraries of the closure (libkutacc, libkupl, the
compiler runtimes such as libomp/libc++/libc++abi/libunwind, libnuma,
libibverbs) are bundled under `kutacc_for_torch/lib` with `$ORIGIN` RPATHs,
so installing the wheel needs no environment variables — like torch or numpy.

Toolchain and dependency locations are passed through the standard
environment variables; nothing is hard-coded to a particular machine:

```bash
pip install patchelf            # once; used to rewrite RPATHs while bundling
./scripts/bootstrap.sh          # once; fetches third_party/{kupl,kutacc}

# compilers for the vendored CMake builds (kutacc requires clang)
export CC=$(which clang)
export CXX=$(which clang++)

# optional: BiSheng-based toolchains provide clang and the libomp/libc++
# runtimes that get bundled into the wheel
# export BISHENG_HOME=/path/to/BiShengCompiler-...

python -m pip install -v --no-build-isolation .
```

Dependency-source selection (`KUTACC_FOR_TORCH_DEPS=auto|vendored|system`,
default `auto` — system when `KUTACC_HOME`/`KUPL_HOME` point at valid
prefixes, vendored otherwise):

```bash
# force building kutacc/kupl from the pinned third_party/ sources
KUTACC_FOR_TORCH_DEPS=vendored python -m pip install -v --no-build-isolation .

# or use existing install prefixes
KUTACC_FOR_TORCH_DEPS=system \
KUTACC_HOME=/path/to/kutacc-install KUPL_HOME=/path/to/kupl-install \
    python -m pip install -v --no-build-isolation .
```

Vendored builds compile `third_party/kupl` and `third_party/kutacc` at the
pinned commits; results are staged in `build/deps/staging`. Set
`KUTACC_FOR_TORCH_DEPS_CLEAN=1` to force a full rebuild. To keep the
historical non-bundled behaviour (absolute RPATHs into the install prefixes,
no bundled libraries) set `KUTACC_FOR_TORCH_BUNDLE=0`.

The wheel version embeds the torch ABI it was compiled against (for example
`0.2.0+torch2.14.0.cpu`), mirroring how torch ships `+cpu`/`+cu124`
variants; build one wheel per torch build.

If `bootstrap.sh` cannot fetch a submodule (for example in a network-restricted
environment), point it at any reachable mirror:

```bash
KUTACC_FOR_TORCH_KUTACC_MIRROR=/path/to/a/kutacc/clone ./scripts/bootstrap.sh
```

Importing `kutacc_for_torch` registers the backend:

```python
import kutacc_for_torch
import torch.distributed as dist

dist.init_process_group("kutacc")
```

Before `init_process_group`, `torch.set_num_threads(n)` controls the requested
Kutacc copy-worker count. To preserve the tuned 920F defaults, it is capped at
eight unless `KUTACC_TORCH_MAX_COPY_THREADS` is set. Existing
`KUTACC_SHM_COPY_THREADS` and `KUTACC_SHM_COPY_POOL_THREADS` variables take
precedence.

The current implementation is built and tested against the af3 environment's
PyTorch `2.7.1a0+gite2d141d` C++ ABI.

The NUMA-bound distributed correctness suite can be launched with OpenMPI:

```bash
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29641
mpirun --bind-to none -x MASTER_ADDR -x MASTER_PORT -np 4 \
  tests/numa_rank_wrapper.sh python tests/test_distributed.py
```
