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

## Build on 920f-4

```bash
source ~/.bashrc
conda activate af3
export CC=/home/ryz/BiShengCompiler-5.1.0.2-aarch64-linux/bin/clang
export CXX=/home/ryz/BiShengCompiler-5.1.0.2-aarch64-linux/bin/clang++
export KUTACC_HOME=/home/ryz/autosync/comm-test/kutacc-install-more
export KUPL_HOME=/home/ryz/autosync/comm-test/kupl-install
export BISHENG_HOME=/home/ryz/BiShengCompiler-5.1.0.2-aarch64-linux
python -m pip install -v --no-build-isolation .
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
