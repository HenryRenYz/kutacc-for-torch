# Third-party notices for the bundled shared libraries

The wheel redistributes the following shared libraries inside
`kutacc_for_torch/lib`:

| Library | Source | License text |
|---|---|---|
| `libkutacc.so*` | kutacc | KUTACC-LICENSE (modified MIT, Kunpeng-only) |
| `libkupl.so*` | kupl | KUPL-LICENSE (Mulan PSL v2) |
| `libomp.so` | LLVM OpenMP runtime (via BiSheng compiler) | APACHE-2.0 (with LLVM exceptions) |
| `libc++.so.1`, `libc++abi.so.1` | LLVM libc++ (via BiSheng compiler) | APACHE-2.0 (with LLVM exceptions) |
| `libunwind.so.1` | LLVM libunwind (via BiSheng compiler) | APACHE-2.0 (with LLVM exceptions) |
| `libnuma.so.1` | numactl — https://github.com/numactl/numactl | LGPL-2.1 |
| `libibverbs.so.1` | rdma-core (core verbs only), BSD-2 branch of its dual license — https://github.com/linux-rdma/rdma-core | BSD-2-Clause.txt |
| `libnl-3.so.200`, `libnl-route-3.so.200` | libnl (dependency of libibverbs) — https://github.com/thom311/libnl | LGPL-2.1 |

The LGPL-2.1 libraries are redistributed unmodified as standalone shared
objects and dynamically linked, so they can be replaced by the recipient;
their source is available at the URLs above.

RDMA provider plugins are intentionally not bundled: they are dlopen'ed by
libibverbs from the system driver directory when rdma-core is installed, and
the shared-memory collectives used by this backend do not require them.

The kutacc license restricts the use of kutacc and its derivative works to
systems with Kunpeng processors; distributing this wheel (which embeds
libkutacc) passes that restriction on to recipients — see KUTACC-LICENSE.
