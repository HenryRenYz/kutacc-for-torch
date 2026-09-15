# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under a modified version of the MIT license. See LICENSE in the project root for license information.

"""Build script for kutacc-for-torch.

Two dependency sources are supported, selected with KUTACC_FOR_TORCH_DEPS
(auto|vendored|system, default auto):

  system   use pre-installed kutacc/kupl prefixes found via KUTACC_HOME and
           KUPL_HOME (or KUPL_PATH), e.g. ``~/kutacc-install`` and
           ``~/kupl-install`` built with the upstream build.sh scripts.
  vendored build ``third_party/kupl`` and ``third_party/kutacc`` from source
           at the pinned commits with CMake (see scripts/bootstrap.sh to
           fetch them), installing into build/deps/staging.

Regardless of the source, the produced wheel is self-contained: every
non-system shared library of the closure (libkutacc, libkupl, compiler
runtimes such as libomp/libc++/libc++abi/libunwind, libnuma, libibverbs) is
copied into ``kutacc_for_torch/lib`` and RPATHs are rewritten to ``$ORIGIN``,
so after ``pip install`` no environment variables or LD_LIBRARY_PATH are
needed.

Build (vendored, recommended):

    python -m pip install -v --no-build-isolation .            # auto mode
    KUTACC_FOR_TORCH_DEPS=vendored python -m build --wheel ... # forced

Set KUTACC_FOR_TORCH_BUNDLE=0 to keep the historical behaviour (link against
KUTACC_HOME/KUPL_HOME and embed their absolute RPATHs, no bundled libs).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import find_packages, setup

try:
    from torch.utils.cpp_extension import BuildExtension, CppExtension
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "error: torch must be importable while building kutacc-for-torch; "
        "install torch first and build with --no-build-isolation"
    ) from exc

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "tools"))
from bundle_libs import bundle  # noqa: E402  (build-time helper, stdlib only)

VERSION = "0.2.0"

DEP_MODE = os.environ.get("KUTACC_FOR_TORCH_DEPS", "auto").strip().lower()
BUNDLE = os.environ.get("KUTACC_FOR_TORCH_BUNDLE", "1") != "0"
CLEAN_DEPS = os.environ.get("KUTACC_FOR_TORCH_DEPS_CLEAN", "0") == "1"

KUTACC_HOME = Path(os.environ.get("KUTACC_HOME", "/usr/local"))
KUPL_HOME = Path(os.environ.get("KUPL_HOME", os.environ.get("KUPL_PATH", "/usr/local")))
# Optional BiSheng installation: provides clang for the vendored builds and
# the libomp/libc++/libunwind runtimes to bundle.  Set BISHENG_HOME to use it;
# nothing here is tied to a specific machine.
BISHENG_HOME = (
    Path(os.environ["BISHENG_HOME"]) if os.environ.get("BISHENG_HOME") else None
)

THIRD_PARTY = ROOT / "third_party"
STAGING = ROOT / "build" / "deps" / "staging"
VENDORED_KUPL = THIRD_PARTY / "kupl"
VENDORED_KUTACC = THIRD_PARTY / "kutacc"
VENDORED_KUPL_PREFIX = STAGING / "kupl"
VENDORED_KUTACC_PREFIX = STAGING / "kutacc"


def _valid_prefix(prefix: Path, libname: str, header: str) -> bool:
    return (
        prefix.joinpath("include", header).is_file()
        and any(prefix.joinpath("lib").glob(f"{libname}.so*"))
    )


def _vendored_sources_present() -> bool:
    return (
        VENDORED_KUPL.joinpath("CMakeLists.txt").is_file()
        and VENDORED_KUTACC.joinpath("CMakeLists.txt").is_file()
    )


system_ok = _valid_prefix(KUTACC_HOME, "libkutacc", "kutacc.h") and _valid_prefix(
    KUPL_HOME, "libkupl", "kupl.h"
)
vendored_ok = _vendored_sources_present()

if DEP_MODE not in ("auto", "vendored", "system"):
    sys.exit(f"error: invalid KUTACC_FOR_TORCH_DEPS={DEP_MODE!r} (auto|vendored|system)")
if DEP_MODE == "system" and not system_ok:
    sys.exit(
        f"error: KUTACC_FOR_TORCH_DEPS=system but KUTACC_HOME={KUTACC_HOME} / "
        f"KUPL_HOME={KUPL_HOME} do not contain the expected lib/include layout"
    )
if DEP_MODE == "auto":
    DEP_MODE = "system" if system_ok else "vendored"
if DEP_MODE == "vendored" and not vendored_ok:
    sys.exit(
        "error: vendored dependencies are missing; run scripts/bootstrap.sh "
        "(or set KUTACC_HOME/KUPL_HOME to pre-installed prefixes)"
    )

# Directories the extension compiles and links against.
if DEP_MODE == "vendored":
    kutacc_prefix, kupl_prefix = VENDORED_KUTACC_PREFIX, VENDORED_KUPL_PREFIX
else:
    kutacc_prefix, kupl_prefix = KUTACC_HOME, KUPL_HOME


def _dep_compilers():
    """Compilers for the vendored CMake builds (kutacc requires clang).

    Taken from CC/CXX as usual for autotools/cmake-style builds; BISHENG_HOME
    is honored as a convenience default for BiSheng-based environments.
    """
    cc, cxx = os.environ.get("CC"), os.environ.get("CXX")
    if cc and cxx:
        return cc, cxx
    if BISHENG_HOME is not None and BISHENG_HOME.joinpath("bin", "clang++").exists():
        return str(BISHENG_HOME / "bin" / "clang"), str(BISHENG_HOME / "bin" / "clang++")
    sys.exit(
        "error: vendored builds need a clang compiler; set CC/CXX (or "
        "BISHENG_HOME pointing at a BiSheng compiler installation)"
    )


def _cmake(source: Path, build: Path, install_prefix: Path, cc: str, cxx: str, extra, env=None):
    build.mkdir(parents=True, exist_ok=True)
    configure = [
        "cmake",
        "-G", "Unix Makefiles",
        "-S", str(source),
        "-B", str(build),
        "-DCMAKE_C_COMPILER=" + cc,
        "-DCMAKE_CXX_COMPILER=" + cxx,
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
        *extra,
    ]
    subprocess.run(configure, check=True, env=env)
    jobs = min(os.cpu_count() or 4, 64)
    subprocess.run(
        ["cmake", "--build", str(build), "--target", "install", "--", f"-j{jobs}"],
        check=True,
        env=env,
    )


def build_vendored_deps():
    if CLEAN_DEPS and STAGING.exists():
        shutil.rmtree(STAGING)
    kupl_done = any(VENDORED_KUPL_PREFIX.joinpath("lib").glob("libkupl.so*"))
    kutacc_done = any(VENDORED_KUTACC_PREFIX.joinpath("lib").glob("libkutacc.so*"))
    if kupl_done and kutacc_done:
        print(f"vendored deps already staged in {STAGING} (set KUTACC_FOR_TORCH_DEPS_CLEAN=1 to rebuild)")
        return

    cc, cxx = _dep_compilers()

    print(f"building vendored kupl with {cxx}")
    _cmake(
        VENDORED_KUPL,
        ROOT / "build" / "deps" / "kupl-build",
        VENDORED_KUPL_PREFIX,
        cc,
        cxx,
        [
            "-DKUPL_BUILD_KIND=src",
            "-DENABLE_KUPL_PROFILE=OFF",
            "-DENABLE_KUPL_TRACE=OFF",
            "-DENABLE_KUPL_MMA=ON",
        ],
    )
    print(f"building vendored kutacc with {cxx}")
    env = dict(os.environ)
    env["CPATH"] = str(VENDORED_KUPL_PREFIX / "include") + (
        ":" + env["CPATH"] if env.get("CPATH") else ""
    )
    env["LIBRARY_PATH"] = str(VENDORED_KUPL_PREFIX / "lib") + (
        ":" + env["LIBRARY_PATH"] if env.get("LIBRARY_PATH") else ""
    )
    _cmake(
        VENDORED_KUTACC,
        ROOT / "build" / "deps" / "kutacc-build",
        VENDORED_KUTACC_PREFIX,
        cc,
        cxx,
        [
            "-DKUTACC_BUILD_KIND=src",
            "-DKUTACC_PARALLEL_BACKEND=kupl",
            "-DKUTACC_LINK_RDMA_PROVIDERS=OFF",
            f"-DCMAKE_LIBRARY_PATH={VENDORED_KUPL_PREFIX / 'lib'}",
            f"-DCMAKE_SHARED_LINKER_FLAGS=-L{VENDORED_KUPL_PREFIX / 'lib'}",
        ],
        env=env,
    )


# _cmake forwards env for kutacc so its sources find the staged kupl headers/libs.


# Keep the plain _C.so name so C++ applications can link the extension
# directly, then extend it with the dependency staging and bundling steps.
_BuildExtensionBase = BuildExtension.with_options(no_python_abi_suffix=True)


class BuildExtensionWithBundle(_BuildExtensionBase):
    def run(self):
        if DEP_MODE == "vendored":
            build_vendored_deps()
        super().run()
        if not BUNDLE:
            return
        ext_path = Path(self.get_ext_fullpath("kutacc_for_torch._C"))
        print(f"bundling shared libraries for {ext_path}")
        search_dirs = [
            kupl_prefix / "lib",
            kutacc_prefix / "lib",
        ]
        if BISHENG_HOME is not None and BISHENG_HOME.joinpath("lib").is_dir():
            search_dirs.append(BISHENG_HOME / "lib")
        search_dirs += [Path("/usr/lib64"), Path("/usr/lib"), Path("/usr/lib/aarch64-linux-gnu")]
        bundle(
            ext_path,
            search_dirs=search_dirs,
            header_dirs=[kutacc_prefix / "include", kupl_prefix / "include"],
        )


def _package_version() -> str:
    tag = os.environ.get("KUTACC_FOR_TORCH_VERSION_TAG")
    if tag is None:
        import torch

        tag = "torch" + torch.__version__.replace("+", ".")
    return f"{VERSION}+{tag}" if tag else VERSION


library_dirs = [str(kutacc_prefix / "lib"), str(kupl_prefix / "lib")]
libraries = ["kutacc", "kupl", "numa", "pthread"]
if BISHENG_HOME is not None and BISHENG_HOME.joinpath("lib", "libomp.so").exists():
    library_dirs.append(str(BISHENG_HOME / "lib"))
    libraries.append("omp")

if BUNDLE:
    extra_link_args = []  # RPATH is rewritten to $ORIGIN/lib after bundling
else:
    extra_link_args = [f"-Wl,-rpath,{path}" for path in library_dirs]

setup(
    name="kutacc-for-torch",
    version=_package_version(),
    description="PyTorch distributed backend powered by Kutacc shared-memory collectives",
    packages=find_packages(),
    package_data={
        "kutacc_for_torch": [
            "include/kutacc_for_torch/*.hpp",
            "include/*.h",
            "lib/*.so*",
            ".licenses/*",
        ]
    },
    ext_modules=[
        CppExtension(
            name="kutacc_for_torch._C",
            sources=[str(ROOT / "csrc" / "process_group_kutacc.cpp")],
            include_dirs=[
                str(ROOT / "kutacc_for_torch" / "include"),
                str(kutacc_prefix / "include"),
                str(kupl_prefix / "include"),
            ],
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=["-O3", "-std=c++20", "-Wall", "-Wextra"],
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtensionWithBundle},
    python_requires=">=3.9",
)
