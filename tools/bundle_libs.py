# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under a modified version of the MIT license. See LICENSE in the project root for license information.

"""Collect the shared-library closure of the built extension into the package.

The wheel must stay self-contained: after ``pip install`` no KUTACC_HOME /
KUPL_HOME / BISHENG_HOME variables, no LD_LIBRARY_PATH and no system copies of
kutacc/kupl may be required.  This module walks the DT_NEEDED graph of the
built ``_C`` extension, copies every dependency that is not platform- or
torch-provided into ``kutacc_for_torch/lib`` under its SONAME, and rewrites
RPATHs to ``$ORIGIN`` so the loader resolves siblings inside that directory.

The scan phase is strictly read-only: ``patchelf`` only ever touches the
extension build output and the copies inside ``lib/``, never the libraries it
discovered on the build machine.

rdma-core provider plugins are detected by rule, not by name list, and are
dropped from DT_NEEDED instead of bundled: they are dlopen'ed by libibverbs
at runtime, and the direct NEEDED entries (an artifact of kutacc linking them
with absolute paths) would make the wheel unloadable on machines without the
exact provider set.
"""

import shutil
import subprocess
import sys
from pathlib import Path

# Provided by the platform or by torch itself; never bundled.  These are
# ABI-stable sonames (glibc runtime, C++ runtimes already required by torch),
# not machine-specific paths.
SYSTEM_LIBS = {
    "ld-linux-aarch64.so.1",
    "ld-linux-x86-64.so.2",
    "libc.so.6",
    "libm.so.6",
    "libpthread.so.0",
    "libdl.so.2",
    "librt.so.1",
    "libresolv.so.2",
    "libgcc_s.so.1",
    "libstdc++.so.6",
    "libc10.so",
    "libtorch.so",
    "libtorch_cpu.so",
    "libtorch_python.so",
}

# Headers a C++ consumer of _C.so needs next to our own public header.
PUBLIC_HEADERS = ["kutacc.h", "kupl.h", "kupl_mma.h"]

_EXTENSION_KEY = "<extension>"


def _patchelf(binary, *args):
    result = subprocess.run(
        [binary, *args], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"patchelf {' '.join(args)} failed:\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


def find_patchelf():
    binary = shutil.which("patchelf")
    if binary is None:
        sys.exit(
            "error: patchelf is required to bundle the shared libraries "
            "(run: pip install patchelf)"
        )
    return binary


def needed_entries(binary, path):
    return [name for name in _patchelf(binary, "--print-needed", str(path)).splitlines() if name]


def soname_of(binary, path):
    return _patchelf(binary, "--print-soname", str(path)) or None


def ldconfig_cache():
    cache = {}
    result = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "=>" not in line:
            continue
        head, _, tail = line.partition("=>")
        cache[head.strip().split()[0]] = tail.strip()
    return cache


def resolve(name, search_dirs, cache):
    for directory in search_dirs:
        candidate = Path(directory) / name
        if candidate.is_file():
            return candidate
    return Path(cache[name]) if name in cache and Path(cache[name]).is_file() else None


def _dynamic_symbols(path, defined):
    """Dynamic symbol names of an ELF, versions stripped; None if unknown."""
    flag = "--defined-only" if defined else "--undefined-only"
    try:
        result = subprocess.run(
            ["nm", "-D", flag, str(path)], capture_output=True, text=True
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    symbols = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        symbols.add(parts[-1].split("@")[0])
    return symbols


def _references(dependent, candidate):
    """Whether ``dependent`` references any dynamic symbol of ``candidate``."""
    undefined = _dynamic_symbols(dependent, defined=False)
    defined = _dynamic_symbols(candidate, defined=True)
    if undefined is None or defined is None:
        return True  # cannot tell; assume it is a real dependency
    return bool(undefined & defined)


def _plugin_by_location(name, resolved):
    """Whether the entry is an rdma-core provider plugin by name or path.

    rdma-core names its plugin files lib<provider>-rdmavN.so across distros
    (rdmav2, rdmav34, ...) and installs them into a libibverbs directory;
    entries recorded without a SONAME keep the file name.
    """
    if name.startswith("lib") and "-rdmav" in name:
        return True
    if resolved is None:
        return False
    try:
        return "libibverbs" in resolved.resolve().parts
    except OSError:
        return False


def bundle(extension_path, search_dirs, header_dirs, verbose=True):
    """Bundle the closure of ``extension_path`` and rewrite its RPATH.

    ``search_dirs`` are consulted in order for every DT_NEEDED entry; prefer
    staging/vendored directories so bundled copies match what was linked.
    """
    extension = Path(extension_path)
    package_dir = extension.parent
    lib_dir = package_dir / "lib"
    patchelf = find_patchelf()

    # Providers may only be reachable through the ibverbs plugin directory,
    # and libraries commonly sit next to their own runtime dependencies.
    search_dirs = list(search_dirs)
    for directory in list(search_dirs):
        plugin_dir = Path(directory) / "libibverbs"
        if plugin_dir.is_dir() and plugin_dir not in search_dirs:
            search_dirs.append(plugin_dir)

    cache = ldconfig_cache()

    # ---- read-only scan: collect candidates over the whole closure --------
    scan_queue = [extension]
    scanned = {extension}
    # SONAME (or file name) of the library a DT_NEEDED entry belongs to.
    def _key_of(path):
        if path == extension:
            return _EXTENSION_KEY
        return soname_of(patchelf, path) or path.name

    # needed name -> {"source": Path|None, "referenced": bool,
    #                 "dependents": {soname keys}}
    candidates = {}
    while scan_queue:
        current = scan_queue.pop()
        current_key = _key_of(current)
        undefined = None
        for name in needed_entries(patchelf, current):
            if name in SYSTEM_LIBS:
                continue
            entry = candidates.setdefault(
                name, {"source": None, "referenced": False, "dependents": set()}
            )
            entry["dependents"].add(current_key)
            if entry["source"] is None:
                entry["source"] = resolve(name, search_dirs, cache)
            source = entry["source"]
            if source is None or _plugin_by_location(name, source):
                continue
            if not entry["referenced"]:
                if undefined is None:
                    undefined = _dynamic_symbols(current, defined=False)
                defined = _dynamic_symbols(source, defined=True)
                if undefined is None or defined is None:
                    entry["referenced"] = True  # cannot tell; keep it
                elif undefined & defined:
                    entry["referenced"] = True
            if source not in scanned:
                scanned.add(source)
                scan_queue.append(source)

    missing = sorted(n for n, e in candidates.items() if e["source"] is None)
    if missing:
        sys.exit(
            "error: cannot find " + ", ".join(missing)
            + " in: " + ", ".join(str(d) for d in search_dirs)
        )

    # ---- decide: bundle real dependencies, drop unreferenced ones ---------
    planned = {}   # target file name in lib/ -> source path
    drops = {}     # dependent key -> [needed names to remove]
    for name, entry in sorted(candidates.items()):
        source = entry["source"]
        if _plugin_by_location(name, source) or not entry["referenced"]:
            if verbose:
                print(f"  dropping unreferenced dependency {name}")
            for dependent_key in entry["dependents"]:
                drops.setdefault(dependent_key, []).append(name)
            continue
        planned[soname_of(patchelf, source) or source.name] = source

    # ---- materialize: copy into lib/, then patch the copies only ----------
    if planned:
        lib_dir.mkdir(parents=True, exist_ok=True)
    for target, source in sorted(planned.items()):
        destination = lib_dir / target
        shutil.copy2(source, destination)
        _patchelf(patchelf, "--set-rpath", "$ORIGIN", str(destination))
        if verbose:
            print(f"  bundled {target}  <-  {source}")

    for dependent_key, names in drops.items():
        if dependent_key == _EXTENSION_KEY:
            target_file = extension
        else:
            target_file = lib_dir / dependent_key
            if not target_file.is_file():
                continue  # the dependent itself was dropped; its copy never exists
        for name in names:
            _patchelf(patchelf, "--remove-needed", name, str(target_file))

    _patchelf(patchelf, "--set-rpath", "$ORIGIN/lib", str(extension))

    # Ship the C API headers so C++ consumers only need our include directory.
    include_dir = package_dir / "include"
    include_dir.mkdir(parents=True, exist_ok=True)
    for header in PUBLIC_HEADERS:
        for directory in header_dirs:
            source = Path(directory) / header
            if source.is_file():
                shutil.copy2(source, include_dir / header)
                break

    _verify(extension, lib_dir, patchelf)
    return sorted(planned)


def _verify(extension, lib_dir, patchelf):
    problems = []
    for elf in [extension, *sorted(lib_dir.glob("*.so*"))]:
        for name in needed_entries(patchelf, elf):
            if name in SYSTEM_LIBS:
                continue
            if not (lib_dir / name).is_file():
                problems.append(f"{elf.name}: {name} not found in bundled lib/")
    if problems:
        sys.exit("error: bundled wheel is not self-contained:\n  " + "\n  ".join(problems))
    print(f"bundle verification passed for {extension}")
