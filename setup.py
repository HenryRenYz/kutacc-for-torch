import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


ROOT = Path(__file__).parent.resolve()
KUTACC_HOME = Path(os.environ.get("KUTACC_HOME", "/usr/local"))
KUPL_HOME = Path(os.environ.get("KUPL_HOME", os.environ.get("KUPL_PATH", "/usr/local")))
BISHENG_HOME = Path(os.environ.get("BISHENG_HOME", "/home/ryz/BiShengCompiler-5.1.0.2-aarch64-linux"))

library_dirs = [str(KUTACC_HOME / "lib"), str(KUPL_HOME / "lib")]
libraries = ["kutacc", "kupl", "numa", "pthread"]
if (BISHENG_HOME.joinpath("lib", "libomp.so").exists()):
    library_dirs.append(str(BISHENG_HOME / "lib"))
    libraries.append("omp")

rpaths = [str(KUTACC_HOME / "lib"), str(KUPL_HOME / "lib")]
if BISHENG_HOME.joinpath("lib").is_dir():
    rpaths.append(str(BISHENG_HOME / "lib"))

setup(
    name="kutacc-for-torch",
    version="0.1.0",
    description="PyTorch distributed backend powered by Kutacc shared-memory collectives",
    packages=find_packages(),
    package_data={"kutacc_for_torch": ["include/kutacc_for_torch/*.hpp"]},
    ext_modules=[
        CppExtension(
            name="kutacc_for_torch._C",
            sources=[str(ROOT / "csrc" / "process_group_kutacc.cpp")],
            include_dirs=[
                str(ROOT / "kutacc_for_torch" / "include"),
                str(KUTACC_HOME / "include"),
                str(KUPL_HOME / "include"),
            ],
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=["-O3", "-std=c++17", "-Wall", "-Wextra"],
            extra_link_args=[f"-Wl,-rpath,{path}" for path in rpaths],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    python_requires=">=3.9",
)
