import os
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def zfp_prefix() -> Path:
    candidates = [os.environ.get("ZFP_DIR"), Path.home() / ".local", "/usr/local", "/usr"]
    for c in filter(None, candidates):
        if (Path(c) / "include" / "zfp.h").exists():
            return Path(c)
    raise RuntimeError("could not find zfp.h; set ZFP_DIR to the libzfp install prefix")


class BuildShared(build_ext):
    # the library is loaded with ctypes, so give it a fixed name instead of the Python ABI suffix
    def get_ext_filename(self, ext_name):
        return os.path.join(*ext_name.split(".")) + ".so"


zfp = zfp_prefix()
csrc = sorted(str(p) for p in Path("skipzfp/csrc").glob("*.c"))
lib_dirs = [str(zfp / d) for d in ("lib", "lib64") if (zfp / d).exists()]

setup(
    ext_modules=[
        Extension(
            "skipzfp._core",
            sources=csrc,
            include_dirs=[str(zfp / "include")],
            library_dirs=lib_dirs,
            runtime_library_dirs=lib_dirs,
            libraries=["zfp", "m"],
            extra_compile_args=["-O3", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        )
    ],
    cmdclass={"build_ext": BuildShared},
)
