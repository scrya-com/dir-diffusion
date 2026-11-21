"""The generators the benchmarks can select with ``--generator`` (``accel_h3.py bench``, ``scripts/bench_accel.sh``).

``dir`` (the default) is our DiR checkpoint, run by ``accel_h3``'s own code, unchanged. Any other entry names a
module with ``VARIANTS`` (its benchmark rows), ``add_bench_args(parser)`` and ``bench(args)``, which writes the same
``<out>/<folder>/meta.json`` that ``accel_h3.bench_table`` reads (with ``meta["labels"]`` naming its rows), so one
``BENCH.md`` holds every generator's rows side by side. The module is imported only when selected.

| Name | Module | What |
|---|---|---|
| ``dir`` | ``accel_h3`` | DiR, 4 iterations, with our DWD stack (johndpope/dir-diffusion-h3, -dwd) |
| ``pih3`` | ``pih3_h3`` | pi-H3 by Hansheng Chen, Shengqu Cai, Jiaming Song, Gordon Wetzstein, Xinchen Yan and Leonidas Guibas: 2 or 4 steps, its own DWD head |
"""

from __future__ import annotations

import importlib

GENERATORS = {"dir": None, "pih3": "pih3_h3"}
DEFAULT = "dir"


def generator_module(name: str):
    """The plug-in module of a generator, or ``None`` for the built-in DiR path."""
    if name not in GENERATORS:
        raise SystemExit(f"--generator {name!r}: not one of {sorted(GENERATORS)}")
    module = GENERATORS[name]
    return None if module is None else importlib.import_module(module)
