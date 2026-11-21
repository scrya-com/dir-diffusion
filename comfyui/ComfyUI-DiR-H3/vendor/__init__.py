"""Pure-torch pieces of dir-diffusion's ``lot/`` code, vendored so the pack works when copied on its own.

Copied from https://github.com/scrya-com/dir-diffusion (``lot/dir_h3.py``, ``lot/h3.py``, ``lot/h3_musubi.py``,
``lot/dwd_h3.py``, ``lot/refine_dwd_h3.py``, ``lot/train_dwd_h3.py``) at commit db1f6f3, Apache-2.0. Only the
inference parts are kept; ``tests/test_cpu.py`` checks them against ``lot/`` when that folder is present.
"""
