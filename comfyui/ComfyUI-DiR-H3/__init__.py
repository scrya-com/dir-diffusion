# Copyright 2026 scrya-com contributors. Licensed under the Apache License, Version 2.0.
"""DiR few-step generation (and DWD decoding) for MiniMax H3 in ComfyUI. See README.md."""

from comfy_api.latest import ComfyExtension

from .nodes import NODES


class DiRH3Extension(ComfyExtension):
    async def get_node_list(self):
        return list(NODES)


async def comfy_entrypoint() -> DiRH3Extension:
    return DiRH3Extension()


__all__ = ["comfy_entrypoint"]
