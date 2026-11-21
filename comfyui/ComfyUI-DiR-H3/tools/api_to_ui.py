#!/usr/bin/env python3
# Copyright 2026 scrya-com contributors. Licensed under the Apache License, Version 2.0.
"""Make a ComfyUI UI-format workflow (the one the editor loads by drag and drop) from an API-format one.

    CUDA_VISIBLE_DEVICES= /path/to/comfyui/python tools/api_to_ui.py ../dir_h3_video_api.json ../dir_h3_video.json

Needs ComfyUI's node schemas (``COMFYUI_DIR``, default /media/2TB/ComfyUI) for the widget order: socket inputs become
links, widget inputs ``widgets_values`` in ``INPUT_TYPES`` order (a seed with ``control_after_generate`` gets "fixed"
after its value). Nodes are laid out in columns by depth. CPU only.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
PACK = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get("COMFYUI_DIR", "/media/2TB/ComfyUI"))
sys.path.insert(0, str(COMFY))
import comfy.cli_args  # noqa: E402

comfy.cli_args.args.cpu = True

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def classes() -> dict:
    import nodes
    from comfy_extras import nodes_images, nodes_minimax_h3

    spec = importlib.util.spec_from_file_location(PACK.name, PACK / "__init__.py", submodule_search_locations=[str(PACK)])
    pack = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pack
    spec.loader.exec_module(pack)
    out = dict(nodes.NODE_CLASS_MAPPINGS)
    for module in (nodes_minimax_h3, nodes_images, pack):
        for node in asyncio.run(asyncio.run(module.comfy_entrypoint()).get_node_list()):
            out[node.GET_SCHEMA().node_id] = node
    return out


def outputs_of(cls) -> list[tuple[str, str]]:
    if hasattr(cls, "GET_NODE_INFO_V1"):
        info = cls.GET_NODE_INFO_V1()
        return list(zip(info["output_name"], info["output"]))
    names = getattr(cls, "RETURN_NAMES", None) or cls.RETURN_TYPES
    return list(zip(names, cls.RETURN_TYPES))


def is_widget(spec) -> bool:
    return isinstance(spec[0], list) or spec[0] in WIDGET_TYPES


def convert(api: dict, table: dict) -> dict:
    ids = {key: i + 1 for i, key in enumerate(api)}
    depth: dict[str, int] = {}

    def level(key):
        if key not in depth:
            parents = [v[0] for v in api[key]["inputs"].values() if isinstance(v, list)]
            depth[key] = 1 + max((level(p) for p in parents), default=-1)
        return depth[key]

    for key in api:
        level(key)
    links, ui_nodes, per_column = [], {}, {}
    for key, node in api.items():
        cls = table[node["class_type"]]
        spec = cls.INPUT_TYPES()
        ordered = list(spec.get("required", {}).items()) + list(spec.get("optional", {}).items())
        inputs, widgets = [], []
        for name, s in ordered:
            value = node["inputs"].get(name)
            if is_widget(s) and not isinstance(value, list):
                default = s[1].get("default") if len(s) > 1 and isinstance(s[1], dict) else None
                widgets.append(default if value is None else value)
                if len(s) > 1 and isinstance(s[1], dict) and s[1].get("control_after_generate"):
                    widgets.append("fixed")
            elif not is_widget(s) or isinstance(value, list):
                kind = s[0] if isinstance(s[0], str) else "COMBO"
                entry = {"name": name, "type": kind, "link": None}
                if is_widget(s):
                    entry["widget"] = {"name": name}
                inputs.append(entry)
        column = depth[key]
        row = per_column.get(column, 0)
        per_column[column] = row + 1
        ui_nodes[key] = {
            "id": ids[key], "type": node["class_type"], "pos": [column * 420, row * 300], "size": [380, 260],
            "flags": {}, "order": 0, "mode": 0, "inputs": inputs,
            "outputs": [{"name": n, "type": t, "links": [], "slot_index": i} for i, (n, t) in enumerate(outputs_of(cls))],
            "properties": {"Node name for S&R": node["class_type"]}, "widgets_values": widgets,
        }
        if node.get("_meta", {}).get("title"):
            ui_nodes[key]["title"] = node["_meta"]["title"]
    for key, node in api.items():
        for slot, entry in enumerate(ui_nodes[key]["inputs"]):
            value = node["inputs"].get(entry["name"])
            if not isinstance(value, list):
                continue
            link_id = len(links) + 1
            src, index = value
            links.append([link_id, ids[src], index, ids[key], slot, ui_nodes[src]["outputs"][index]["type"]])
            entry["link"] = link_id
            ui_nodes[src]["outputs"][index]["links"].append(link_id)
    for order, key in enumerate(sorted(api, key=lambda k: (depth[k], ids[k]))):
        ui_nodes[key]["order"] = order
    return {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(api, sort_keys=True))), "revision": 0,
            "last_node_id": len(ids), "last_link_id": len(links), "nodes": list(ui_nodes.values()), "links": links,
            "groups": [], "config": {}, "extra": {"ds": {"scale": 0.8, "offset": [0, 0]}}, "version": 0.4}


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    ui = convert(json.loads(src.read_text()), classes())
    dst.write_text(json.dumps(ui, indent=1))
    print(f"{dst}: {len(ui['nodes'])} nodes, {len(ui['links'])} links")


if __name__ == "__main__":
    main()
