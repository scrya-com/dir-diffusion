# third_party

An unmodified subset of [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) at commit `f8a1b03` (2026-09-27), Apache-2.0. It is the import closure of the H3 transformer (`minimax_h3/model.py`, `packing.py`), the ConvRot int8 and NVFP4 loaders, block swap, the kohya-style LoRA network, the H3 video VAE and the Qwen3-VL text encoder, as used by `lot/h3_musubi.py`. To use a full checkout instead, set `MUSUBI_SRC=/path/to/musubi-tuner/src`.

To refresh: copy the same paths from a newer musubi-tuner `src/`, rerun `lot/test_lot.py` and `lot/parity_h3.py` (`rel_rms` must stay 0), and update the commit here and in `NOTICE`.

Files:

- `musubi_tuner/convert_lora.py`
- `musubi_tuner/dataset/audio_utils.py`
- `musubi_tuner/dataset/datasources.py`
- `musubi_tuner/dataset/__init__.py`
- `musubi_tuner/dataset/media_utils.py`
- `musubi_tuner/__init__.py`
- `musubi_tuner/minimax_h3/checkpoint.py`
- `musubi_tuner/minimax_h3/__init__.py`
- `musubi_tuner/minimax_h3/media.py`
- `musubi_tuner/minimax_h3/model.py`
- `musubi_tuner/minimax_h3/packing.py`
- `musubi_tuner/minimax_h3/text_encoder.py`
- `musubi_tuner/minimax_h3/video_vae.py`
- `musubi_tuner/modules/attention.py`
- `musubi_tuner/modules/comfy_quant_utils.py`
- `musubi_tuner/modules/convrot_int8_kernels.py`
- `musubi_tuner/modules/convrot_int8_utils.py`
- `musubi_tuner/modules/custom_offloading_utils.py`
- `musubi_tuner/modules/fp8_optimization_utils.py`
- `musubi_tuner/modules/__init__.py`
- `musubi_tuner/modules/nvfp4_utils.py`
- `musubi_tuner/networks/__init__.py`
- `musubi_tuner/networks/lora_minimax_h3.py`
- `musubi_tuner/networks/lora.py`
- `musubi_tuner/utils/cv2_compat.py`
- `musubi_tuner/utils/device_utils.py`
- `musubi_tuner/utils/__init__.py`
- `musubi_tuner/utils/lora_utils.py`
- `musubi_tuner/utils/model_utils.py`
- `musubi_tuner/utils/safetensors_utils.py`
