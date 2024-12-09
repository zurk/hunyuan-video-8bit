import logging
import os
import sys
from typing import Sequence, Mapping, Any, Union
import torch


_log = logging.getLogger("cog_predictor")


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


class HunyanVideoRunner:
    def __init__(self):
        add_comfyui_directory_to_sys_path()
        from nodes import NODE_CLASS_MAPPINGS

        import_custom_nodes()
        with torch.inference_mode():
            _log.info("Loading")
            hyvideomodelloader = NODE_CLASS_MAPPINGS["HyVideoModelLoader"]()
            _log.info("HyVideoModelLoader Loaded")
            self._hyvideomodelloader_1 = hyvideomodelloader.loadmodel(
                model="hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors",
                base_precision="bf16",
                quantization="fp8_e4m3fn",
                load_device="offload_device",
                attention_mode="sageattn_varlen",
            )
            _log.info("hunyuan_video_720_cfgdistill_fp8_e4m3fn Loaded")

            hyvideovaeloader = NODE_CLASS_MAPPINGS["HyVideoVAELoader"]()
            self._hyvideovaeloader_7 = hyvideovaeloader.loadmodel(
                model_name="hunyuan_video_vae_bf16.safetensors", precision="fp16"
            )
            _log.info("hunyuan_video_vae_bf16 Loaded")

            downloadandloadhyvideotextencoder = NODE_CLASS_MAPPINGS[
                "DownloadAndLoadHyVideoTextEncoder"
            ]()
            self._downloadandloadhyvideotextencoder_16 = (
                downloadandloadhyvideotextencoder.loadmodel(
                    llm_model="Kijai/llava-llama-3-8b-text-encoder-tokenizer",
                    clip_model="openai/clip-vit-large-patch14",
                    precision="fp16",
                    apply_final_norm=False,
                    hidden_state_skip_layer=2,
                )
            )
            _log.info("llava-llama-3-8b-text-encoder-tokenizer Loaded")

            self._hyvideotextencode = NODE_CLASS_MAPPINGS["HyVideoTextEncode"]()
            self._hyvideosampler = NODE_CLASS_MAPPINGS["HyVideoSampler"]()
            self._hyvideodecode = NODE_CLASS_MAPPINGS["HyVideoDecode"]()
            self._vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

    def run_hunyan_video(
        self,
        prompt="A cat walks on the grass, realistic style.",
        negative_prompt="",
        width=960,
        height=544,
        num_frames=41,  # Add early check
        steps=30,
        embedded_guidance_scale=6,
        flow_shift=7,
        seed=None,
    ) -> str:
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        _log.info(f"{seed=}")

        with torch.inference_mode():

            hyvideotextencode_30 = self._hyvideotextencode.process(
                prompt=prompt,
                negative_prompt=negative_prompt,
                force_offload=True,
                text_encoders=get_value_at_index(
                    self._downloadandloadhyvideotextencoder_16, 0
                ),
            )

            hyvideosampler_3 = self._hyvideosampler.process(
                width=width,
                height=height,
                num_frames=num_frames,
                steps=steps,
                embedded_guidance_scale=embedded_guidance_scale,
                flow_shift=flow_shift,
                seed=seed,
                force_offload=1,
                denoise_strength=1,
                model=get_value_at_index(self._hyvideomodelloader_1, 0),
                hyvid_embeds=get_value_at_index(hyvideotextencode_30, 0),
            )

            hyvideodecode_5 = self._hyvideodecode.decode(
                enable_vae_tiling=True,
                temporal_tiling_sample_size=8,
                vae=get_value_at_index(self._hyvideovaeloader_7, 0),
                samples=get_value_at_index(hyvideosampler_3, 0),
            )

            vhs_videocombine_34 = self._vhs_videocombine.combine_video(
                frame_rate=16,
                loop_count=0,
                filename_prefix="HunyuanVideo",
                format="video/h264-mp4",
                pix_fmt="yuv420p",
                crf=19,
                save_metadata=False,
                pingpong=False,
                save_output=True,
                images=get_value_at_index(hyvideodecode_5, 0),
                unique_id=18083947834017701655,
            )

            return vhs_videocombine_34["result"][0][1][1]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level
        format="%(asctime)s-%(levelname)s-%(name)s-%(message)s",
        handlers=[logging.StreamHandler()],  # Log to the console
    )
    runner = HunyanVideoRunner()
    runner.run_hunyan_video()
