import os
import torch
import json
from einops import rearrange
from contextlib import nullcontext
from typing import List
from pathlib import Path
from .utils import log, check_diffusers_version, print_memory
from diffusers.video_processor import VideoProcessor

from .hyvideo.constants import PROMPT_TEMPLATE, NEGATIVE_PROMPT, PRECISION_TO_TYPE
from .hyvideo.vae import load_vae
from .hyvideo.text_encoder import TextEncoder
from .hyvideo.utils.data_utils import align_to
from .hyvideo.modules.posemb_layers import get_nd_rotary_pos_embed
from .hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler
from .hyvideo.diffusion.pipelines import HunyuanVideoPipeline
from .hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from .hyvideo.modules.models import HYVideoDiffusionTransformer
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file

script_directory = os.path.dirname(os.path.abspath(__file__))

def get_rotary_pos_embed(transformer, video_length, height, width):
        target_ndim = 3
        ndim = 5 - 2
        rope_theta = 225
        patch_size = transformer.patch_size
        rope_dim_list = transformer.rope_dim_list
        hidden_size = transformer.hidden_size
        heads_num = transformer.heads_num
        head_dim = hidden_size // heads_num

        # 884
        latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]

        if isinstance(patch_size, int):
            assert all(s % patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // patch_size for s in latents_size]
        elif isinstance(patch_size, list):
            assert all(
                s % patch_size[idx] == 0
                for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [
                s // patch_size[idx] for idx, s in enumerate(latents_size)
            ]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

class HyVideoBlockSwap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "double_blocks_to_swap": ("INT", {"default": 20, "min": 0, "max": 20, "step": 1, "tooltip": "Number of double blocks to swap"}),
                "single_blocks_to_swap": ("INT", {"default": 0, "min": 0, "max": 40, "step": 1, "tooltip": "Number of single blocks to swap"}),
            },
        }
    RETURN_TYPES = ("BLOCKSWAPARGS",)
    RETURN_NAMES = ("block_swap_args",)
    FUNCTION = "setargs"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "Settings for block swapping, reduces VRAM use by swapping blocks to CPU memory"

    def setargs(self, **kwargs):
        return (kwargs, )
    
#region Model loading
class HyVideoModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),
            
            "base_precision": (["fp16", "fp32", "bf16"], {"default": "bf16"}),
            "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'torchao_fp8dq', "torchao_fp8dqrow", "torchao_int8dq", "torchao_fp6"], {"default": 'disabled', "tooltip": "optional quantization method"}),
            "load_device": (["main_device", "offload_device"], {"default": "main_device"}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn",
                    "sageattn_varlen",
                    ], {"default": "flash_attn"}),
                "compile_args": ("COMPILEARGS", ),
                "block_swap_args": ("BLOCKSWAPARGS", ),
            }
        }

    RETURN_TYPES = ("HYVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper"

    def loadmodel(self, model, base_precision, load_device,  quantization,
                  compile_args=None, attention_mode="sdpa", block_swap_args=None):
        transformer = None
        manual_offloading = True
        if "sage" in attention_mode:
            try:
                from sageattention import sageattn_varlen
            except Exception as e:
                raise ValueError(f"Can't import SageAttention: {str(e)}")

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        manual_offloading = True
        transformer_load_device = device if load_device == "main_device" else offload_device
        mm.soft_empty_cache()

        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[base_precision]

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        sd = load_torch_file(model_path, device=transformer_load_device)

        in_channels = out_channels = 16
        factor_kwargs = {"device": transformer_load_device, "dtype": base_dtype}
        HUNYUAN_VIDEO_CONFIG = {
            "mm_double_blocks_depth": 20,
            "mm_single_blocks_depth": 40,
            "rope_dim_list": [16, 56, 56],
            "hidden_size": 3072,
            "heads_num": 24,
            "mlp_width_ratio": 4,
            "guidance_embed": True,
        }
        with init_empty_weights():
            transformer = HYVideoDiffusionTransformer(
                in_channels=in_channels,
                out_channels=out_channels,
                attention_mode=attention_mode,
                main_device=device,
                offload_device=offload_device,
                **HUNYUAN_VIDEO_CONFIG,
                **factor_kwargs
            )

        log.info("Using accelerate to load and assign model weights to device...")
        if quantization == "fp8_e4m3fn" or quantization == "fp8_e4m3fn_fast":
            dtype = torch.float8_e4m3fn
        else:
            dtype = base_dtype
        params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
        for name, param in transformer.named_parameters():
            dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
            set_module_tensor_to_device(transformer, name, device=transformer_load_device, dtype=dtype_to_use, value=sd[name])
        transformer.eval()

        if quantization == "fp8_e4m3fn_fast":
            from .fp8_optimization import convert_fp8_linear
            if "1.5" in model:
                params_to_keep.update({"ff"}) #otherwise NaNs
            convert_fp8_linear(transformer, base_dtype, params_to_keep=params_to_keep)
        
        #compile
        if compile_args is not None:
            torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
            if compile_args["compile_single_blocks"]:
                for i, block in enumerate(transformer.single_blocks):
                    transformer.single_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
            if compile_args["compile_double_blocks"]:
                for i, block in enumerate(transformer.double_blocks):
                    transformer.double_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
            if compile_args["compile_txt_in"]:
                transformer.txt_in = torch.compile(transformer.txt_in, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
            if compile_args["compile_vector_in"]:
                transformer.vector_in = torch.compile(transformer.vector_in, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
            if compile_args["compile_final_layer"]:
                transformer.final_layer = torch.compile(transformer.final_layer, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])

        if "torchao" in quantization:
            try:
                from torchao.quantization import (
                quantize_,
                fpx_weight_only,
                float8_dynamic_activation_float8_weight,
                int8_dynamic_activation_int8_weight
            )
            except:
                raise ImportError("torchao is not installed, please install torchao to use fp8dq")

            # def filter_fn(module: nn.Module, fqn: str) -> bool:
            #     target_submodules = {'attn1', 'ff'} # avoid norm layers, 1.5 at least won't work with quantized norm1 #todo: test other models
            #     if any(sub in fqn for sub in target_submodules):
            #         return isinstance(module, nn.Linear)
            #     return False
            
            if "fp6" in quantization: #slower for some reason on 4090
                quant_func = fpx_weight_only(3, 2)
            elif "fp8dq" in quantization: #very fast on 4090 when compiled
                quant_func = float8_dynamic_activation_float8_weight()
            elif 'fp8dqrow' in quantization:
                from torchao.quantization.quant_api import PerRow
                quant_func = float8_dynamic_activation_float8_weight(granularity=PerRow())
            elif 'int8dq' in quantization:
                quant_func = int8_dynamic_activation_int8_weight()
        
            quantize_(transformer, quant_func)
            
            manual_offloading = False # to disable manual .to(device) calls
            log.info(f"Quantized transformer blocks to {quantization}")

        
        scheduler = FlowMatchDiscreteScheduler(
            shift=9.0,
            reverse=True,
            solver="euler",
        )
        
        pipe = HunyuanVideoPipeline(
            transformer=transformer,
            scheduler=scheduler,
            progress_bar_config=None
        )

        pipeline = {
            "pipe": pipe,
            "dtype": base_dtype,
            "base_path": model_path,
            "model_name": model,
            "manual_offloading": manual_offloading,
            "quantization": "disabled",
            "block_swap_args": block_swap_args
        }
        return (pipeline,)
    
#region load VAE

class HyVideoVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae'"}),
            },
            "optional": {
                "precision": (["fp16", "fp32", "bf16"],
                    {"default": "bf16"}
                ),
                "compile_args":("COMPILEARGS", ),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae", )
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "Loads Hunyuan VAE model from 'ComfyUI/models/vae'"

    def loadmodel(self, model_name, precision, compile_args=None):
        
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        with open(os.path.join(script_directory, 'configs', 'hy_vae_config.json')) as f:
            vae_config = json.load(f)
        model_path = folder_paths.get_full_path("vae", model_name)
        vae_sd = load_torch_file(model_path)

        vae = AutoencoderKLCausal3D.from_config(vae_config).to(dtype).to(offload_device)
        vae.load_state_dict(vae_sd)
        vae.requires_grad_(False)
        vae.eval()
        #compile
        if compile_args is not None:
            torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
            vae = torch.compile(vae, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])

        return (vae,)
    

    
class HyVideoTorchCompileSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                "compile_single_blocks": ("BOOLEAN", {"default": True, "tooltip": "Compile single blocks"}),
                "compile_double_blocks": ("BOOLEAN", {"default": True, "tooltip": "Compile double blocks"}),
                "compile_txt_in": ("BOOLEAN", {"default": False, "tooltip": "Compile txt_in layers"}),
                "compile_vector_in": ("BOOLEAN", {"default": False, "tooltip": "Compile vector_in layers"}),
                "compile_final_layer": ("BOOLEAN", {"default": False, "tooltip": "Compile final layer"}),

            },
        }
    RETURN_TYPES = ("COMPILEARGS",)
    RETURN_NAMES = ("torch_compile_args",)
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "torch.compile settings, when connected to the model loader, torch.compile of the selected layers is attempted. Requires Triton and torch 2.5.0 is recommended"

    def loadmodel(self, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, compile_single_blocks, compile_double_blocks, compile_txt_in, compile_vector_in, compile_final_layer):

        compile_args = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": mode,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "compile_single_blocks": compile_single_blocks,
            "compile_double_blocks": compile_double_blocks,
            "compile_txt_in": compile_txt_in,
            "compile_vector_in": compile_vector_in,
            "compile_final_layer": compile_final_layer
        }

        return (compile_args, )
    
#region TextEncode

class DownloadAndLoadHyVideoTextEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llm_model": (["Kijai/llava-llama-3-8b-text-encoder-tokenizer",],),
                "clip_model": (["disabled","openai/clip-vit-large-patch14",],),
                
                 "precision": (["fp16", "fp32", "bf16"],
                    {"default": "bf16"}
                ),
            },
            "optional": {
                "apply_final_norm": ("BOOLEAN", {"default": False}),
                "hidden_state_skip_layer": ("INT", {"default": 2}),
            }
        }

    RETURN_TYPES = ("HYVIDTEXTENCODER",)
    RETURN_NAMES = ("hyvid_text_encoder", )
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "Loads Hunyuan text_encoder model from 'ComfyUI/models/LLM'"

    def loadmodel(self, llm_model, clip_model, precision, apply_final_norm=False, hidden_state_skip_layer=2):
        
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        if clip_model != "disabled":
            clip_model_path = os.path.join(folder_paths.models_dir, "clip", "clip-vit-large-patch14")
            if not os.path.exists(clip_model_path):
                log.info(f"Downloading clip model to: {clip_model_path}")
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=clip_model,
                    ignore_patterns=["*.msgpack", "*.bin", "*.h5"],
                    local_dir=clip_model_path,
                    local_dir_use_symlinks=False,
                )

            text_encoder_2 = TextEncoder(
            text_encoder_path=clip_model_path,
            text_encoder_type="clipL",
            max_length=77,
            text_encoder_precision=precision,
            tokenizer_type="clipL",
            logger=log,
            device=device,
        )
        else:
            text_encoder_2 = None

        download_path = os.path.join(folder_paths.models_dir,"LLM")
        base_path = os.path.join(download_path, (llm_model.split("/")[-1]))
        if not os.path.exists(base_path):
            log.info(f"Downloading model to: {base_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=llm_model,
                local_dir=base_path,
                local_dir_use_symlinks=False,
            )
        # prompt_template 
        prompt_template = (
            PROMPT_TEMPLATE["dit-llm-encode"]
        )
        # prompt_template_video
        prompt_template_video = (
            PROMPT_TEMPLATE["dit-llm-encode-video"]
        )
       
        text_encoder = TextEncoder(
            text_encoder_path=base_path,
            text_encoder_type="llm",
            max_length=256,
            text_encoder_precision=precision,
            tokenizer_type="llm",
            prompt_template=prompt_template,
            prompt_template_video=prompt_template_video,
            hidden_state_skip_layer=hidden_state_skip_layer,
            apply_final_norm=apply_final_norm,
            logger=log,
            device=device,
        )
       
        
        hyvid_text_encoders = {
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
        }
       
        return (hyvid_text_encoders,)
    
class HyVideoTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text_encoders": ("HYVIDTEXTENCODER",),
            "prompt": ("STRING", {"default": "", "multiline": True} ),
            #"negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "force_offload": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("HYVIDEMBEDS", )
    RETURN_NAMES = ("hyvid_embeds",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper"

    def process(self, text_encoders, prompt, negative_prompt, force_offload=True):
        device = mm.text_encoder_device()
        offload_device = mm.text_encoder_offload_device()

        text_encoder_1 = text_encoders["text_encoder"]
        text_encoder_2 = text_encoders["text_encoder_2"]


        def encode_prompt(self, prompt, negative_prompt, text_encoder):
            batch_size = 1
            num_videos_per_prompt = 1
            do_classifier_free_guidance = True
            data_type = "video"
            
            text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

            prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type, device=device)
            prompt_embeds = prompt_outputs.hidden_state
            
            attention_mask = prompt_outputs.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                attention_mask = attention_mask.view(
                    bs_embed * num_videos_per_prompt, seq_len
                )

            if text_encoder is not None:
                prompt_embeds_dtype = text_encoder.dtype
            elif self.transformer is not None:
                prompt_embeds_dtype = self.transformer.dtype
            else:
                prompt_embeds_dtype = prompt_embeds.dtype

            prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            if prompt_embeds.ndim == 2:
                bs_embed, _ = prompt_embeds.shape
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
                prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
            else:
                bs_embed, seq_len, _ = prompt_embeds.shape
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(
                    bs_embed * num_videos_per_prompt, seq_len, -1
                )

            # get unconditional embeddings for classifier free guidance
            if do_classifier_free_guidance:
                uncond_tokens: List[str]
                if negative_prompt is None:
                    uncond_tokens = [""] * batch_size
                elif prompt is not None and type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt]
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = negative_prompt

                # max_length = prompt_embeds.shape[1]
                uncond_input = text_encoder.text2tokens(uncond_tokens, data_type=data_type)

                negative_prompt_outputs = text_encoder.encode(
                    uncond_input, data_type=data_type, device=device
                )
                negative_prompt_embeds = negative_prompt_outputs.hidden_state

                negative_attention_mask = negative_prompt_outputs.attention_mask
                if negative_attention_mask is not None:
                    negative_attention_mask = negative_attention_mask.to(device)
                    _, seq_len = negative_attention_mask.shape
                    negative_attention_mask = negative_attention_mask.repeat(
                        1, num_videos_per_prompt
                    )
                    negative_attention_mask = negative_attention_mask.view(
                        batch_size * num_videos_per_prompt, seq_len
                    )

            if do_classifier_free_guidance:
                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                seq_len = negative_prompt_embeds.shape[1]

                negative_prompt_embeds = negative_prompt_embeds.to(
                    dtype=prompt_embeds_dtype, device=device
                )

                if negative_prompt_embeds.ndim == 2:
                    negative_prompt_embeds = negative_prompt_embeds.repeat(
                        1, num_videos_per_prompt
                    )
                    negative_prompt_embeds = negative_prompt_embeds.view(
                        batch_size * num_videos_per_prompt, -1
                    )
                else:
                    negative_prompt_embeds = negative_prompt_embeds.repeat(
                        1, num_videos_per_prompt, 1
                    )
                    negative_prompt_embeds = negative_prompt_embeds.view(
                        batch_size * num_videos_per_prompt, seq_len, -1
                    )

            return (
                prompt_embeds,
                negative_prompt_embeds,
                attention_mask,
                negative_attention_mask,
            )
        text_encoder_1.to(device)
        prompt_embeds, negative_prompt_embeds, attention_mask, negative_attention_mask = encode_prompt(self, prompt, negative_prompt, text_encoder_1)
        if force_offload:
            text_encoder_1.to(offload_device)
            mm.soft_empty_cache()

        if text_encoder_2 is not None:
            text_encoder_2.to(device)
            prompt_embeds_2, negative_prompt_embeds_2, attention_mask_2, negative_attention_mask_2 = encode_prompt(self, prompt, negative_prompt, text_encoder_2)
            if force_offload:
                text_encoder_2.to(offload_device)
                mm.soft_empty_cache()
        else:
            prompt_embeds_2 = None
            negative_prompt_embeds_2 = None
            attention_mask_2 = None
            negative_attention_mask_2 = None
        
        prompt_embeds_dict = {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "attention_mask": attention_mask,
                "negative_attention_mask": negative_attention_mask,
                "prompt_embeds_2": prompt_embeds_2,
                "negative_prompt_embeds_2": negative_prompt_embeds_2,
                "attention_mask_2": attention_mask_2,
                "negative_attention_mask_2": negative_attention_mask_2,
            }
        return (prompt_embeds_dict,)
   

#region Sampler    
class HyVideoSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("HYVIDEOMODEL",),
                "hyvid_embeds": ("HYVIDEMBEDS", ),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
                "num_frames": ("INT", {"default": 49, "min": 1, "max": 1024, "step": 4}),
                "steps": ("INT", {"default": 30, "min": 1}),
                "embedded_guidance_scale": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "flow_shift": ("FLOAT", {"default": 9.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True}),
                
            },
            "optional": {
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"} ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper"

    def process(self, model, hyvid_embeds, flow_shift, steps, embedded_guidance_scale, seed, width, height, num_frames, samples=None, denoise_strength=1.0, force_offload=True):
        mm.unload_all_models()
        mm.soft_empty_cache()

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = model["dtype"]
        
        generator = torch.Generator(device=torch.device("cpu")).manual_seed(seed)

        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        if width <= 0 or height <= 0 or num_frames <= 0:
            raise ValueError(
                f"`height` and `width` and `video_length` must be positive integers, got height={height}, width={width}, video_length={num_frames}"
            )
        if (num_frames - 1) % 4 != 0:
            raise ValueError(
                f"`video_length-1` must be a multiple of 4, got {num_frames}"
            )

        log.info(
            f"Input (height, width, video_length) = ({height}, {width}, {num_frames})"
        )

        target_height = align_to(height, 16)
        target_width = align_to(width, 16)

        freqs_cos, freqs_sin = get_rotary_pos_embed(
            model["pipe"].transformer, num_frames, target_height, target_width
        )
        n_tokens = freqs_cos.shape[0]

        model["pipe"].scheduler.shift = flow_shift
  
        # autocast_context = torch.autocast(
        #     mm.get_autocast_device(device), dtype=dtype
        # ) if any(q in model["quantization"] for q in ("e4m3fn", "GGUF")) else nullcontext()
        #with autocast_context:
        if model["block_swap_args"] is not None:
            for name, param in model["pipe"].transformer.named_parameters():
                #print(name, param.data.device)
                if "single" not in name and "double" not in name:
                    param.data = param.data.to(device)
                
            model["pipe"].transformer.block_swap(model["block_swap_args"]["double_blocks_to_swap"] , model["block_swap_args"]["single_blocks_to_swap"])
            # for name, param in model["pipe"].transformer.named_parameters():
            #     print(name, param.data.device)
          
        elif model["manual_offloading"]:
            model["pipe"].transformer.to(device)
        
        out_latents = model["pipe"](
            num_inference_steps=steps,
            height = target_height,
            width = target_width,
            video_length = num_frames,
            guidance_scale=1.0,
            embedded_guidance_scale=embedded_guidance_scale,
            latents=samples["samples"] if samples is not None else None,
            denoise_strength=denoise_strength,
            prompt_embed_dict=hyvid_embeds,
            generator=generator,
            freqs_cis=(freqs_cos, freqs_sin),
            n_tokens=n_tokens,
        )

        print_memory(device)
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        if force_offload:
            if model["manual_offloading"]:
                model["pipe"].transformer.to(offload_device)
                mm.soft_empty_cache()

        return ({
            "samples": out_latents
            },)

    
#region VideoDecode    
class HyVideoDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("VAE",),
                    "samples": ("LATENT",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": True, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "temporal_tiling_sample_size": ("INT", {"default": 16, "min": 4, "max": 256, "tooltip": "Smaller values use less VRAM, model default is 64 which doesn't fit on most GPUs"}),
                    },            
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "HunyuanVideoWrapper"

    def decode(self, vae, samples, enable_vae_tiling, temporal_tiling_sample_size):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        latents = samples["samples"]
        generator = torch.Generator(device=torch.device("cpu"))#.manual_seed(seed)
        vae.to(device)
        vae.sample_tsize = temporal_tiling_sample_size
        
        
        expand_temporal_dim = False
        if len(latents.shape) == 4:
            if isinstance(vae, AutoencoderKLCausal3D):
                latents = latents.unsqueeze(2)
                expand_temporal_dim = True
        elif len(latents.shape) == 5:
            pass
        else:
            raise ValueError(
                f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
            )

        latents = latents / vae.config.scaling_factor
        latents = latents.to(vae.dtype).to(device)

        if enable_vae_tiling:
            vae.enable_tiling()
            video = vae.decode(
                latents, return_dict=False, generator=generator
            )[0]
        else:
            video = vae.decode(
                latents, return_dict=False, generator=generator
            )[0]

        if expand_temporal_dim or video.shape[2] == 1:
            video = video.squeeze(2)

        vae.to(offload_device)
        mm.soft_empty_cache()
       
        video_processor = VideoProcessor(vae_scale_factor=8)
        video_processor.config.do_resize = False

        video = video_processor.postprocess_video(video=video, output_type="pt")
        video = video[0].permute(0, 2, 3, 1).cpu().float()

        return (video,)
    
class HyVideoEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("VAE",),
                    "image": ("IMAGE",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": True, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "temporal_tiling_sample_size": ("INT", {"default": 16, "min": 4, "max": 256, "tooltip": "Smaller values use less VRAM, model default is 64 which doesn't fit on most GPUs"}),
                    },            
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "HunyuanVideoWrapper"

    def encode(self, vae, image, enable_vae_tiling, temporal_tiling_sample_size):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        generator = torch.Generator(device=torch.device("cpu"))#.manual_seed(seed)
        vae.to(device)
        vae.sample_tsize = temporal_tiling_sample_size

        image = (image * 2.0 - 1.0).to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W
        if enable_vae_tiling:
            vae.enable_tiling()
        latents = vae.encode(image).latent_dist.sample(generator)
        latents = latents * vae.config.scaling_factor
        vae.to(offload_device)
        print("encoded latents shape",latents.shape)      
        

        return ({"samples": latents},)

class CogVideoLatentPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                 "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                 "min_val": ("FLOAT", {"default": -0.15, "min": -1.0, "max": 0.0, "step": 0.001}),
                 "max_val": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.001}),
                 "r_bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                 "g_bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                 "b_bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", )
    RETURN_NAMES = ("images", "latent_rgb_factors",)
    FUNCTION = "sample"
    CATEGORY = "PyramidFlowWrapper"

    def sample(self, samples, seed, min_val, max_val, r_bias, g_bias, b_bias):
        mm.soft_empty_cache()

        latents = samples["samples"].clone()
        print("in sample", latents.shape)
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
 
        #[[0.0658900170023352, 0.04687556512203313, -0.056971557475649186], [-0.01265770449940036, -0.02814809569100843, -0.0768912512529372], [0.061456544746314665, 0.0005511617552452358, -0.0652574975291287], [-0.09020669168815276, -0.004755440180558637, -0.023763970904494294], [0.031766964513999865, -0.030959599938418375, 0.08654669098083616], [-0.005981764690055846, -0.08809119252349802, -0.06439852368217663], [-0.0212114426433989, 0.08894281999597677, 0.05155629477559985], [-0.013947446911030725, -0.08987475069900677, -0.08923124751217484], [-0.08235967967978511, 0.07268025379974379, 0.08830486164536037], [-0.08052049179735378, -0.050116143175332195, 0.02023752569687405], [-0.07607527759162447, 0.06827156419895981, 0.08678111754261035], [-0.04689089232553825, 0.017294986041038893, -0.10280492336438908], [-0.06105783150270304, 0.07311850680875913, 0.019995735372550075], [-0.09232589996527711, -0.012869815059053047, -0.04355587834255975], [-0.06679931010802251, 0.018399815879067458, 0.06802404982033876], [-0.013062632927118165, -0.04292991477896661, 0.07476243356192845]]
        latent_rgb_factors =[[0.11945946736445662, 0.09919175788574555, -0.004832707433877734], [-0.0011977028264356232, 0.05496505130267682, 0.021321622433638193], [-0.014088548986590666, -0.008701477861945644, -0.020991313281459367], [0.03063921972519621, 0.12186477097625073, 0.0139593690235148], [0.0927403067854673, 0.030293187650929136, 0.05083134241694003], [0.0379112441305742, 0.04935199882777209, 0.058562766246777774], [0.017749911959153715, 0.008839453404921545, 0.036005638019226294], [0.10610119248526109, 0.02339855688237826, 0.057154257614084596], [0.1273639464837117, -0.010959856130713416, 0.043268631260428896], [-0.01873510946881321, 0.08220930648486932, 0.10613256772247093], [0.008429116376722327, 0.07623856561000408, 0.09295712117576727], [0.12938137079617007, 0.12360403483892413, 0.04478930933220116], [0.04565908794779364, 0.041064156741596365, -0.017695041535528512], [0.00019003240570281826, -0.013965147883381978, 0.05329669529635849], [0.08082391586738358, 0.11548306825496074, -0.021464170006615893], [-0.01517932393230994, -0.0057985555313003236, 0.07216646476618871]]
        import random
        random.seed(seed)
        latent_rgb_factors = [[random.uniform(min_val, max_val) for _ in range(3)] for _ in range(16)]
        out_factors = latent_rgb_factors
        print(latent_rgb_factors)
       
        latent_rgb_factors_bias = [0.085, 0.137, 0.158]
        #latent_rgb_factors_bias = [r_bias, g_bias, b_bias]
        
        latent_rgb_factors = torch.tensor(latent_rgb_factors, device=latents.device, dtype=latents.dtype).transpose(0, 1)
        latent_rgb_factors_bias = torch.tensor(latent_rgb_factors_bias, device=latents.device, dtype=latents.dtype)

        print("latent_rgb_factors", latent_rgb_factors.shape)

        latent_images = []
        for t in range(latents.shape[2]):
            latent = latents[:, :, t, :, :]
            latent = latent[0].permute(1, 2, 0)
            latent_image = torch.nn.functional.linear(
                latent,
                latent_rgb_factors,
                bias=latent_rgb_factors_bias
            )
            latent_images.append(latent_image)
        latent_images = torch.stack(latent_images, dim=0)
        print("latent_images", latent_images.shape)
        latent_images_min = latent_images.min()
        latent_images_max = latent_images.max()
        latent_images = (latent_images - latent_images_min) / (latent_images_max - latent_images_min)
        
        return (latent_images.float().cpu(), out_factors)
    
NODE_CLASS_MAPPINGS = {
    "HyVideoSampler": HyVideoSampler,
    "HyVideoDecode": HyVideoDecode,
    "HyVideoTextEncode": HyVideoTextEncode,
    "HyVideoModelLoader": HyVideoModelLoader,
    "HyVideoVAELoader": HyVideoVAELoader,
    "DownloadAndLoadHyVideoTextEncoder": DownloadAndLoadHyVideoTextEncoder,
    "HyVideoEncode": HyVideoEncode,
    "HyVideoBlockSwap": HyVideoBlockSwap,
    "HyVideoTorchCompileSettings": HyVideoTorchCompileSettings,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HyVideoSampler": "HunyuanVideo Sampler",
    "HyVideoDecode": "HunyuanVideo Decode",
    "HyVideoTextEncode": "HunyuanVideo TextEncode",
    "HyVideoModelLoader": "HunyuanVideo Model Loader",
    "HyVideoVAELoader": "HunyuanVideo VAE Loader",
    "DownloadAndLoadHyVideoTextEncoder": "(Down)Load HunyuanVideo TextEncoder",
    "HyVideoEncode": "HunyuanVideo Encode",
    "HyVideoBlockSwap": "HunyuanVideo BlockSwap",
    "HyVideoTorchCompileSettings": "HunyuanVideo Torch Compile Settings",
    }
