import time
import random

from pathlib import Path
from loguru import logger

import torch
from hyvideo.constants import PROMPT_TEMPLATE, NEGATIVE_PROMPT, PRECISION_TO_TYPE
from hyvideo.vae import load_vae
from hyvideo.text_encoder import TextEncoder
from hyvideo.utils.data_utils import align_to
from hyvideo.modules.posemb_layers import get_nd_rotary_pos_embed
from hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler
from hyvideo.diffusion.pipelines import HunyuanVideoPipeline

from hyvideo.modules.models import HYVideoDiffusionTransformer, HUNYUAN_VIDEO_CONFIG
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
import safetensors.torch

class Inference(object):
    def __init__(
        self,
        args,
        vae,
        vae_kwargs,
        text_encoder,
        model,
        text_encoder_2=None,
        pipeline=None,
        use_cpu_offload=False,
        device=None,
        logger=None,
    ):
        self.vae = vae
        self.vae_kwargs = vae_kwargs

        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2

        self.model = model
        self.pipeline = pipeline
        self.use_cpu_offload = use_cpu_offload

        self.args = args
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.logger = logger

    @classmethod
    def from_pretrained(cls, pretrained_model_path, args, device=None, **kwargs):
        """
        Initialize the Inference pipeline.

        Args:
            pretrained_model_path (str or pathlib.Path): The model path, including t2v, text encoder and vae checkpoints.
            args (argparse.Namespace): The arguments for the pipeline.
            device (int): The device for inference. Default is 0.
        """
        # ========================================================================
        logger.info(f"Got text-to-video model root path: {pretrained_model_path}")

        # ======================== Get the args path =============================

        # Set device and disable gradient
        #if device is None:
        #    device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_grad_enabled(False)
        device = "cpu"

        # =========================== Build main model ===========================
        logger.info("Building model...")
        factor_kwargs = {"device": device, "dtype": PRECISION_TO_TYPE[args.precision]}
        in_channels = args.latent_channels
        out_channels = args.latent_channels

        # model = load_model(
        #     args,
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     factor_kwargs=factor_kwargs,
        # )
        with init_empty_weights():
            transformer = HYVideoDiffusionTransformer(
                args,
                in_channels=in_channels,
                out_channels=out_channels,
                **HUNYUAN_VIDEO_CONFIG[args.model],
                **factor_kwargs,
            )
        
        #model = Inference.load_state_dict(args, model, pretrained_model_path)
        model_path = "ckpts/hunyuan-video-t2v-720p/transformers/hunyuan_video_720_fp8_e4m3fn_keep_bias.safetensors"
        sd = safetensors.torch.load_file(model_path)
        base_dtype = torch.bfloat16
        dtype = torch.float8_e4m3fn
        params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
        for name, param in transformer.named_parameters():
            dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
            set_module_tensor_to_device(transformer, name, device=device, dtype=dtype_to_use, value=sd[name])
        transformer.eval()
        

        # ============================= Build extra models ========================
        # VAE
        
        vae, _, s_ratio, t_ratio = load_vae(
            vae_type = "884-16c-hy",
            vae_precision = "bf16",
            logger=logger,
            device=device if not args.use_cpu_offload else "cpu",
        )
        vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}

        # Text encoder
        if args.prompt_template_video is not None:
            crop_start = PROMPT_TEMPLATE[args.prompt_template_video].get(
                "crop_start", 0
            )
        elif args.prompt_template is not None:
            crop_start = PROMPT_TEMPLATE[args.prompt_template].get("crop_start", 0)
        else:
            crop_start = 0
        max_length = args.text_len + crop_start

        # prompt_template
        prompt_template = (
            PROMPT_TEMPLATE[args.prompt_template]
            if args.prompt_template is not None
            else None
        )

        # prompt_template_video
        prompt_template_video = (
            PROMPT_TEMPLATE[args.prompt_template_video]
            if args.prompt_template_video is not None
            else None
        )

        text_encoder = TextEncoder(
            text_encoder_type=args.text_encoder,
            max_length=max_length,
            text_encoder_precision=args.text_encoder_precision,
            tokenizer_type=args.tokenizer,
            prompt_template=prompt_template,
            prompt_template_video=prompt_template_video,
            hidden_state_skip_layer=args.hidden_state_skip_layer,
            apply_final_norm=args.apply_final_norm,
            reproduce=args.reproduce,
            logger=logger,
            device=device if not args.use_cpu_offload else "cpu",
        )
        text_encoder_2 = None
        if args.text_encoder_2 is not None:
            text_encoder_2 = TextEncoder(
                text_encoder_type=args.text_encoder_2,
                max_length=args.text_len_2,
                text_encoder_precision=args.text_encoder_precision_2,
                tokenizer_type=args.tokenizer_2,
                reproduce=args.reproduce,
                logger=logger,
                device=device if not args.use_cpu_offload else "cpu",
            )

        return cls(
            args=args,
            vae=vae,
            vae_kwargs=vae_kwargs,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            model=transformer,
            use_cpu_offload=args.use_cpu_offload,
            device=device,
            logger=logger,
        )


    @staticmethod
    def parse_size(size):
        if isinstance(size, int):
            size = [size]
        if not isinstance(size, (list, tuple)):
            raise ValueError(f"Size must be an integer or (height, width), got {size}.")
        if len(size) == 1:
            size = [size[0], size[0]]
        if len(size) != 2:
            raise ValueError(f"Size must be an integer or (height, width), got {size}.")
        return size


class HunyuanVideoSampler(Inference):
    def __init__(
        self,
        args,
        vae,
        vae_kwargs,
        text_encoder,
        model,
        text_encoder_2=None,
        pipeline=None,
        use_cpu_offload=True,
        device=0,
        logger=None,
    ):
        super().__init__(
            args,
            vae,
            vae_kwargs,
            text_encoder,
            model,
            text_encoder_2=text_encoder_2,
            pipeline=pipeline,
            use_cpu_offload=use_cpu_offload,
            device=device,
            logger=logger,
        )

        self.pipeline = self.load_diffusion_pipeline(
            args=args,
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            model=self.model,
            device=self.device,
        )

        self.default_negative_prompt = NEGATIVE_PROMPT

    def load_diffusion_pipeline(
        self,
        args,
        vae,
        text_encoder,
        text_encoder_2,
        model,
        scheduler=None,
        device=None,
        progress_bar_config=None,
        data_type="video",
    ):
        """Load the denoising scheduler for inference."""
        if scheduler is None:
            if args.denoise_type == "flow":
                scheduler = FlowMatchDiscreteScheduler(
                    shift=args.flow_shift,
                    reverse=args.flow_reverse,
                    solver=args.flow_solver,
                )
            else:
                raise ValueError(f"Invalid denoise type {args.denoise_type}")

        pipeline = HunyuanVideoPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            transformer=model,
            scheduler=scheduler,
            progress_bar_config=progress_bar_config,
            args=args,
        )
        if self.use_cpu_offload:
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline = pipeline.to(device)

        return pipeline

    def get_rotary_pos_embed(self, video_length, height, width):
        target_ndim = 3
        ndim = 5 - 2
        # 884
        if "884" in self.args.vae:
            latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]
        elif "888" in self.args.vae:
            latents_size = [(video_length - 1) // 8 + 1, height // 8, width // 8]
        else:
            latents_size = [video_length, height // 8, width // 8]

        if isinstance(self.model.patch_size, int):
            assert all(s % self.model.patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // self.model.patch_size for s in latents_size]
        elif isinstance(self.model.patch_size, list):
            assert all(
                s % self.model.patch_size[idx] == 0
                for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [
                s // self.model.patch_size[idx] for idx, s in enumerate(latents_size)
            ]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        head_dim = self.model.hidden_size // self.model.heads_num
        rope_dim_list = self.model.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.args.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

    @torch.no_grad()
    def predict(
        self,
        prompt,
        height=192,
        width=336,
        video_length=129,
        seed=None,
        negative_prompt=None,
        infer_steps=50,
        guidance_scale=6,
        flow_shift=5.0,
        embedded_guidance_scale=None,
        batch_size=1,
        num_videos_per_prompt=1,
        **kwargs,
    ):
        """
        Predict the image/video from the given text.

        Args:
            prompt (str or List[str]): The input text.
            kwargs:
                height (int): The height of the output video. Default is 192.
                width (int): The width of the output video. Default is 336.
                video_length (int): The frame number of the output video. Default is 129.
                seed (int or List[str]): The random seed for the generation. Default is a random integer.
                negative_prompt (str or List[str]): The negative text prompt. Default is an empty string.
                guidance_scale (float): The guidance scale for the generation. Default is 6.0.
                num_images_per_prompt (int): The number of images per prompt. Default is 1.
                infer_steps (int): The number of inference steps. Default is 100.
        """
        out_dict = dict()

        # ========================================================================
        # Arguments: seed
        # ========================================================================
        if isinstance(seed, torch.Tensor):
            seed = seed.tolist()
        if seed is None:
            seeds = [
                random.randint(0, 1_000_000)
                for _ in range(batch_size * num_videos_per_prompt)
            ]
        elif isinstance(seed, int):
            seeds = [
                seed + i
                for _ in range(batch_size)
                for i in range(num_videos_per_prompt)
            ]
        elif isinstance(seed, (list, tuple)):
            if len(seed) == batch_size:
                seeds = [
                    int(seed[i]) + j
                    for i in range(batch_size)
                    for j in range(num_videos_per_prompt)
                ]
            elif len(seed) == batch_size * num_videos_per_prompt:
                seeds = [int(s) for s in seed]
            else:
                raise ValueError(
                    f"Length of seed must be equal to number of prompt(batch_size) or "
                    f"batch_size * num_videos_per_prompt ({batch_size} * {num_videos_per_prompt}), got {seed}."
                )
        else:
            raise ValueError(
                f"Seed must be an integer, a list of integers, or None, got {seed}."
            )
        generator = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]
        out_dict["seeds"] = seeds

        # ========================================================================
        # Arguments: target_width, target_height, target_video_length
        # ========================================================================
        if width <= 0 or height <= 0 or video_length <= 0:
            raise ValueError(
                f"`height` and `width` and `video_length` must be positive integers, got height={height}, width={width}, video_length={video_length}"
            )
        if (video_length - 1) % 4 != 0:
            raise ValueError(
                f"`video_length-1` must be a multiple of 4, got {video_length}"
            )

        logger.info(
            f"Input (height, width, video_length) = ({height}, {width}, {video_length})"
        )

        target_height = align_to(height, 16)
        target_width = align_to(width, 16)
        target_video_length = video_length

        out_dict["size"] = (target_height, target_width, target_video_length)

        # ========================================================================
        # Arguments: prompt, new_prompt, negative_prompt
        # ========================================================================
        if not isinstance(prompt, str):
            raise TypeError(f"`prompt` must be a string, but got {type(prompt)}")
        prompt = [prompt.strip()]

        # negative prompt
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = self.default_negative_prompt
        if not isinstance(negative_prompt, str):
            raise TypeError(
                f"`negative_prompt` must be a string, but got {type(negative_prompt)}"
            )
        negative_prompt = [negative_prompt.strip()]

        # ========================================================================
        # Scheduler
        # ========================================================================
        scheduler = FlowMatchDiscreteScheduler(
            shift=flow_shift,
            reverse=self.args.flow_reverse,
            solver=self.args.flow_solver
        )
        self.pipeline.scheduler = scheduler

        # ========================================================================
        # Build Rope freqs
        # ========================================================================
        freqs_cos, freqs_sin = self.get_rotary_pos_embed(
            target_video_length, target_height, target_width
        )
        n_tokens = freqs_cos.shape[0]

        # ========================================================================
        # Print infer args
        # ========================================================================
        debug_str = f"""
                        height: {target_height}
                         width: {target_width}
                  video_length: {target_video_length}
                        prompt: {prompt}
                    neg_prompt: {negative_prompt}
                          seed: {seed}
                   infer_steps: {infer_steps}
         num_videos_per_prompt: {num_videos_per_prompt}
                guidance_scale: {guidance_scale}
                      n_tokens: {n_tokens}
                    flow_shift: {flow_shift}
       embedded_guidance_scale: {embedded_guidance_scale}"""
        logger.debug(debug_str)

        # ========================================================================
        # Pipeline inference
        # ========================================================================
        start_time = time.time()
        samples = self.pipeline(
            prompt=prompt,
            height=target_height,
            width=target_width,
            video_length=target_video_length,
            num_inference_steps=infer_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            output_type="pil",
            freqs_cis=(freqs_cos, freqs_sin),
            n_tokens=n_tokens,
            embedded_guidance_scale=embedded_guidance_scale,
            data_type="video" if target_video_length > 1 else "image",
            is_progress_bar=True,
            vae_ver=self.args.vae,
            enable_tiling=self.args.vae_tiling,
        )[0]
        out_dict["samples"] = samples
        out_dict["prompts"] = prompt

        gen_time = time.time() - start_time
        logger.info(f"Success, time: {gen_time}")

        return out_dict
