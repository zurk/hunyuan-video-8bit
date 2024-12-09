# HunyuanVideo API, 8bit version

Checkout [Replicate API](https://replicate.com/zurk/hunyuan-video-8bit).

HunyuanVideo is a state-of-the-art text-to-video generation model that creates high-quality videos from text descriptions. 
This model outperforms many closed-source alternatives in text alignment, motion quality, and visual quality.

This is API with [8bit version of this model](https://huggingface.co/Kijai/HunyuanVideo_comfy/tree/main).
Runs on cheaper GPU and can be faster.

## Examples

```python
import replicate

output = replicate.run(
    "zurk/hunyuan-video-8bit:main",
    input={
        "prompt": "A cat walks on the grass, realistic style.",
        "negative_prompt": "Ugly",
        "width": 960,
        "height": 544,
        "video_length": 65,
        "embedded_guidance_scale": 6.0,
        "num_inference_steps": 40,
        "seed": 43,
    }
)
```

## Parameters

- prompt (string, required) - Text description of the video you want to generate
- negative_prompt (string, optional) - Text describing what you don't want in the video
- width (integer, default: 960) - Video width in pixels
- height (integer, default: 544) - Video height in pixels 
- video_length (integer, default: 65) - Number of frames (max 129)
- seed (integer, optional) - Random seed for reproducibility. Check logs if you did not set it and need to find out its value
- embedded_guidance_scale (float, default: 6.0) - Embedded guidance scale
- num_inference_steps (integer, default: 40) - Number of denoising steps
- flow_shift (float, default: 7.0) - Flow shift parameter for motion control


## Limitations

- Maximum video length is 129 frames (5.3 seconds)
- You need to provide video_length values 4*n+1: 17, 21, 25, etc.

## Feedback

If you have any issues running this API, create an issue on https://github.com/zurk/replicate-hunyuan-video-8bit/issues
I try to fix them as soon as possible.


For more details, visit the [HunyuanVideo GitHub repository](https://github.com/Tencent/HunyuanVideo)
and [ComfyUI wrapper nodes for HunyuanVideo](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper).
