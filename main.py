import os
import sys

# Add the src directory to the Python path if it's not already there
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import cv2 as cv
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers.utils import load_image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
# from streamdiffusion.acceleration.sfast import accelerate_with_stable_fast

# You can load any models using diffuser's StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7").to(
    device=torch.device("cuda"),
    dtype=torch.float16,
)
# pipe.load_lora_weights('FirstLast/RealisticVision-LoRA-libr-0.2', weight_name='pytorch_lora_weights.safetensors')
# Wrap the pipeline in StreamDiffusion
stream = StreamDiffusion(
    pipe,
    t_index_list=[ 16, 38 ],
    torch_dtype=torch.float16,
    cfg_type="self",
)

# If the loaded model is not LCM, merge LCM
#stream.load_lcm_lora("latent-consistency/lcm-lora-sdv1-5")
# stream.fuse_lora()

# Use Tiny VAE for further acceleration
stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)

############# ACCELERATE #############

# Enable acceleration with xformers memory efficient attention
# pipe.enable_xformers_memory_efficient_attention()

stream = accelerate_with_tensorrt(
    stream, "engines", max_batch_size=2,
)

# Uncomment the following line to use StableFast
# stream = accelerate_with_stable_fast(stream)

###################################################################
prompt = "cat, 8k, digital art"
#prompt = "anime dress, 8k, oil painting"
# Prepare the stream
stream.prepare(prompt=prompt,
               negative_prompt="anime, handdrawn, pencil, manga",
            num_inference_steps=50,
            guidance_scale=1.1)

stream.update_prompts(weighted_prompts=[("cat, 8k, digital art", 0.5), ("cat, pixel art, picasso, cubism, strandinsky, monochrome", 0.5)])
# Prepare image
init_image = load_image("http://picsum.photos/512").resize((512, 512))

# Warmup >= len(t_index_list) x frame_buffer_size
for _ in range(2):
    stream(init_image)

x_output = stream(init_image)
image = postprocess_image(x_output)
dir(image)
dir(image[0])
image[0].save("/tmp/foo.png")
