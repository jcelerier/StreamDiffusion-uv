import os
import sys
from typing import List, Dict, Literal, Optional
from diffusers.utils import load_image

import fire


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# thibaud/controlnet-sd21-canny-diffusers

# lllyasviel/control_v11p_sd15_canny

def main(
    input: str = os.path.join(CURRENT_DIR, "..", "..", "images", "inputs", "input.png"),
    output: str = os.path.join(CURRENT_DIR, "..", "..", "images", "outputs", "output.png"),
    model_id_or_path: str = "runwayml/stable-diffusion-v1-5", #"IDKiro/sdxs-512-0.9", #"runwayml/stable-diffusion-v1-5", #"stabilityai/sd-turbo",
    lora_dict: Optional[Dict[str, float]] = None,
    controlnet_dicts: Optional[List[Dict[str, float]]] = None, #[{"lllyasviel/control_v11p_sd15_canny" : 0.75}], #None,
    prompt: str = "1girl with brown dog hair, thick glasses, smiling",
    negative_prompt: str = "low quality, bad quality, blurry, low resolution",
    t_index_list: List[int] = [32, 45], #[22, 32, 45] # TRT will need to re-compile plus there is FPS difference based on these steps
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
    CM_lora_type: Literal["lcm", "Hyper_SD", "none"] = "none",
    use_tiny_vae: bool = True,
    use_denoising_batch: bool = True,
    guidance_scale: float = 1.2,
    cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    seed: int = 2,
    delta: float = 0.5,
):
    """
    Initializes the StreamDiffusionWrapper.

    Parameters
    ----------
    input : str, optional
        The input image file to load images from.
    output : str, optional
        The output image file to save images to.
    model_id_or_path : str
        The model id or path to load.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    negative_prompt : str, optional
        The negative prompt to use.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    acceleration : Literal["none", "xformers", "tensorrt"], optional
        The acceleration method, by default "tensorrt".
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default True.
    guidance_scale : float, optional
        The CFG scale, by default 1.2.
    cfg_type : Literal["none", "full", "self", "initialize"],
    optional
        The cfg_type for img2img mode, by default "self".
        You cannot use anything other than "none" for txt2img mode.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    delta : float, optional
        The delta multiplier of virtual residual noise,
        by default 1.0.
    """

    if guidance_scale <= 1.0:
        cfg_type = "none"

    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=t_index_list,
        controlnet_dicts=controlnet_dicts,
        frame_buffer_size=1,
        width=width,
        height=height,
        acceleration=acceleration,
        mode="img2img",
        CM_lora_type=CM_lora_type, #"Hyper_SD",
        use_tiny_vae=use_tiny_vae,
        use_denoising_batch=use_denoising_batch,
        cfg_type=cfg_type,
        seed=seed,
    )

    stream.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=guidance_scale,
        delta=delta,
    )

    ###################################################################

    # Import time for measuring performance
    import time
    import numpy as np

    # Prepare image
    init_image = load_image("assets/img2img_example.png").resize((512, 512))
    # image_tensor = stream.preprocess_image(init_image)

    # Warmup >= len(t_index_list) x frame_buffer_size
    for _ in range(stream.batch_size - 1):
        stream(
            image=init_image,
            controlnet_images=init_image if controlnet_dicts else None,
        )

    # Number of iterations
    iterations = 1000
    times = []

    # Run the stream for specified iterations
    for i in range(iterations):
        start_time = time.time()
        x_output = stream(
            image=init_image,
            controlnet_images=init_image if controlnet_dicts else None,
        )
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        
        if (i + 1) % 100 == 0:
            print(f"Completed {i+1}/{iterations} iterations")

    # Calculate statistics
    times = np.array(times)
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    fps = 1.0 / avg_time

    # Print statistics
    print("\nPerformance Statistics:")
    print(f"Total images generated: {iterations}")
    print(f"Average time per image: {avg_time:.4f} seconds")
    print(f"Minimum time per image: {min_time:.4f} seconds")
    print(f"Maximum time per image: {max_time:.4f} seconds")
    print(f"Standard deviation: {std_time:.4f} seconds")
    print(f"Frames per second (FPS): {fps:.2f}")


if __name__ == "__main__":
    fire.Fire(main)
