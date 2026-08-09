import os
import torch
from diffusers import DiffusionPipeline

MODEL_ID = "segmind/tiny-sd"
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "generated_image.png")


def load_pipeline():
    """Load the pre-trained text-to-image model."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Using device: {device}")

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype
    )
    pipe = pipe.to(device)

    # Helps reduce memory usage on supported versions.
    if device == "cpu":
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    return pipe, device


def generate_image(prompt, seed=None, steps=25):
    pipe, device = load_pipeline()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if seed is None:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    generator = torch.Generator(device=device).manual_seed(seed)

    print("Generating image...")
    print(f"Prompt: {prompt}")
    print(f"Seed: {seed}")
    print(f"Inference steps: {steps}")

    result = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        generator=generator
    )

    image = result.images[0]
    image.save(OUTPUT_FILE)

    print(f"Image saved to: {OUTPUT_FILE}")
    return image


if __name__ == "__main__":
    print("=" * 60)
    print("TASK-02: TEXT-TO-IMAGE GENERATION")
    print("=" * 60)

    prompt = input(
        "Enter a text prompt: "
    ).strip()

    if not prompt:
        prompt = (
            "A futuristic city at sunset, cinematic lighting, "
            "highly detailed"
        )

    generate_image(prompt)
