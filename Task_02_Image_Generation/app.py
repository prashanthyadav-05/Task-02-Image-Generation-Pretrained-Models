import os
import torch
import gradio as gr
from diffusers import DiffusionPipeline

MODEL_ID = "segmind/tiny-sd"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"Loading model: {MODEL_ID}")
print(f"Using device: {DEVICE}")

pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE
)
pipe = pipe.to(DEVICE)

if DEVICE == "cpu":
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass


def generate(prompt, steps, seed):
    if not prompt or not prompt.strip():
        return None, "Please enter a text prompt."

    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = 42

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    result = pipe(
        prompt=prompt.strip(),
        num_inference_steps=int(steps),
        generator=generator
    )

    image = result.images[0]
    filename = os.path.join(
        OUTPUT_DIR,
        f"generated_{seed}.png"
    )
    image.save(filename)

    return image, f"Generated successfully. Seed: {seed}"


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(
            label="Text Prompt",
            placeholder="Describe the image you want to generate...",
            lines=3
        ),
        gr.Slider(
            minimum=10,
            maximum=40,
            value=25,
            step=1,
            label="Inference Steps"
        ),
        gr.Number(
            value=42,
            precision=0,
            label="Seed"
        )
    ],
    outputs=[
        gr.Image(label="Generated Image"),
        gr.Textbox(label="Status")
    ],
    title="Task-02: AI Image Generator",
    description=(
        "Generate an image from a text prompt using a "
        "pre-trained Stable Diffusion model."
    )
)

if __name__ == "__main__":
    demo.launch()
