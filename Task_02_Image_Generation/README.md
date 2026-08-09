# Task-02: Image Generation with Pre-trained Models

## Internship Task
Generate images from text prompts using a pre-trained generative AI model.

## Model Used
This project uses `segmind/tiny-sd`, a distilled Stable Diffusion text-to-image model available through Hugging Face Diffusers.

## Features
- Text prompt input
- Pre-trained Stable Diffusion model
- Text-to-image generation
- Random seed support for reproducible results
- CPU/GPU automatic device selection
- Simple Gradio web interface
- Generated images saved in the `outputs` folder

## Project Structure

```text
Task_02_Image_Generation/
│
├── app.py
├── generate_image.py
├── requirements.txt
├── README.md
├── .gitignore
└── outputs/
```

## Recommended Python Version
Python 3.11 is recommended for this project.

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the command-line version

```bash
python generate_image.py
```

Enter a prompt such as:

```text
A futuristic city at sunset, cinematic lighting, highly detailed
```

The generated image is saved in:

```text
outputs/generated_image.png
```

## Run the Web UI

```bash
python app.py
```

Open the local Gradio URL shown in the terminal.

## Example Prompts

1. `A futuristic city at sunset, cinematic lighting, highly detailed`
2. `A cute robot exploring a magical forest, fantasy art`
3. `A peaceful mountain landscape with a lake, realistic photography`
4. `An astronaut standing on Mars, dramatic lighting, digital art`

## CPU Note

The project can run on CPU, but image generation may be slow. A CUDA-compatible NVIDIA GPU is strongly preferred for faster generation.

## Output

Save screenshots of:
- Terminal showing successful execution
- Gradio interface
- Generated images
- Project folder / GitHub repository

These screenshots can be included in the internship submission and LinkedIn post.

## What I Learned

- Basics of text-to-image generative AI
- Stable Diffusion and diffusion-based image generation
- Loading pre-trained models with Hugging Face Diffusers
- Converting text prompts into images
- Prompt engineering
- Controlling generation with inference steps and random seeds
- Using PyTorch for model inference
- Building a simple AI interface with Gradio
- Saving and documenting generated outputs
