# sdcpp-cli-python

A Python wrapper for [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) CLI tool, providing a fluent, chainable API for image generation with Stable Diffusion models.

## Features

- **Fluent API**: Chainable method calls for intuitive configuration
- **Full CLI Support**: Complete coverage of stable-diffusion.cpp CLI options
- **Multiple Modes**: Support for image generation, video generation, upscaling, and model conversion
- **Memory Optimization**: Built-in methods for low-VRAM scenarios (CPU offloading, memory mapping)
- **Advanced Features**: ControlNet, LoRA, PHOTOMAKER, reference images, and more
- **Flexible Sampling**: Support for various sampling methods and schedulers

## Requirements

- Python 3.7+
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) binaries
- Compatible Stable Diffusion models (`.safetensors`, `.gguf`, `.bin`)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/g0g5/sdcpp-cli-python.git
cd sdcpp-cli-python
```

2. Download stable-diffusion.cpp binaries:
   - Visit the [stable-diffusion.cpp releases](https://github.com/leejet/stable-diffusion.cpp/releases)
   - Download the appropriate binaries for your platform
   - Extract them to the `bin/` directory

3. (Optional) Place your model files in the `models/` directory

## Integrating into Other Python Projects (pip + git)

If you want to pull this wrapper into another Python project without cloning the repo manually, you can install directly from Git:

```bash
python -m pip install "git+https://github.com/g0g5/sdcpp-cli-python.git"
```

To pin a specific ref (tag, branch, or commit):

```bash
python -m pip install "git+https://github.com/g0g5/sdcpp-cli-python.git@main"
```

Then import and point the wrapper at your local `sd-cli` binary location:

```python
from sdcli import SdCppCli

cli = SdCppCli(binary_path="./bin/sd-cli")
```

Notes:
- You still need compatible `stable-diffusion.cpp` binaries and model files available on disk.
- For deployment, consider pinning a commit hash in your `requirements.txt` for reproducible installs.

## Usage

### Basic Example

```python
from sdcli import SdCppCli

cli = (
    SdCppCli()
    .set_model("models/your_model.safetensors")
    .set_prompt("a beautiful landscape, mountains, sunset")
    .set_negative_prompt("ugly, low quality, worst quality")
    .set_width(1024)
    .set_height(1024)
    .set_steps(30)
    .set_cfg_scale(7.5)
    .set_sampling_method("euler_a")
    .set_scheduler("karras")
    .set_output("./output.png")
)

cli.run()
```

### Advanced Example with Memory Optimization

```python
from sdcli import SdCppCli

cli = (
    SdCppCli()
    .set_model("models/sdxl_model.safetensors")
    .set_prompt("a cinematic portrait, 35mm photography")
    .set_width(768)
    .set_height(512)
    .set_steps(20)
    .set_cfg_scale(5.0)
    .set_sampling_method("dpm++2m")
    .offload_to_cpu()  # Reduce VRAM usage
    .enable_mmap()     # Memory-map model weights
    .set_output("./portrait.png")
)

cli.run()
```

### Using ControlNet

```python
cli = (
    SdCppCli()
    .set_model("models/sd_model.safetensors")
    .set_control_net("models/controlnet_canny.safetensors")
    .set_control_image("./edge_map.png")
    .set_control_strength(0.9)
    .enable_canny()
    .set_prompt("a beautiful cityscape")
    .set_output("./output.png")
)
```

### Video Generation

```python
cli = (
    SdCppCli()
    .set_model("models/video_model.safetensors")
    .set_mode("vid_gen")
    .set_prompt("a flowing river")
    .set_video_frames(24)
    .set_fps(12)
    .set_output("./output_%03d.png")
    .set_output_begin_idx(0)
)

cli.run()
```

### Upscaling

```python
cli = (
    SdCppCli()
    .set_model("models/sd_model.safetensors")
    .set_upscale_model("models/esrgan.safetensors")
    .set_mode("upscale")
    .set_init_img("./input.png")
    .set_output("./upscaled.png")
)
```

## API Reference

### Core Methods

| Method | Description |
|--------|-------------|
| `set_model(path)` | Path to the Stable Diffusion model |
| `set_prompt(text)` | Text prompt for generation |
| `set_negative_prompt(text)` | Negative prompt to avoid certain features |
| `set_width(pixels)` | Image width |
| `set_height(pixels)` | Image height |
| `set_steps(n)` | Number of sampling steps |
| `set_cfg_scale(value)` | Classifier-free guidance scale |
| `set_sampling_method(method)` | Sampling method (euler, euler_a, dpm++2m, etc.) |
| `set_scheduler(method)` | Sigma scheduler (discrete, karras, exponential, etc.) |
| `set_seed(value)` | RNG seed (negative for random) |
| `set_output(path)` | Output file path |

### Memory Optimization

| Method | Description |
|--------|-------------|
| `offload_to_cpu()` | Offload weights to RAM, load to VRAM as needed |
| `enable_mmap()` | Memory-map model weights |
| `vae_on_cpu()` | Keep VAE on CPU (low VRAM) |
| `clip_on_cpu()` | Keep CLIP on CPU (low VRAM) |
| `control_net_on_cpu()` | Keep ControlNet on CPU (low VRAM) |

### Advanced Features

| Method | Description |
|--------|-------------|
| `set_control_net(path)` | Path to ControlNet model |
| `set_control_image(path)` | Control image for ControlNet |
| `set_lora_model_dir(path)` | Directory containing LoRA models |
| `set_embd_dir(path)` | Embeddings directory |
| `add_ref_image(path)` | Add reference image (can be called multiple times) |
| `set_mode(mode)` | Run mode: img_gen, vid_gen, upscale, convert |

### Run Methods

| Method | Description |
|--------|-------------|
| `run()` | Execute the CLI command |
| `build_args()` | Return the list of arguments without running |

## CLI Options

The wrapper supports all options from the underlying stable-diffusion.cpp CLI:

- **Context Options**: Model paths, encoder models, VAE, LoRA, embeddings
- **Generation Options**: Prompts, dimensions, steps, guidance, sampling
- **Memory Options**: CPU offloading, tiling, quantization
- **Preview Options**: Real-time preview, update intervals
- **Advanced Features**: ControlNet, PHOTOMAKER, reference images

For a complete list of available options, see [Official CLI manual](https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/cli/README.md) or run:
```bash
./bin/sd-cli --help
```

## Project Structure

```
sdcpp-cli-python/
├── sdcli.py              # Main wrapper class
├── main.py               # Example usage
├── bin/                  # stable-diffusion.cpp binaries
│   ├── sd-cli.exe
│   ├── sd-server.exe
│   └── stable-diffusion.dll
├── models/               # Model files directory
│   └── your_model.safetensors
└── README.md             # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) - The underlying CLI tool