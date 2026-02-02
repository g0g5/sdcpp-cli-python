import subprocess
from typing import Iterable, List, Optional, Sequence, Tuple


# Example usage:
#
# cli = (
#     SdCppCli()
#     .set_model("models/sd.safetensors")
#     .set_prompt("a cinematic portrait, 35mm")
#     .set_negative_prompt("blurry, low-res")
#     .set_width(768)
#     .set_height(512)
#     .set_steps(30)
#     .set_cfg_scale(7.5)
#     .set_output("./output.png")
# )
# cli.run()


class SdCppCli:
    def __init__(self, binary_path: str = "./bin/sd-cli") -> None:
        self.binary_path = binary_path
        self._args: List[Tuple[str, Optional[str]]] = []

    def _remove_flag(self, flag: str) -> None:
        self._args = [pair for pair in self._args if pair[0] != flag]

    def _set_value(
        self, flag: str, value: Optional[object], allow_multiple: bool = False
    ) -> "SdCppCli":
        if value is None:
            return self
        value_str = str(value)
        if allow_multiple:
            self._args.append((flag, value_str))
            return self
        self._remove_flag(flag)
        self._args.append((flag, value_str))
        return self

    def _set_list(self, flag: str, values: Optional[Iterable[object]]) -> "SdCppCli":
        if values is None:
            return self
        if isinstance(values, str):
            value_str = values
        else:
            value_str = ",".join(str(v) for v in values)
        return self._set_value(flag, value_str)

    def _set_flag(self, flag: str, enabled: bool = True) -> "SdCppCli":
        if enabled:
            self._remove_flag(flag)
            self._args.append((flag, None))
        else:
            self._remove_flag(flag)
        return self

    def build_args(self) -> List[str]:
        args: List[str] = [self.binary_path]
        for flag, value in self._args:
            args.append(flag)
            if value is not None:
                args.append(value)
        return args

    def run(
        self, check: bool = True, cwd: Optional[str] = None
    ) -> subprocess.CompletedProcess:
        return subprocess.run(self.build_args(), check=check, cwd=cwd)

    # CLI options
    def set_output(self, path: str) -> "SdCppCli":
        """Path to write result image to; supports printf-style %d."""
        return self._set_value("--output", path)

    def set_output_begin_idx(self, index: int) -> "SdCppCli":
        """Starting index for output image sequence; must be non-negative."""
        return self._set_value("--output-begin-idx", index)

    def set_preview_path(self, path: str) -> "SdCppCli":
        """Path to write preview image to."""
        return self._set_value("--preview-path", path)

    def set_preview_interval(self, interval: int) -> "SdCppCli":
        """Preview update interval in denoising steps."""
        return self._set_value("--preview-interval", interval)

    def enable_canny(self, enabled: bool = True) -> "SdCppCli":
        """Apply canny preprocessor (edge detection)."""
        return self._set_flag("--canny", enabled)

    def enable_convert_name(self, enabled: bool = True) -> "SdCppCli":
        """Convert tensor name (for convert mode)."""
        return self._set_flag("--convert-name", enabled)

    def set_verbose(self, enabled: bool = True) -> "SdCppCli":
        """Print extra info."""
        return self._set_flag("--verbose", enabled)

    def set_color(self, enabled: bool = True) -> "SdCppCli":
        """Colorize logging tags by level."""
        return self._set_flag("--color", enabled)

    def set_taesd_preview_only(self, enabled: bool = True) -> "SdCppCli":
        """Use taesd only for previews, not final decode."""
        return self._set_flag("--taesd-preview-only", enabled)

    def set_preview_noisy(self, enabled: bool = True) -> "SdCppCli":
        """Preview noisy inputs instead of denoised outputs."""
        return self._set_flag("--preview-noisy", enabled)

    def set_mode(self, mode: str) -> "SdCppCli":
        """Run mode: one of [img_gen, vid_gen, upscale, convert]."""
        return self._set_value("--mode", mode)

    def set_preview(self, method: str) -> "SdCppCli":
        """Preview method: one of [none, proj, tae, vae]."""
        return self._set_value("--preview", method)

    # Context options
    def set_model(self, path: str) -> "SdCppCli":
        """Path to full model."""
        return self._set_value("--model", path)

    def set_clip_l(self, path: str) -> "SdCppCli":
        """Path to the clip-l text encoder."""
        return self._set_value("--clip_l", path)

    def set_clip_g(self, path: str) -> "SdCppCli":
        """Path to the clip-g text encoder."""
        return self._set_value("--clip_g", path)

    def set_clip_vision(self, path: str) -> "SdCppCli":
        """Path to the clip-vision encoder."""
        return self._set_value("--clip_vision", path)

    def set_t5xxl(self, path: str) -> "SdCppCli":
        """Path to the t5xxl text encoder."""
        return self._set_value("--t5xxl", path)

    def set_llm(self, path: str) -> "SdCppCli":
        """Path to the LLM text encoder."""
        return self._set_value("--llm", path)

    def set_llm_vision(self, path: str) -> "SdCppCli":
        """Path to the LLM vision encoder."""
        return self._set_value("--llm_vision", path)

    def set_diffusion_model(self, path: str) -> "SdCppCli":
        """Path to the standalone diffusion model."""
        return self._set_value("--diffusion-model", path)

    def set_high_noise_diffusion_model(self, path: str) -> "SdCppCli":
        """Path to the standalone high-noise diffusion model."""
        return self._set_value("--high-noise-diffusion-model", path)

    def set_vae(self, path: str) -> "SdCppCli":
        """Path to standalone VAE model."""
        return self._set_value("--vae", path)

    def set_taesd(self, path: str) -> "SdCppCli":
        """Path to taesd (Tiny AutoEncoder) for fast decode."""
        return self._set_value("--taesd", path)

    def set_control_net(self, path: str) -> "SdCppCli":
        """Path to control net model."""
        return self._set_value("--control-net", path)

    def set_embd_dir(self, path: str) -> "SdCppCli":
        """Embeddings directory."""
        return self._set_value("--embd-dir", path)

    def set_lora_model_dir(self, path: str) -> "SdCppCli":
        """LoRA model directory."""
        return self._set_value("--lora-model-dir", path)

    def set_tensor_type_rules(self, rules: str) -> "SdCppCli":
        """Weight type per tensor pattern."""
        return self._set_value("--tensor-type-rules", rules)

    def set_photo_maker(self, path: str) -> "SdCppCli":
        """Path to PHOTOMAKER model."""
        return self._set_value("--photo-maker", path)

    def set_upscale_model(self, path: str) -> "SdCppCli":
        """Path to ESRGAN model."""
        return self._set_value("--upscale-model", path)

    def set_threads(self, threads: int) -> "SdCppCli":
        """Number of threads to use (<=0 means auto)."""
        return self._set_value("--threads", threads)

    def set_chroma_t5_mask_pad(self, value: int) -> "SdCppCli":
        """T5 mask pad size for chroma."""
        return self._set_value("--chroma-t5-mask-pad", value)

    def set_vae_tile_overlap(self, value: float) -> "SdCppCli":
        """VAE tile overlap fraction (default 0.5)."""
        return self._set_value("--vae-tile-overlap", value)

    def set_flow_shift(self, value: float) -> "SdCppCli":
        """Shift value for Flow models (SD3.x/WAN)."""
        return self._set_value("--flow-shift", value)

    def enable_vae_tiling(self, enabled: bool = True) -> "SdCppCli":
        """Process VAE in tiles to reduce memory usage."""
        return self._set_flag("--vae-tiling", enabled)

    def force_sdxl_vae_conv_scale(self, enabled: bool = True) -> "SdCppCli":
        """Force use of conv scale on SDXL VAE."""
        return self._set_flag("--force-sdxl-vae-conv-scale", enabled)

    def offload_to_cpu(self, enabled: bool = True) -> "SdCppCli":
        """Offload weights to RAM and load into VRAM when needed."""
        return self._set_flag("--offload-to-cpu", enabled)

    def enable_mmap(self, enabled: bool = True) -> "SdCppCli":
        """Memory-map model weights."""
        return self._set_flag("--mmap", enabled)

    def control_net_on_cpu(self, enabled: bool = True) -> "SdCppCli":
        """Keep control net on CPU (low VRAM)."""
        return self._set_flag("--control-net-cpu", enabled)

    def clip_on_cpu(self, enabled: bool = True) -> "SdCppCli":
        """Keep CLIP on CPU (low VRAM)."""
        return self._set_flag("--clip-on-cpu", enabled)

    def vae_on_cpu(self, enabled: bool = True) -> "SdCppCli":
        """Keep VAE on CPU (low VRAM)."""
        return self._set_flag("--vae-on-cpu", enabled)

    def enable_diffusion_fa(self, enabled: bool = True) -> "SdCppCli":
        """Use flash attention in the diffusion model."""
        return self._set_flag("--diffusion-fa", enabled)

    def enable_diffusion_conv_direct(self, enabled: bool = True) -> "SdCppCli":
        """Use ggml_conv2d_direct in diffusion model."""
        return self._set_flag("--diffusion-conv-direct", enabled)

    def enable_vae_conv_direct(self, enabled: bool = True) -> "SdCppCli":
        """Use ggml_conv2d_direct in VAE model."""
        return self._set_flag("--vae-conv-direct", enabled)

    def enable_circular(self, enabled: bool = True) -> "SdCppCli":
        """Enable circular padding for convolutions."""
        return self._set_flag("--circular", enabled)

    def enable_circularx(self, enabled: bool = True) -> "SdCppCli":
        """Enable circular RoPE wrapping on x-axis only."""
        return self._set_flag("--circularx", enabled)

    def enable_circulary(self, enabled: bool = True) -> "SdCppCli":
        """Enable circular RoPE wrapping on y-axis only."""
        return self._set_flag("--circulary", enabled)

    def disable_chroma_dit_mask(self, enabled: bool = True) -> "SdCppCli":
        """Disable DiT mask for chroma."""
        return self._set_flag("--chroma-disable-dit-mask", enabled)

    def enable_chroma_t5_mask(self, enabled: bool = True) -> "SdCppCli":
        """Enable T5 mask for chroma."""
        return self._set_flag("--chroma-enable-t5-mask", enabled)

    def set_type(self, value: str) -> "SdCppCli":
        """Weight type (e.g., f16, q4_0, q8_0)."""
        return self._set_value("--type", value)

    def set_rng(self, value: str) -> "SdCppCli":
        """RNG backend: one of [std_default, cuda, cpu]."""
        return self._set_value("--rng", value)

    def set_sampler_rng(self, value: str) -> "SdCppCli":
        """Sampler RNG backend; defaults to --rng."""
        return self._set_value("--sampler-rng", value)

    def set_prediction(self, value: str) -> "SdCppCli":
        """Prediction type override (eps, v, edm_v, sd3_flow, flux_flow, flux2_flow)."""
        return self._set_value("--prediction", value)

    def set_lora_apply_mode(self, value: str) -> "SdCppCli":
        """LoRA apply mode: one of [auto, immediately, at_runtime]."""
        return self._set_value("--lora-apply-mode", value)

    def set_vae_tile_size(self, value: str) -> "SdCppCli":
        """Tile size for VAE tiling, format [X]x[Y]."""
        return self._set_value("--vae-tile-size", value)

    def set_vae_relative_tile_size(self, value: str) -> "SdCppCli":
        """Relative VAE tile size, format [X]x[Y]."""
        return self._set_value("--vae-relative-tile-size", value)

    # Generation options
    def set_prompt(self, prompt: str) -> "SdCppCli":
        """Prompt to render."""
        return self._set_value("--prompt", prompt)

    def set_negative_prompt(self, prompt: str) -> "SdCppCli":
        """Negative prompt."""
        return self._set_value("--negative-prompt", prompt)

    def set_init_img(self, path: str) -> "SdCppCli":
        """Path to init image."""
        return self._set_value("--init-img", path)

    def set_end_img(self, path: str) -> "SdCppCli":
        """Path to end image (required by flf2v)."""
        return self._set_value("--end-img", path)

    def set_mask(self, path: str) -> "SdCppCli":
        """Path to mask image."""
        return self._set_value("--mask", path)

    def set_control_image(self, path: str) -> "SdCppCli":
        """Path to control image for control net."""
        return self._set_value("--control-image", path)

    def set_control_video(self, path: str) -> "SdCppCli":
        """Path to control video frames directory (lexicographical order)."""
        return self._set_value("--control-video", path)

    def set_pm_id_images_dir(self, path: str) -> "SdCppCli":
        """Path to PHOTOMAKER input ID images directory."""
        return self._set_value("--pm-id-images-dir", path)

    def set_pm_id_embed_path(self, path: str) -> "SdCppCli":
        """Path to PHOTOMAKER v2 ID embed."""
        return self._set_value("--pm-id-embed-path", path)

    def set_height(self, value: int) -> "SdCppCli":
        """Image height in pixels."""
        return self._set_value("--height", value)

    def set_width(self, value: int) -> "SdCppCli":
        """Image width in pixels."""
        return self._set_value("--width", value)

    def set_steps(self, value: int) -> "SdCppCli":
        """Number of sampling steps."""
        return self._set_value("--steps", value)

    def set_high_noise_steps(self, value: int) -> "SdCppCli":
        """High-noise sampling steps (-1 for auto)."""
        return self._set_value("--high-noise-steps", value)

    def set_clip_skip(self, value: int) -> "SdCppCli":
        """Ignore last layers of CLIP network."""
        return self._set_value("--clip-skip", value)

    def set_batch_count(self, value: int) -> "SdCppCli":
        """Batch count."""
        return self._set_value("--batch-count", value)

    def set_video_frames(self, value: int) -> "SdCppCli":
        """Number of video frames."""
        return self._set_value("--video-frames", value)

    def set_fps(self, value: int) -> "SdCppCli":
        """Frames per second."""
        return self._set_value("--fps", value)

    def set_timestep_shift(self, value: int) -> "SdCppCli":
        """Shift timestep for NitroFusion models."""
        return self._set_value("--timestep-shift", value)

    def set_upscale_repeats(self, value: int) -> "SdCppCli":
        """Run ESRGAN upscaler this many times."""
        return self._set_value("--upscale-repeats", value)

    def set_upscale_tile_size(self, value: int) -> "SdCppCli":
        """Tile size for ESRGAN upscaling."""
        return self._set_value("--upscale-tile-size", value)

    def set_cfg_scale(self, value: float) -> "SdCppCli":
        """Unconditional guidance scale."""
        return self._set_value("--cfg-scale", value)

    def set_img_cfg_scale(self, value: float) -> "SdCppCli":
        """Image guidance scale for inpaint or instruct-pix2pix."""
        return self._set_value("--img-cfg-scale", value)

    def set_guidance(self, value: float) -> "SdCppCli":
        """Distilled guidance scale for models with guidance input."""
        return self._set_value("--guidance", value)

    def set_slg_scale(self, value: float) -> "SdCppCli":
        """Skip layer guidance (SLG) scale for DiT models."""
        return self._set_value("--slg-scale", value)

    def set_skip_layer_start(self, value: float) -> "SdCppCli":
        """SLG enabling point."""
        return self._set_value("--skip-layer-start", value)

    def set_skip_layer_end(self, value: float) -> "SdCppCli":
        """SLG disabling point."""
        return self._set_value("--skip-layer-end", value)

    def set_eta(self, value: float) -> "SdCppCli":
        """Eta for DDIM/TCD samplers."""
        return self._set_value("--eta", value)

    def set_high_noise_cfg_scale(self, value: float) -> "SdCppCli":
        """High-noise unconditional guidance scale."""
        return self._set_value("--high-noise-cfg-scale", value)

    def set_high_noise_img_cfg_scale(self, value: float) -> "SdCppCli":
        """High-noise image guidance scale."""
        return self._set_value("--high-noise-img-cfg-scale", value)

    def set_high_noise_guidance(self, value: float) -> "SdCppCli":
        """High-noise distilled guidance scale."""
        return self._set_value("--high-noise-guidance", value)

    def set_high_noise_slg_scale(self, value: float) -> "SdCppCli":
        """High-noise SLG scale for DiT models."""
        return self._set_value("--high-noise-slg-scale", value)

    def set_high_noise_skip_layer_start(self, value: float) -> "SdCppCli":
        """High-noise SLG enabling point."""
        return self._set_value("--high-noise-skip-layer-start", value)

    def set_high_noise_skip_layer_end(self, value: float) -> "SdCppCli":
        """High-noise SLG disabling point."""
        return self._set_value("--high-noise-skip-layer-end", value)

    def set_high_noise_eta(self, value: float) -> "SdCppCli":
        """High-noise eta for DDIM/TCD samplers."""
        return self._set_value("--high-noise-eta", value)

    def set_strength(self, value: float) -> "SdCppCli":
        """Strength for noising/unnoising."""
        return self._set_value("--strength", value)

    def set_pm_style_strength(self, value: float) -> "SdCppCli":
        """PHOTOMAKER style strength."""
        return self._set_value("--pm-style-strength", value)

    def set_control_strength(self, value: float) -> "SdCppCli":
        """Control Net strength (1.0 destroys init image info)."""
        return self._set_value("--control-strength", value)

    def set_moe_boundary(self, value: float) -> "SdCppCli":
        """Timestep boundary for Wan2.2 MoE (requires auto high-noise)."""
        return self._set_value("--moe-boundary", value)

    def set_vace_strength(self, value: float) -> "SdCppCli":
        """Wan VACE strength."""
        return self._set_value("--vace-strength", value)

    def increase_ref_index(self, enabled: bool = True) -> "SdCppCli":
        """Auto-increase reference image indices based on order."""
        return self._set_flag("--increase-ref-index", enabled)

    def disable_auto_resize_ref_image(self, enabled: bool = True) -> "SdCppCli":
        """Disable auto resize of reference images."""
        return self._set_flag("--disable-auto-resize-ref-image", enabled)

    def set_seed(self, value: int) -> "SdCppCli":
        """RNG seed; use < 0 for random."""
        return self._set_value("--seed", value)

    def set_sampling_method(self, value: str) -> "SdCppCli":
        """Sampling method (euler, euler_a, dpm++2m, etc.)."""
        return self._set_value("--sampling-method", value)

    def set_high_noise_sampling_method(self, value: str) -> "SdCppCli":
        """High-noise sampling method."""
        return self._set_value("--high-noise-sampling-method", value)

    def set_scheduler(self, value: str) -> "SdCppCli":
        """Denoiser sigma scheduler (discrete, karras, etc.)."""
        return self._set_value("--scheduler", value)

    def set_sigmas(self, value: Sequence[float] | str) -> "SdCppCli":
        """Custom sigma values for the sampler (comma-separated)."""
        if isinstance(value, str):
            return self._set_value("--sigmas", value)
        return self._set_list("--sigmas", value)

    def set_skip_layers(self, layers: Sequence[int] | str) -> "SdCppCli":
        """Layers to skip for SLG steps."""
        if isinstance(layers, str):
            return self._set_value("--skip-layers", layers)
        return self._set_list("--skip-layers", layers)

    def set_high_noise_skip_layers(self, layers: Sequence[int] | str) -> "SdCppCli":
        """High-noise layers to skip for SLG steps."""
        if isinstance(layers, str):
            return self._set_value("--high-noise-skip-layers", layers)
        return self._set_list("--high-noise-skip-layers", layers)

    def add_ref_image(self, path: str) -> "SdCppCli":
        """Reference image for Flux Kontext models (repeatable)."""
        return self._set_value("--ref-image", path, allow_multiple=True)

    def set_cache_mode(self, value: str) -> "SdCppCli":
        """Caching method (easycache, ucache, dbcache, taylorseer, cache-dit)."""
        return self._set_value("--cache-mode", value)

    def set_cache_option(self, value: str) -> "SdCppCli":
        """Named cache params (key=value, comma-separated)."""
        return self._set_value("--cache-option", value)

    def set_cache_preset(self, value: str) -> "SdCppCli":
        """Cache-dit preset: slow/s, medium/m, fast/f, ultra/u."""
        return self._set_value("--cache-preset", value)

    def set_scm_mask(self, value: str) -> "SdCppCli":
        """SCM steps mask for cache-dit (comma-separated 0/1)."""
        return self._set_value("--scm-mask", value)

    def set_scm_policy(self, value: str) -> "SdCppCli":
        """SCM policy: dynamic (default) or static."""
        return self._set_value("--scm-policy", value)
