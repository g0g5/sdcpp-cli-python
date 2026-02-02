from sdcli import SdCppCli


def main() -> None:
    cli = (
        SdCppCli()
        .set_model("models/oneObsession_v19.safetensors")
        .set_seed(456456456456)
        .set_prompt("realistic, photo, a lovely dog in room")
        .set_negative_prompt("ugly, low quality, worst quality")
        .set_width(1024)
        .set_height(1024)
        .set_steps(30)
        .set_cfg_scale(4.5)
        .set_sampling_method("euler_a")
        .set_scheduler("karras")
        .offload_to_cpu()
        .enable_diffusion_fa()
        .enable_vae_conv_direct()
        .set_output("./output.png")
    )
    cli.run()


if __name__ == "__main__":
    main()
