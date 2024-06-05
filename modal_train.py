import modal

# Define the Modal app
stub = modal.Stub("open-flamingo-finetuning")

# Define the image with dependencies from environment.yml
image = modal.Image.conda().from_yaml("environment.yml")


# Define the function to run your training script
@stub.function(
    image=image,
    gpu="A100",
    memory="64GB",
    secret=modal.Secret.from_name("your-wandb-secret"),
)
def train_model():
    import os
    import subprocess

    # Set environment variables if needed
    os.environ["WANDB_API_KEY"] = "f3ac954df2d182db0dade02a382a0eb63290be6d"

    # Command to run the training script with the appropriate arguments
    command = [
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=4",
        "open_flamingo/train/train.py",
        "--lm_path",
        "anas-awadalla/mpt-1b-redpajama-200b",
        "--tokenizer_path",
        "anas-awadalla/mpt-1b-redpajama-200b",
        "--cross_attn_every_n_layers",
        "1",
        "--dataset_resampled",
        "--batch_size_mmc4",
        "32",
        "--batch_size_laion",
        "64",
        "--train_num_samples_mmc4",
        "125000",
        "--train_num_samples_laion",
        "250000",
        "--loss_multiplier_laion",
        "0.2",
        "--workers",
        "4",
        "--run_name",
        "OpenFlamingo-3B-vitl-mpt1b",
        "--num_epochs",
        "480",
        "--warmup_steps",
        "1875",
        "--mmc4_textsim_threshold",
        "0.24",
        "--laion_shards",
        "/path/to/shards/shard-{0..11}.tar",
        "--mmc4_shards",
        "/path/to/shards/shard-{0..11}.tar",
        "--report_to_wandb",
        "--freeze_base_model",  # Add this flag to freeze base model layers
    ]

    # Run the training script
    subprocess.run(command, check=True)


# Run the function in the Modal environment
if __name__ == "__main__":
    with stub.run():
        train_model()
