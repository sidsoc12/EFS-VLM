import modal
from modal import Image

# Define the Modal app
app = modal.App("open-flamingo-finetuning")

# Define the image with dependencies from environment.yml
image = (
    Image.debian_slim(python_version="3.9")
    .pip_install_from_requirements("combined_requirements.txt")
    .copy_local_dir(".", "/root")
)


# Define the function to run your training script
@app.function(
    image=image,
    gpu="A100",
    memory="128GiB",
    secrets=[modal.Secret.from_name("wandb")],
)
def run_model():
    import os
    import subprocess

    # Set environment variables if needed
    os.environ["WANDB_API_KEY"] = "f3ac954df2d182db0dade02a382a0eb63290be6d"
    wandb_project = "Mammo"
    wandb_entity = "Mammo"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
        "gs://emory-dataset/train/shard-{0..11}.tar",
        "--mmc4_shards",
        "gs://emory-dataset/train/shard-{0..11}.tar",
        "--report_to_wandb",
        "--wandb_project",
        wandb_project,
        "--wandb_entity",
        wandb_entity,
        "--freeze_lm_embeddings",  # Add this flag to freeze base model layers
    ]

    # Run the training script
    subprocess.run(command, check=True)
