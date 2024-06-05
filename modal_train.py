import modal
from modal import Image
import torch

# Define the Modal app
app = modal.App("open-flamingo-finetuning")

# Define the image with dependencies from environment.yml
image = (
    Image.debian_slim(python_version="3.9")
    .pip_install_from_requirements("combined_requirements.txt")
    .pip_install("google-cloud-storage")
    .run_commands(
        "apt-get update -y",
        "apt-get install -y wget unzip",
        "wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-405.0.0-linux-x86_64.tar.gz",
        "tar -xvzf google-cloud-sdk-405.0.0-linux-x86_64.tar.gz",
        "mv google-cloud-sdk /root/",
        "/root/google-cloud-sdk/install.sh --quiet",
    )
    .env({"PATH": "/root/google-cloud-sdk/bin:$PATH"})
    .copy_local_dir(".", "/root")
)


# Define the function to run your training script
@app.function(
    image=image,
    gpu="A100:4",
    memory="128GiB",
    timeout=86400,
    secrets=[modal.Secret.from_name("wandb"), modal.Secret.from_name("mammo-secret")],
)
def run_model():
    import os
    import subprocess

    print(f"Available GPUs: {torch.cuda.device_count()}")
    # Print names of available GPUs

    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Set environment variables if needed
    os.environ["WANDB_API_KEY"] = "f3ac954df2d182db0dade02a382a0eb63290be6d"

    # Authenticate with GCP using the service account key
    # Authenticate with GCP using the service account key

    # Authenticate with GCP using the service account key from cloud-key.json
    gcloud_key_path = "/root/gcloud-key.json"

    subprocess.run(
        ["gcloud", "auth", "activate-service-account", "--key-file", gcloud_key_path],
        check=True,
    )
    subprocess.run(
        ["gcloud", "config", "set", "project", "our-service-423520-a9"], check=True
    )

    subprocess.run(["wandb", "login", os.environ["WANDB_API_KEY"]], check=True)

    wandb_project = "Mammo"
    wandb_entity = "cs231n-vlm"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

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
