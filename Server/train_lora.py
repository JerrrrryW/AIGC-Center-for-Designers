
# MODIFIED SCRIPT FOR PROGRAMMATIC EXECUTION
# Original script: https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py
# This script has been modified to be called as a Python function rather than from the command line.
# Key changes:
# - Replaced argparse with a TrainingConfig dataclass.
# - The main logic is wrapped in a `start_training` function.
# - Added a `status_updater` dictionary argument to report progress back to the main app.
# - Removed/disabled features not relevant for local, single-GPU (MPS) training.

import os
import gc
import math
import shutil
import logging
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import free_memory
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

check_min_version("0.36.0.dev0")

logger = get_logger(__name__)

@dataclass
class TrainingConfig:
    pretrained_model_name_or_path: str
    instance_data_dir: str
    instance_prompt: str
    output_dir: str
    user_model_name: Optional[str] = None
    max_train_steps: int = 1000
    learning_rate: float = 1e-4
    revision: Optional[str] = None
    variant: Optional[str] = None
    tokenizer_name: Optional[str] = None
    seed: Optional[int] = None
    resolution: int = 512
    center_crop: bool = False
    train_text_encoder: bool = False
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    dataloader_num_workers: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0
    logging_dir: str = "logs"
    report_to: str = "tensorboard"
    mixed_precision: str = "no"
    rank: int = 4
    lora_dropout: float = 0.0
    image_interpolation_mode: str = "lanczos"
    tokenizer_max_length: Optional[int] = None

class DreamBoothDataset(Dataset):
    # ... (omitting unchanged class for brevity)
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        size=512,
        center_crop=False,
        tokenizer_max_length=None,
        image_interpolation_mode="lanczos",
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exist.")

        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}
        self.instance_images_path = [
            img
            for img in Path(instance_data_root).iterdir()
            if img.is_file() and img.suffix.lower() in valid_exts
        ]
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        interpolation = getattr(transforms.InterpolationMode, image_interpolation_mode.upper(), None)
        if interpolation is None:
            raise ValueError(f"Unsupported interpolation mode {image_interpolation_mode}.")

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=interpolation),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        image_path = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(image_path)
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        base_prompt = self.instance_prompt.strip()
        caption_text = ""
        caption_path = image_path.with_suffix(".txt")
        if caption_path.is_file():
            try:
                caption_text = caption_path.read_text(encoding="utf-8").strip()
            except OSError:
                caption_text = ""

        if caption_text:
            caption_text = f"{base_prompt} {caption_text}".strip() if base_prompt else caption_text
        else:
            caption_text = base_prompt

        text_inputs = tokenize_prompt(
            self.tokenizer, caption_text, tokenizer_max_length=self.tokenizer_max_length
        )
        example["instance_prompt_ids"] = text_inputs.input_ids
        example["instance_attention_mask"] = text_inputs.attention_mask

        return example

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    # ... (omitting unchanged function for brevity)
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return text_inputs

def collate_fn(examples, with_prior_preservation=False):
    # ... (omitting unchanged function for brevity)
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    if has_attention_mask:
        batch["attention_mask"] = torch.cat(attention_mask, dim=0)

    return batch

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    # ... (omitting unchanged function for brevity)
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def start_training(config: TrainingConfig, status_updater: Optional[dict] = None):
    try:
        if status_updater:
            status_updater.update({"status": "initializing", "progress": 0, "message": "Initializing training..."})

        logging_dir = Path(config.output_dir, config.logging_dir)
        accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with=config.report_to,
            project_config=ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir),
        )

        if accelerator.device.type == "mps":
            accelerator.native_amp = False

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        if config.seed is not None:
            set_seed(config.seed)

        if accelerator.is_main_process:
            os.makedirs(config.output_dir, exist_ok=True)

        # ... (rest of the setup logic: tokenizer, models, etc.)
        if status_updater:
            status_updater.update({"status": "loading_models", "progress": 5, "message": "Loading models..."})
        
        # All the model loading logic from the original script goes here...
        if config.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, revision=config.revision, use_fast=False)
        elif config.pretrained_model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                config.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=config.revision,
                use_fast=False,
            )

        text_encoder_cls = import_model_class_from_model_name_or_path(config.pretrained_model_name_or_path, config.revision)
        noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")
        text_encoder = text_encoder_cls.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="text_encoder", revision=config.revision, variant=config.variant
        )
        vae = AutoencoderKL.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="vae", revision=config.revision, variant=config.variant
        )
        unet = UNet2DConditionModel.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="unet", revision=config.revision, variant=config.variant
        )

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        unet.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)

        if config.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        unet_lora_config = LoraConfig(
            r=config.rank,
            lora_alpha=config.rank,
            lora_dropout=config.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet.add_adapter(unet_lora_config)

        params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon,
        )

        train_dataset = DreamBoothDataset(
            instance_data_root=config.instance_data_dir,
            instance_prompt=config.instance_prompt,
            tokenizer=tokenizer,
            size=config.resolution,
            center_crop=config.center_crop,
            tokenizer_max_length=config.tokenizer_max_length,
            image_interpolation_mode=config.image_interpolation_mode,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, False),
            num_workers=config.dataloader_num_workers,
        )
        
        num_train_epochs = math.ceil(config.max_train_steps / (len(train_dataloader) / config.gradient_accumulation_steps))

        lr_scheduler = get_scheduler(
            config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps * config.gradient_accumulation_steps,
            num_training_steps=config.max_train_steps * config.gradient_accumulation_steps,
        )

        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

        logger.info("***** Running training *****")
        # ... (logging info)

        if status_updater:
            status_updater.update({"status": "training", "progress": 10, "message": "Starting training loop..."})

        global_step = 0
        terminated = False # Flag to indicate if training was stopped early

        progress_bar = tqdm(range(global_step, config.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for epoch in range(num_train_epochs):
            unet.train()
            for step, batch in enumerate(train_dataloader):
                # Check for termination signal
                if status_updater and status_updater.get("should_stop", False):
                    terminated = True
                    logger.warning("Termination signal received. Stopping training.")
                    if status_updater:
                        status_updater.update({"status": "failed", "progress": status_updater.get("progress", 0), "message": "Training was cancelled by the user."})
                    break # Exit the inner loop

                with accelerator.accumulate(unet):
                    # ... (core training step logic)
                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = model_input * vae.config.scaling_factor

                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                    timesteps = timesteps.long()

                    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                    
                    encoder_hidden_states = encode_prompt(
                        text_encoder,
                        batch["input_ids"],
                        batch["attention_mask"],
                    )

                    model_pred = unet(noisy_model_input, timesteps, encoder_hidden_states, return_dict=False)[0]

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_to_optimize, config.max_grad_norm)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if status_updater:
                        progress_percent = (global_step / config.max_train_steps) * 100
                        status_updater.update({
                            "progress": round(progress_percent, 2),
                            "message": f"Step {global_step}/{config.max_train_steps}"
                        })

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= config.max_train_steps:
                    break
            
            if terminated:
                break # Exit the outer loop

        accelerator.wait_for_everyone()

        # Save the model only if training completed successfully
        if accelerator.is_main_process and not terminated:
            unet = accelerator.unwrap_model(unet)
            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
            
            StableDiffusionLoraLoaderMixin.save_lora_weights(
                save_directory=config.output_dir,
                unet_lora_layers=unet_lora_state_dict,
                text_encoder_lora_layers=None, # Not training text encoder
            )
            logger.info(f"LoRA weights saved to {config.output_dir}")

            thumbnail_filename = None
            # Copy a training image into the model directory as a thumbnail
            try:
                valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}
                first_image_path = next(
                    path
                    for path in sorted(Path(config.instance_data_dir).iterdir())
                    if path.is_file() and path.suffix.lower() in valid_exts
                )
                thumbnail_filename = f"thumbnail{first_image_path.suffix.lower()}"
                thumbnail_path = Path(config.output_dir) / thumbnail_filename
                shutil.copyfile(first_image_path, thumbnail_path)
                logger.info(f"Thumbnail copied to {thumbnail_path}")
            except StopIteration:
                logger.warning("No training images available to copy for thumbnail.")
            except Exception as thumb_err:
                logger.warning(f"Failed to copy thumbnail for {config.output_dir}: {thumb_err}")

            # Persist lightweight metadata for the frontend
            try:
                metadata = {
                    "name": Path(config.output_dir).name,
                    "model_name": config.user_model_name or Path(config.output_dir).name,
                    "base_model": config.pretrained_model_name_or_path,
                    "prompt": config.instance_prompt,
                    "creation_time": datetime.utcnow().isoformat(),
                }
                if thumbnail_filename:
                    metadata["thumbnail"] = thumbnail_filename
                metadata_path = Path(config.output_dir) / "metadata.json"
                metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                logger.info(f"Metadata written to {metadata_path}")
            except Exception as metadata_err:
                logger.warning(f"Failed to write metadata for {config.output_dir}: {metadata_err}")

        accelerator.end_training()
        
        if status_updater and not terminated:
            status_updater.update({"status": "completed", "progress": 100, "message": f"Training complete! Model saved to {config.output_dir}"})

    except Exception as e:
        logger.error(f"Training failed with an error: {e}", exc_info=True)
        if status_updater:
            status_updater.update({"status": "failed", "progress": 0, "message": str(e)})
    finally:
        # Check if training was successful. If not, and the output dir is empty, clean it up.
        is_successful = status_updater and status_updater.get("status") == "completed"
        if not is_successful and os.path.exists(config.output_dir):
            # Check if the directory is empty
            if not os.listdir(config.output_dir):
                logger.info(f"Training was not successful. Deleting empty output directory: {config.output_dir}")
                try:
                    shutil.rmtree(config.output_dir)
                except OSError as e:
                    logger.error(f"Error deleting directory {config.output_dir}: {e}")

        # Cleanup
        del unet, text_encoder, vae, optimizer, train_dataloader, lr_scheduler, accelerator
        free_memory()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    # ... (omitting unchanged function for brevity)
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds
