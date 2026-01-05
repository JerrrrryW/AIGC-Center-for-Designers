from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
from typing import List, Optional
import shutil
import os
import json
import re
from datetime import datetime
import io
from diffusers import DiffusionPipeline
from threading import Lock

from .train_lora import TrainingConfig, start_training as run_lora_training
from .captioning import caption_images

app = FastAPI()

# --- In-memory store for training status ---
# In a real-world multi-user app, you'd use a database or Redis.
# For this local single-user app, a simple dict is sufficient.
training_status = {
    "status": "idle", # idle, initializing, loading_models, training, completed, failed
    "progress": 0,    # 0-100
    "message": "服务器已就绪。",
    "should_stop": False,
}

caption_status = {
    "status": "idle",  # idle, loading, processing, completed, failed
    "progress": 0,
    "message": "可开始生成描述。",
    "results": {},
}

BASE_MODEL_DEFAULT = "runwayml/stable-diffusion-v1-5"
DISABLE_SAFETY_CHECKER = True
_pipe_cache = {
    "base_model": None,
    "pipe": None,
}
_pipe_lock = Lock()

# --- CORS Middleware ---
# Allow LAN access by default while keeping localhost whitelisted.
default_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
origins_env = os.getenv("AIGC_CORS_ORIGINS", "")
origins = [origin.strip() for origin in origins_env.split(",") if origin.strip()] or default_origins
origin_regex_env = os.getenv("AIGC_CORS_ORIGIN_REGEX", "").strip()
origin_regex = origin_regex_env or r"^http://(localhost|127\.0\.0\.1|\d{1,3}(\.\d{1,3}){3}|.*\.local):5173$"
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    base_model: Optional[str] = None
    lora_models: Optional[List[str]] = None
    lora_model: Optional[str] = None # Backward compat for single LoRA

import uuid

inference_status = {
    "status": "idle", # idle, loading, processing, completed, failed
    "progress": 0,
    "step": 0,
    "total_steps": 50,
    "message": "可开始推理。",
    "image_id": None,
}

# In-memory store for generated images
# In a real app, you might use a temporary file store or a cache like Redis
generated_images = {}

# ... (omitting other parts of the file for brevity)

def run_inference_task(req: GenerateRequest):
    """The actual long-running task for generating an image."""
    inference_status.update({
        "status": "loading",
        "progress": 0,
        "step": 0,
        "message": "正在加载 Stable Diffusion 模型...",
        "image_id": None,
    })

    def progress_callback(pipe, step, timestep, callback_kwargs):
        inference_status.update({
            "status": "processing",
            "step": step,
            "progress": (step / inference_status["total_steps"]) * 100,
            "message": f"推理中... 步骤 {step}/{inference_status['total_steps']}",
        })
        return callback_kwargs

    pipe = None
    try:
        base_model_id = req.base_model or BASE_MODEL_DEFAULT
        lora_models = req.lora_models or []
        if not lora_models and req.lora_model:
            lora_models = [req.lora_model]
        lora_models = [name for name in lora_models if name and name != "None"]
        lora_models = list(dict.fromkeys(lora_models))

        with _pipe_lock:
            if _pipe_cache["pipe"] is None or _pipe_cache["base_model"] != base_model_id:
                if _pipe_cache["pipe"] is not None:
                    try:
                        _pipe_cache["pipe"].unload_lora_weights()
                    except Exception:
                        pass
                    del _pipe_cache["pipe"]
                    _pipe_cache["pipe"] = None
                    torch.cuda.empty_cache()
                    if torch.backends.mps.is_available():
                        try:
                            torch.mps.empty_cache()
                        except AttributeError:
                            pass

                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                pipe = DiffusionPipeline.from_pretrained(
                    base_model_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                )
                if DISABLE_SAFETY_CHECKER:
                    pipe.safety_checker = None
                    if hasattr(pipe, "feature_extractor"):
                        pipe.feature_extractor = None
                if device != "cpu":
                    pipe = pipe.to(device)
                pipe.enable_attention_slicing()
                _pipe_cache["pipe"] = pipe
                _pipe_cache["base_model"] = base_model_id
            else:
                pipe = _pipe_cache["pipe"]

            if lora_models:
                inference_status["message"] = f"正在加载 {len(lora_models)} 个 LoRA 模型..."
                adapter_names = []
                supports_multi = hasattr(pipe, "set_adapters")
                if not supports_multi and len(lora_models) > 1:
                    raise ValueError("当前 diffusers 版本不支持多个 LoRA 适配器。")
                for lora_name in lora_models:
                    lora_path = os.path.join("lora_models", lora_name)
                    if os.path.isdir(lora_path):
                        if supports_multi:
                            adapter_names.append(lora_name)
                            pipe.load_lora_weights(lora_path, adapter_name=lora_name)
                        else:
                            pipe.load_lora_weights(lora_path)
                    else:
                        raise FileNotFoundError(f"未找到 LoRA 模型目录：{lora_path}")

                if supports_multi:
                    pipe.set_adapters(adapter_names, adapter_weights=[1.0] * len(adapter_names))

        inference_status["status"] = "processing"
        image = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=inference_status["total_steps"],
            guidance_scale=7.5,
            callback_on_step_end=progress_callback,
        ).images[0]

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        
        image_id = str(uuid.uuid4())
        generated_images[image_id] = img_byte_arr.getvalue()

        inference_status.update({
            "status": "completed",
            "progress": 100,
            "message": "图片生成完成。",
            "image_id": image_id,
        })

    except Exception as e:
        print(f"Error during image generation: {e}")
        inference_status.update({"status": "failed", "message": str(e)})
    finally:
        if pipe is not None:
            with _pipe_lock:
                try:
                    pipe.unload_lora_weights()
                except Exception:
                    pass


def run_caption_task(images_payload, prefix=None, suffix=None):
    caption_status.update(
        {
            "status": "loading",
            "progress": 0,
            "message": "正在加载描述模型...",
            "results": {},
        }
    )
    try:
        caption_images(
            images_payload,
            prefix=prefix,
            suffix=suffix,
            status_updater=caption_status,
        )
    except Exception as e:
        print(f"Error during captioning: {e}")
        caption_status.update({"status": "failed", "message": str(e)})

@app.post("/generate")
async def start_generation(req: GenerateRequest, background_tasks: BackgroundTasks):
    if inference_status["status"] in ["loading", "processing"]:
        return {"status": "error", "message": "已有推理任务正在进行。"}
    
    background_tasks.add_task(run_inference_task, req)
    return {"status": "success", "message": "图片生成已在后台开始。"}

@app.get("/generate/status")
def get_inference_status():
    return inference_status

@app.post("/caption")
async def start_captioning(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(...),
    prefix: Optional[str] = Form(None),
    suffix: Optional[str] = Form(None),
):
    if caption_status["status"] in ["loading", "processing"]:
        return {"status": "error", "message": "已有描述生成任务正在进行。"}

    images_payload = []
    for image in images:
        filename = os.path.basename(str(image.filename))
        content = await image.read()
        images_payload.append({"filename": filename, "content": content})

    caption_status.update(
        {
            "status": "loading",
            "progress": 0,
            "message": "正在准备描述生成...",
            "results": {},
        }
    )
    background_tasks.add_task(run_caption_task, images_payload, prefix, suffix)
    return {"status": "success", "message": "描述生成已在后台开始。"}

@app.get("/caption/status")
def get_caption_status():
    return caption_status

@app.get("/models")
def get_lora_models(request: Request):
    models_dir = "lora_models"
    if not os.path.exists(models_dir):
        return []

    model_folders = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    
    models_info = []
    for folder in model_folders:
        folder_path = os.path.join(models_dir, folder)
        # Check if the directory is empty and delete it if so
        if not os.listdir(folder_path):
            print(f"Found and deleting empty model directory: {folder_path}")
            try:
                shutil.rmtree(folder_path)
            except OSError as e:
                print(f"Error deleting empty directory {folder_path}: {e}")
            continue

        try:
            metadata_path = os.path.join(folder_path, "metadata.json")
            metadata = {}
            if os.path.isfile(metadata_path):
                try:
                    with open(metadata_path, "r", encoding="utf-8") as meta_file:
                        metadata = json.load(meta_file)
                except (OSError, json.JSONDecodeError) as meta_err:
                    print(f"Failed to read metadata for {folder}: {meta_err}")

            prompt = metadata.get("prompt")
            creation_time = metadata.get("creation_time")
            model_display_name = metadata.get("model_name") or metadata.get("display_name")
            base_model = metadata.get("base_model") or metadata.get("base_model_name")

            if not creation_time:
                try:
                    parts = folder.split('-')
                    date = parts[1]
                    time = parts[2]
                    creation_time = datetime.strptime(f"{date}-{time}", "%Y%m%d-%H%M%S").isoformat()
                except (IndexError, ValueError):
                    creation_time = datetime.fromtimestamp(os.path.getmtime(folder_path)).isoformat()

            if not prompt:
                try:
                    prompt = folder.split('-')[0].replace('_', ' ')
                except IndexError:
                    prompt = folder

            thumbnail_url = None
            thumbnail_file = metadata.get("thumbnail", "thumbnail.png")
            thumbnail_path = os.path.join(folder_path, thumbnail_file)
            if os.path.isfile(thumbnail_path):
                base_url = str(request.base_url).rstrip("/")
                thumbnail_url = f"{base_url}/models/{folder}/thumbnail"

            models_info.append({
                "name": folder,
                "model_name": model_display_name or folder,
                "base_model": base_model or "未知",
                "prompt": prompt,
                "creation_time": creation_time,
                "thumbnail_url": thumbnail_url,
            })
        except Exception as e:
            print(f"Failed to process model folder '{folder}': {e}")
            continue
            
    # Sort models by creation time, newest first
    models_info.sort(key=lambda x: x["creation_time"], reverse=True)
    
    return models_info

@app.get("/models/download/{model_name}")
def download_lora_model(model_name: str):
    models_dir = "lora_models"
    model_path = os.path.join(models_dir, model_name)

    if not os.path.isdir(model_path):
        return JSONResponse(status_code=404, content={"message": "未找到模型。"})

    # Check if the directory is empty
    if not os.listdir(model_path):
        print(f"Attempted to download an empty model directory. Deleting it: {model_path}")
        try:
            shutil.rmtree(model_path)
        except OSError as e:
            print(f"Error deleting empty directory {model_path}: {e}")
        return JSONResponse(status_code=404, content={"message": "模型目录为空，已删除。请刷新模型列表。"})

    # Create a zip archive of the model directory
    shutil.make_archive(model_name, 'zip', model_path)

    return FileResponse(f"{model_name}.zip", media_type='application/zip', filename=f"{model_name}.zip")

@app.delete("/models/delete/{model_name}")
def delete_lora_model(model_name: str):
    models_dir = "lora_models"
    model_path = os.path.join(models_dir, model_name)

    if not os.path.isdir(model_path):
        return {"status": "error", "message": "未找到模型。"}

    try:
        shutil.rmtree(model_path)
        return {"status": "success", "message": f"模型 '{model_name}' 已删除。"}
    except Exception as e:
        return {"status": "error", "message": f"删除模型失败：{e}"}

@app.get("/models/{model_name}/thumbnail")
def get_model_thumbnail(model_name: str):
    models_dir = "lora_models"
    thumbnail_path = os.path.join(models_dir, model_name, "thumbnail.png")
    if not os.path.isfile(thumbnail_path):
        return JSONResponse(status_code=404, content={"message": "未找到缩略图。"})
    return FileResponse(thumbnail_path, media_type="image/png")


@app.get("/generate/image/{image_id}")
def get_generated_image(image_id: str):
    image_data = generated_images.get(image_id)
    if not image_data:
        return {"status": "error", "message": "未找到图片。"}
    
    # Clean up the image from memory after it's been fetched once
    # del generated_images[image_id]
    
    return StreamingResponse(io.BytesIO(image_data), media_type="image/png")

@app.get("/check-mps")
def check_mps():
    # ... (omitting unchanged endpoint for brevity)
    if torch.backends.mps.is_available():
        return {"status": "success", "message": "MPS 可用，已准备好在 Mac 上进行 GPU 加速。"}
    else:
        return {"status": "error", "message": "MPS 不可用，服务器将使用 CPU。"}

@app.get("/train/status")
def get_training_status():
    """Endpoint for the frontend to poll for training status."""
    return training_status

@app.post("/train")
async def trigger_training(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(...),
    baseModel: str = Form(...),
    instancePrompt: str = Form(...),
    modelName: Optional[str] = Form(None),
    useCaptions: bool = Form(False),
    captions: Optional[str] = Form(None),
    steps: int = Form(...),
    learningRate: float = Form(...),
    resolution: int = Form(...),
    trainBatchSize: int = Form(...),
):
    if training_status["status"] == "training":
        return {"status": "error", "message": "已有训练任务正在进行。"}

    image_dir = "temp_training_images"
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir)

    safe_prompt = instancePrompt.replace(' ', '_').replace('\\','').replace('/','')
    safe_prompt = re.sub(r'[^A-Za-z0-9_-]', '', safe_prompt) or "instance"

    captions_map = {}
    if useCaptions:
        if captions:
            try:
                captions_map = json.loads(captions)
            except json.JSONDecodeError:
                return {"status": "error", "message": "描述数据无效。"}
        else:
            return {"status": "error", "message": "启用描述训练时必须提供描述。"}

    for index, image in enumerate(images, start=1):
        original_name = os.path.basename(str(image.filename))
        _, ext = os.path.splitext(original_name)
        ext = ext if ext else ".png"
        filename = f"{safe_prompt}-{index:03d}{ext}"
        file_path = os.path.join(image_dir, filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        if useCaptions:
            caption_text = captions_map.get(original_name, "")
            caption_text = caption_text.strip() if caption_text else ""
            if caption_text:
                caption_path = os.path.splitext(file_path)[0] + ".txt"
                with open(caption_path, "w", encoding="utf-8") as caption_file:
                    caption_file.write(caption_text)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"lora_models/{safe_prompt}-{timestamp}"

    user_model_name = modelName.strip() if modelName else None

    training_config = TrainingConfig(
        pretrained_model_name_or_path=baseModel,
        instance_data_dir=image_dir,
        output_dir=output_dir,
        instance_prompt=instancePrompt,
        user_model_name=user_model_name,
        max_train_steps=steps,
        learning_rate=learningRate,
        resolution=resolution,
        train_batch_size=trainBatchSize,
        # Using some sensible defaults for other params
        gradient_accumulation_steps=1,
        gradient_checkpointing=True, # Good for memory saving
        lr_scheduler="constant",
        report_to="tensorboard", # Will create local logs
         mixed_precision="no", # Required for MPS
    )

    # Reset status and add the training function to background tasks
    training_status.update({"status": "initializing", "progress": 0, "message": "已收到请求...", "should_stop": False})
    background_tasks.add_task(run_lora_training, config=training_config, status_updater=training_status)

    return {
        "status": "success",
        "message": f"训练已在后台开始，模型将保存到 '{output_dir}'。",
    }

@app.post("/train/terminate")
def terminate_training():
    """Endpoint to signal the training process to stop."""
    if training_status["status"] in ["training", "initializing", "loading_models"]:
        training_status["should_stop"] = True
        training_status["message"] = "已收到终止信号，正在完成当前步骤..."
        return {"status": "success", "message": "已发送终止信号。"}
    else:
        return {"status": "error", "message": "当前没有正在进行的训练。"}
