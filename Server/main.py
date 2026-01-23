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
import base64
import urllib.request
import urllib.error
from diffusers import DiffusionPipeline
from threading import Lock
from PIL import Image, ImageDraw, ImageFont

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
LAYOUT_SCHEMA_VERSION = "1.2"
llm_runtime_config = {
    "api_key": None,
    "base_url": None,
    "model": None,
    "temperature": None,
    "image_api_key": None,
    "image_base_url": None,
    "image_model": None,
}
IMAGE_PROVIDER_ENV = "IMAGE_GENERATION_PROVIDER"
_pipe_cache = {
    "base_model": None,
    "pipe": None,
}
_pipe_lock = Lock()


def _ensure_pipe(base_model_id: str):
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
    return _pipe_cache["pipe"]

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


class TextRequest(BaseModel):
    prompt: str


class LlmConfigRequest(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    image_api_key: Optional[str] = None
    image_base_url: Optional[str] = None
    image_model: Optional[str] = None


class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    base_model: Optional[str] = None
    lora_models: Optional[List[str]] = None
    width: Optional[int] = 512
    height: Optional[int] = 512
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5
    image_provider: Optional[str] = None


class LayoutRequest(BaseModel):
    prompt: str
    chat_history: Optional[str] = None
    selected_options: Optional[dict] = None


class LayeredRequest(BaseModel):
    prompt: str
    aspect_ratio: Optional[str] = "3:4"
    count: Optional[int] = 3  # total layers including background


@app.post("/generate-json")
def generate_json(req: TextRequest):
    """LiteDraw-compatible: return a JSON object with text_response + form choices."""
    if not req.prompt:
        return JSONResponse(status_code=400, content={"error": "prompt 不能为空"})
    result = generate_llm_form(req.prompt)
    return {"json_object": result}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    selected_options: Optional[dict] = None


@app.post("/chat")
def chat(req: ChatRequest):
    """Multi-turn chat with optional form generation; returns LiteDraw-like JSON payload."""
    if not req.messages:
        return JSONResponse(status_code=400, content={"error": "messages 不能为空"})
    selected_options = req.selected_options or {}
    result = generate_llm_chat(req.messages, selected_options=selected_options)

    # Ensure the very first turn has a form (LiteDraw expectation).
    user_messages = [m for m in req.messages if (m.role or "").lower() == "user"]
    should_force_form = len(user_messages) <= 1 and not selected_options
    if should_force_form and not (result.get("form") or []):
        last_user_prompt = user_messages[-1].content if user_messages else ""
        if last_user_prompt:
            result = generate_llm_form(last_user_prompt)
    return {"json_object": result}


@app.get("/llm-config")
def get_llm_config():
    config = _resolve_llm_config()
    image_config = _resolve_image_config()
    return {
        "api_key_set": bool(config.get("api_key")),
        "base_url": config.get("base_url"),
        "model": config.get("model"),
        "temperature": config.get("temperature"),
        "image_api_key_set": bool(image_config.get("api_key")),
        "image_base_url": image_config.get("base_url"),
        "image_model": image_config.get("model"),
    }


@app.post("/llm-config")
def set_llm_config(req: LlmConfigRequest):
    if req.api_key is not None:
        api_key = req.api_key.strip()
        llm_runtime_config["api_key"] = api_key or None
    if req.base_url is not None:
        base_url = req.base_url.strip()
        llm_runtime_config["base_url"] = base_url or None
    if req.model is not None:
        model = req.model.strip()
        llm_runtime_config["model"] = model or None
    if req.temperature is not None:
        llm_runtime_config["temperature"] = float(req.temperature)
    if req.image_api_key is not None:
        image_api_key = req.image_api_key.strip()
        llm_runtime_config["image_api_key"] = image_api_key or None
    if req.image_base_url is not None:
        image_base_url = req.image_base_url.strip()
        llm_runtime_config["image_base_url"] = image_base_url or None
    if req.image_model is not None:
        image_model = req.image_model.strip()
        llm_runtime_config["image_model"] = image_model or None
    config = _resolve_llm_config()
    image_config = _resolve_image_config()
    return {
        "status": "ok",
        "api_key_set": bool(config.get("api_key")),
        "base_url": config.get("base_url"),
        "model": config.get("model"),
        "temperature": config.get("temperature"),
        "image_api_key_set": bool(image_config.get("api_key")),
        "image_base_url": image_config.get("base_url"),
        "image_model": image_config.get("model"),
    }


def _resolve_llm_temperature() -> float:
    override = llm_runtime_config.get("temperature")
    if isinstance(override, (int, float)):
        return float(override)
    env_temp = os.getenv("SILICONFLOW_TEMPERATURE", "").strip()
    if env_temp:
        try:
            return float(env_temp)
        except ValueError:
            pass
    return 0.4


def _resolve_llm_config() -> dict:
    api_key = (llm_runtime_config.get("api_key") or "").strip()
    if not api_key:
        api_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    base_url = (llm_runtime_config.get("base_url") or "").strip()
    if not base_url:
        base_url = os.getenv("SILICONFLOW_BASE_URL", "").strip()
    base_url = base_url or "https://api.siliconflow.cn/v1/chat/completions"
    model = (llm_runtime_config.get("model") or "").strip()
    if not model:
        model = os.getenv("SILICONFLOW_MODEL", "").strip()
    model = model or "Qwen/Qwen2.5-7B-Instruct"
    temperature = _resolve_llm_temperature()
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "temperature": temperature,
    }


def _siliconflow_enabled() -> bool:
    return bool(_resolve_llm_config().get("api_key"))


def _resolve_image_config() -> dict:
    api_key = (llm_runtime_config.get("image_api_key") or "").strip()
    if not api_key:
        api_key = (llm_runtime_config.get("api_key") or "").strip()
    if not api_key:
        api_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    base_url = (llm_runtime_config.get("image_base_url") or "").strip()
    if not base_url:
        base_url = os.getenv("SILICONFLOW_IMAGE_BASE_URL", "").strip()
    base_url = base_url or "https://api.siliconflow.cn/v1"
    model = (llm_runtime_config.get("image_model") or "").strip()
    if not model:
        model = os.getenv("SILICONFLOW_IMAGE_MODEL", "").strip()
    model = model or "Qwen/Qwen-Image"
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
    }


def _siliconflow_image_enabled() -> bool:
    return bool(_resolve_image_config().get("api_key"))


def _call_siliconflow_chat(messages: List[dict], timeout_seconds: int = 60) -> str:
    """
    Minimal SiliconFlow chat-completions call.
    Env:
      - SILICONFLOW_API_KEY (required)
      - SILICONFLOW_BASE_URL (optional, default OpenAI-compatible endpoint)
      - SILICONFLOW_MODEL (optional)
    """
    config = _resolve_llm_config()
    api_key = (config.get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError("SILICONFLOW_API_KEY 未设置")
    base_url = config.get("base_url")
    model = config.get("model")
    temperature = config.get("temperature", 0.4)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(body)
    # OpenAI-compatible: choices[0].message.content
    return (
        parsed.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )


def _resolve_image_provider(requested: Optional[str]) -> str:
    provider = (requested or os.getenv(IMAGE_PROVIDER_ENV, "")).strip().lower()
    if not provider:
        provider = "siliconflow" if _siliconflow_image_enabled() else "local"
    if provider == "auto":
        return "siliconflow" if _siliconflow_image_enabled() else "local"
    if provider not in {"siliconflow", "local"}:
        raise RuntimeError(f"不支持的生图链路：{provider}")
    if provider == "siliconflow" and not _siliconflow_image_enabled():
        raise RuntimeError("SILICONFLOW_API_KEY 未设置")
    return provider


def _call_siliconflow_image(prompt: str, width: int, height: int, steps: int, timeout_seconds: int = 120) -> str:
    """
    SiliconFlow image generation (OpenAI-compatible).
    Env:
      - SILICONFLOW_API_KEY (required)
      - SILICONFLOW_IMAGE_BASE_URL (optional, default https://api.siliconflow.cn/v1)
      - SILICONFLOW_IMAGE_MODEL (optional, default Qwen/Qwen-Image)
    """
    config = _resolve_image_config()
    api_key = config.get("api_key", "").strip()
    if not api_key:
        raise RuntimeError("SILICONFLOW_API_KEY 未设置")
    base_url = (config.get("base_url") or "").rstrip("/")
    model = config.get("model")

    payload = {
        "model": model,
        "prompt": prompt,
        "size": f"{width}x{height}",
        "n": 1,
        "response_format": "b64_json",
        "extra_body": {"step": steps},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/images/generations",
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SiliconFlow 生图失败: {body}") from e

    parsed = json.loads(body)
    items = parsed.get("data") or []
    if not items:
        raise RuntimeError("SiliconFlow 未返回图片数据")
    item = items[0] or {}
    image_b64 = item.get("b64_json")
    if image_b64:
        return image_b64
    url = item.get("url")
    if url:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            image_bytes = response.read()
        return base64.b64encode(image_bytes).decode("utf-8")
    raise RuntimeError("SiliconFlow 返回数据缺少 b64_json/url")


def _safe_parse_json_object(text: str) -> dict:
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt to extract the first {...} block.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
        return {}


def _coerce_options(value) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        parts = re.split(r"[,，、;\n；]+", value)
        return [p.strip() for p in parts if p.strip()]
    return []


def _coerce_number(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _coerce_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1", "on"):
            return True
        if lowered in ("false", "no", "0", "off"):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _coerce_palette_options(value) -> List[dict]:
    if not isinstance(value, list):
        return []
    options = []
    for raw in value:
        if isinstance(raw, dict):
            raw_value = raw.get("value") or raw.get("label") or raw.get("color")
            if not raw_value:
                continue
            entry = {"value": str(raw_value).strip()}
            label = raw.get("label")
            if isinstance(label, str) and label.strip():
                entry["label"] = label.strip()
            color = raw.get("color")
            if isinstance(color, str) and color.strip():
                entry["color"] = color.strip()
            options.append(entry)
        else:
            text = str(raw).strip()
            if not text:
                continue
            options.append({"value": text, "label": text})
    return options[:10]


def _normalize_form_sections(form_value) -> List[dict]:
    if not isinstance(form_value, list):
        return []
    normalized: List[dict] = []
    for idx, item in enumerate(form_value):
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        title = title.strip() if isinstance(title, str) and title.strip() else f"问题 {idx + 1}"
        options = _coerce_options(item.get("options"))
        options = list(dict.fromkeys([opt for opt in options if opt]))[:6]
        if len(options) < 1:
            continue
        normalized.append({"title": title, "options": options})
    return normalized[:4]


def _normalize_llm_payload(obj: dict, default_text: str) -> dict:
    if not isinstance(obj, dict):
        return {"text_response": default_text, "form": []}
    text = obj.get("text_response")
    if not isinstance(text, str) or not text.strip():
        text = default_text
    form = _normalize_form_sections(obj.get("form"))
    return {"text_response": text.strip(), "form": form}


def generate_llm_form(user_prompt: str) -> dict:
    """Generate initial form. Falls back to static form when SiliconFlow is unavailable."""
    fallback = {
        "text_response": "好的！为了更接近你想要的效果，请先选几项偏好：",
        "form": [
            {"title": "画幅比例", "options": ["3:4", "1:1", "16:9"]},
            {"title": "风格方向", "options": ["摄影", "插画", "3D 渲染", "平面海报"]},
            {"title": "氛围", "options": ["明亮清爽", "暗黑电影感", "梦幻柔光", "复古胶片"]},
        ],
    }
    if not _siliconflow_enabled():
        fallback["text_response"] += "（当前未配置 SILICONFLOW_API_KEY，使用本地兜底表单）"
        return fallback

    system = (
        "你是一个文生图应用的对话助手。任务：根据用户的初始想法生成 2-4 个选择题表单，用于澄清需求。\n"
        "输出必须是单个 JSON 对象，且只包含这些字段：text_response(string), form(array).\n"
        "form 中每一项为 {title: string, options: string[]}，options 2-4 个，高层抽象，不要数值参数。\n"
        "全部用中文。"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"用户想生成：{user_prompt}"},
    ]
    try:
        content = _call_siliconflow_chat(messages)
        obj = _safe_parse_json_object(content)
        normalized = _normalize_llm_payload(obj, default_text=fallback["text_response"])
        if normalized.get("form"):
            return normalized
    except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, ValueError) as e:
        fallback["text_response"] += f"（SiliconFlow 调用失败：{e}，使用本地兜底表单）"
        return fallback
    except Exception as e:
        fallback["text_response"] += f"（SiliconFlow 未知错误：{e}，使用本地兜底表单）"
        return fallback
    fallback["text_response"] += "（SiliconFlow 返回格式异常，使用本地兜底表单）"
    return fallback


def generate_llm_chat(messages: List[ChatMessage], selected_options: dict) -> dict:
    """
    Multi-turn assistant. If a form has not been generated yet, generate it.
    Otherwise respond and optionally return a refined form (can be empty).
    """
    fallback = {
        "text_response": "收到。你也可以继续补充细节，或点击进入编辑界面。",
        "form": [],
    }
    if not _siliconflow_enabled():
        fallback["text_response"] += "（当前未配置 SILICONFLOW_API_KEY，使用本地回复）"
        return fallback

    system = (
        "你是一个文生图应用的对话助手。你需要进行多轮对话：\n"
        "1) 如果用户还没完成需求澄清，请给出简短追问，并可返回 form(2-4题)；\n"
        "2) 如果用户已给出足够信息，可只返回 text_response，form 可以为空数组。\n"
        "输出必须是单个 JSON 对象，字段：text_response(string), form(array)。form 允许为空数组。\n"
        "全部用中文。"
    )
    sf_messages = [{"role": "system", "content": system}]
    for m in messages[-20:]:
        role = "assistant" if m.role in ("assistant", "bot") else "user"
        sf_messages.append({"role": role, "content": m.content})
    sf_messages.append(
        {"role": "user", "content": f"当前用户已选表单项(JSON)：{json.dumps(selected_options, ensure_ascii=False)}"}
    )
    try:
        content = _call_siliconflow_chat(sf_messages)
        obj = _safe_parse_json_object(content)
        return _normalize_llm_payload(obj, default_text=fallback["text_response"])
    except Exception:
        return fallback
    return fallback

def _pick_selected_first(selected: dict, keys: List[str]) -> Optional[str]:
    for key in keys:
        value = selected.get(key)
        if isinstance(value, list):
            for item in value:
                text = str(item).strip()
                if text:
                    return text
        elif isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return None


def _pick_selected_list(selected: dict, keys: List[str]) -> List[str]:
    for key in keys:
        value = selected.get(key)
        if isinstance(value, list):
            items = [str(v).strip() for v in value if str(v).strip()]
            if items:
                return items
        elif isinstance(value, str):
            items = _coerce_options(value)
            if items:
                return items
    return []


def _format_selected_summary(selected_options: dict, limit: int = 3) -> str:
    if not isinstance(selected_options, dict):
        return ""
    parts = []
    for key, value in selected_options.items():
        if not key:
            continue
        if isinstance(value, list):
            values = [str(v).strip() for v in value if str(v).strip()]
            if not values:
                continue
            parts.append(f"{key}:{'、'.join(values)}")
        elif isinstance(value, str) and value.strip():
            parts.append(f"{key}:{value.strip()}")
        if len(parts) >= limit:
            break
    return "；".join(parts)


def _infer_aspect_ratio(prompt: str, selected: dict) -> str:
    selected_ratio = _pick_selected_first(selected, ["画幅比例", "比例", "画幅", "比例选择"])
    if isinstance(selected_ratio, str) and ":" in selected_ratio:
        return selected_ratio.strip()
    if prompt:
        if re.search(r"(横版|横向|横幅|宽屏|16:9|电影|宽画幅)", prompt, re.IGNORECASE):
            return "16:9"
        if re.search(r"(方形|正方形|1:1)", prompt):
            return "1:1"
        if re.search(r"(竖版|竖向|海报|长图|9:16|4:5|3:4)", prompt):
            return "3:4"
    return "3:4"


def _infer_style(prompt: str) -> str:
    if not prompt:
        return "摄影"
    if re.search(r"(插画|漫画|手绘|绘本)", prompt):
        return "插画"
    if re.search(r"(3d|建模|渲染)", prompt, re.IGNORECASE):
        return "3D 渲染"
    if re.search(r"(海报|banner|平面|宣传)", prompt, re.IGNORECASE):
        return "平面海报"
    if re.search(r"(摄影|照片|写实|realistic)", prompt, re.IGNORECASE):
        return "摄影"
    return "摄影"


def _infer_moods(prompt: str) -> List[str]:
    if not prompt:
        return ["明亮清爽"]
    moods = []
    if re.search(r"(明亮|清新|阳光|通透|清爽)", prompt):
        moods.append("明亮清爽")
    if re.search(r"(暗黑|夜|阴影|电影感|暗调)", prompt):
        moods.append("暗黑电影感")
    if re.search(r"(梦幻|柔光|仙|童话|迷幻)", prompt):
        moods.append("梦幻柔光")
    if re.search(r"(复古|胶片|怀旧|老照片)", prompt):
        moods.append("复古胶片")
    return moods or ["明亮清爽"]


def _infer_subject_count(prompt: str) -> int:
    if not prompt:
        return 2
    if re.search(r"(多人|群像|人群|群体|大量)", prompt):
        return 3
    if re.search(r"(三|3)\s*(人|个|位|只|件|主体|主角)", prompt):
        return 3
    if re.search(r"(两|双|2)\s*(人|个|位|只|件|主体|主角)", prompt):
        return 2
    if re.search(r"(一|单|1)\s*(人|个|位|只|件|主体|主角)", prompt):
        return 1
    if re.search(r"(单人|单体|独立主体|一个人|一人)", prompt):
        return 1
    if re.search(r"(二人|两人|双人)", prompt):
        return 2
    if re.search(r"(三人)", prompt):
        return 3
    return 2


def _infer_subject_label(prompt: str) -> str:
    if not prompt:
        return "主体"
    if re.search(r"(人物|人像|模特|男|女|儿童|老人|角色|演员)", prompt):
        return "人物"
    if re.search(r"(动物|猫|狗|鸟|马|鹿)", prompt):
        return "主角"
    return "主体"


def _needs_decor_slot(prompt: str, style: str) -> bool:
    if style == "平面海报":
        return True
    if not prompt:
        return False
    return bool(re.search(r"(logo|LOGO|标志|标识|文字|文案|标题|海报|banner|宣传|广告|排版)", prompt))


def _compose_slot_prompt(prompt: str, style: str, moods: List[str]) -> str:
    parts = []
    base = (prompt or "").strip()
    if base:
        parts.append(base)
    if style:
        parts.append(f"风格{style}")
    if moods:
        parts.append(f"氛围{'/'.join(moods)}")
    return "，".join(parts)


def _merge_options(base: List[str], additions: List[str]) -> List[str]:
    merged = list(base)
    for item in additions:
        if item and item not in merged:
            merged.append(item)
    return merged


def _build_rule_layout_payload(prompt: str, chat_history: Optional[str], selected_options: Optional[dict]) -> dict:
    selected = selected_options or {}
    aspect_ratio = _infer_aspect_ratio(prompt, selected)
    style_options = ["摄影", "插画", "3D 渲染", "平面海报"]
    mood_options = ["明亮清爽", "暗黑电影感", "梦幻柔光", "复古胶片"]

    selected_style = _pick_selected_first(selected, ["风格", "风格方向"])
    style_default = selected_style or _infer_style(prompt)
    style_options = _merge_options(style_options, [style_default])

    selected_moods = _pick_selected_list(selected, ["氛围", "情绪氛围", "情绪", "氛围关键词"])
    mood_defaults = selected_moods or _infer_moods(prompt)
    mood_options = _merge_options(mood_options, mood_defaults)

    subject_count = _infer_subject_count(prompt)
    subject_label = _infer_subject_label(prompt)
    include_decor = _needs_decor_slot(prompt, style_default)
    slot_prompt = _compose_slot_prompt(prompt, style_default, mood_defaults)

    slots = [
        {
            "id": "background",
            "label": "背景",
            "layerType": "background",
            "prompt": f"{slot_prompt}，背景",
        }
    ]
    for idx in range(subject_count):
        label = subject_label if subject_count == 1 else f"{subject_label} {idx + 1}"
        slots.append(
            {
                "id": f"subject-{idx + 1}",
                "label": label,
                "layerType": "subject",
                "prompt": f"{slot_prompt}，{label}",
            }
        )
    if include_decor:
        slots.append(
            {
                "id": "decor-1",
                "label": "文字/装饰",
                "layerType": "decor",
                "prompt": f"{slot_prompt}，文字或装饰元素",
            }
        )

    selected_summary = _format_selected_summary(selected, limit=3)
    summary = f"为「{prompt}」生成的专属配置页"
    if selected_summary:
        summary = f"{summary}（已结合 {selected_summary}）"

    text_response = f"已根据你的需求生成编辑页：{prompt}。"
    if selected_summary:
        text_response = f"{text_response} 当前选项：{selected_summary}。"

    layout_config = {
        "meta": {
            "sourcePrompt": prompt,
            "summary": summary,
            "aspect_ratio": aspect_ratio,
            "aspect_ratio_field": "scene-aspect",
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "layout": {
                "columns": 3,
                "minCardWidth": 280,
                "gap": 2,
                "dense": True,
            },
        },
        "sections": [
            {
                "id": "base",
                "title": "场景基础",
                "description": "命名、比例与核心描述",
                "components": [
                    {
                        "id": "scene-title",
                        "type": "text-input",
                        "label": "场景名称",
                        "placeholder": "例如：未来城市夜景",
                        "default": prompt[:20] if prompt else "",
                    },
                    {
                        "id": "scene-aspect",
                        "type": "ratio-select",
                        "label": "画幅比例",
                        "options": ["3:4", "1:1", "16:9", "9:16"],
                        "default": aspect_ratio,
                        "helperText": "用于预览与导出比例",
                    },
                    {
                        "id": "scene-notes",
                        "type": "textarea",
                        "label": "补充描述",
                        "placeholder": "补充材质、色彩、光照等细节",
                    },
                ],
            },
            {
                "id": "style",
                "title": "风格与氛围",
                "description": "更偏向视觉方向的选择",
                "components": [
                    {
                        "id": "scene-style",
                        "type": "select",
                        "label": "风格方向",
                        "options": style_options,
                        "default": style_default,
                        "display": "chips",
                    },
                    {
                        "id": "scene-mood",
                        "type": "multi-select",
                        "label": "情绪氛围",
                        "options": mood_options,
                        "default": mood_defaults,
                        "display": "chips",
                    },
                    {
                        "id": "color-tone",
                        "type": "color-palette",
                        "label": "主色调",
                        "options": [
                            {"value": "暖色", "label": "暖色", "color": "#F4A261"},
                            {"value": "冷色", "label": "冷色", "color": "#5DADE2"},
                            {"value": "高对比", "label": "高对比", "color": "#1F1F1F"},
                            {"value": "粉彩", "label": "粉彩", "color": "#F7C7D9"},
                        ],
                        "default": "暖色",
                    },
                ],
            },
            {
                "id": "prompt",
                "title": "提示词",
                "description": "编辑生成核心提示词",
                "layout": {"span": 2, "tone": "accent"},
                "components": [
                    {
                        "id": "prompt-editor",
                        "type": "prompt-editor",
                        "title": "提示词编辑",
                        "fields": [
                            {
                                "id": "positive",
                                "label": "正向提示词",
                                "placeholder": "主体、风格、氛围、镜头、材质",
                                "default": prompt,
                            },
                            {
                                "id": "negative",
                                "label": "负向提示词",
                                "placeholder": "不需要出现的元素",
                                "default": "",
                            },
                        ],
                    }
                ],
            },
            {
                "id": "generation",
                "title": "生成参数",
                "description": "基础生成控制项",
                "components": [
                    {
                        "id": "cfg-scale",
                        "type": "slider",
                        "label": "提示词强度",
                        "min": 3,
                        "max": 12,
                        "step": 0.5,
                        "default": 7,
                        "helperText": "数值越高越贴合描述，但可能损失多样性",
                    },
                    {
                        "id": "steps",
                        "type": "number-input",
                        "label": "推理步数",
                        "min": 10,
                        "max": 60,
                        "step": 1,
                        "default": 28,
                        "unit": "步",
                        "helperText": "步数越高越细腻，但耗时更长",
                    },
                    {
                        "id": "seed-lock",
                        "type": "toggle",
                        "label": "固定随机种子",
                        "default": False,
                        "helperText": "开启后多次生成更稳定",
                    },
                ],
            },
            {
                "id": "materials",
                "title": "素材准备",
                "description": "上传或生成素材图层",
                "layout": {"span": 2, "tone": "soft"},
                "components": [
                    {
                        "id": "materials-uploader",
                        "type": "media-uploader",
                        "title": "上传或生成素材",
                        "required": True,
                        "max": max(1, len(slots)),
                        "slots": slots,
                    }
                ],
            },
        ],
    }
    return {"text_response": text_response, "layout_config": layout_config}


def _normalize_layer_type(raw_type: Optional[str], label: str) -> str:
    if isinstance(raw_type, str):
        value = raw_type.strip().lower()
        if value in ("background", "subject", "decor"):
            return value
        if "bg" in value or "背景" in raw_type:
            return "background"
        if "decor" in value or "装饰" in raw_type or "文字" in raw_type or "logo" in raw_type.lower():
            return "decor"
        if "subject" in value or "主体" in raw_type or "人物" in raw_type:
            return "subject"
    if label and "背景" in label:
        return "background"
    if label and any(k in label for k in ["文字", "logo", "标语", "装饰"]):
        return "decor"
    return "subject"


def _ensure_unique_id(raw_id: str, used_ids: set, fallback_prefix: str) -> str:
    base = str(raw_id).strip() if raw_id is not None else ""
    if not base:
        base = f"{fallback_prefix}-{len(used_ids) + 1}"
    candidate = base
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base}-{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


def _normalize_number_range(min_val, max_val, step_val, default_val) -> dict:
    min_num = _coerce_number(min_val)
    max_num = _coerce_number(max_val)
    step_num = _coerce_number(step_val)
    default_num = _coerce_number(default_val)

    if min_num is None and max_num is None:
        min_num, max_num = 0.0, 1.0
    elif min_num is None:
        min_num = 0.0
    elif max_num is None:
        max_num = min_num + 1.0

    if max_num < min_num:
        min_num, max_num = max_num, min_num

    span = max_num - min_num
    if step_num is None or step_num <= 0:
        if span >= 10:
            step_num = 1.0
        elif span <= 1:
            step_num = 0.05
        else:
            step_num = round(span / 10, 2)

    if default_num is None:
        default_num = min_num
    if default_num < min_num:
        default_num = min_num
    if default_num > max_num:
        default_num = max_num

    return {"min": min_num, "max": max_num, "step": step_num, "default": default_num}


def _sanitize_prompt_fields(raw_fields, base_prompt: str) -> List[dict]:
    fields = []
    used_ids: set = set()
    if isinstance(raw_fields, list):
        for idx, field in enumerate(raw_fields):
            if not isinstance(field, dict):
                continue
            field_id = _ensure_unique_id(field.get("id") or f"field-{idx + 1}", used_ids, "field")
            label = field.get("label")
            placeholder = field.get("placeholder")
            default = field.get("default")
            helper_text = field.get("helperText")
            entry = {"id": field_id}
            if isinstance(label, str) and label.strip():
                entry["label"] = label.strip()
            if isinstance(placeholder, str) and placeholder.strip():
                entry["placeholder"] = placeholder.strip()
            if isinstance(helper_text, str) and helper_text.strip():
                entry["helperText"] = helper_text.strip()
            if default is not None:
                entry["default"] = str(default)
            fields.append(entry)
    if not fields:
        fields = [
            {
                "id": "positive",
                "label": "正向提示词",
                "placeholder": "主体、风格、氛围、细节",
                "default": base_prompt or "",
            },
            {
                "id": "negative",
                "label": "负向提示词",
                "placeholder": "不希望出现的元素",
                "default": "",
            },
        ]
    return fields


def _normalize_layout_meta(raw_layout) -> dict:
    if not isinstance(raw_layout, dict):
        return {}
    layout = {}
    columns = _coerce_number(raw_layout.get("columns") or raw_layout.get("cols"))
    if columns is not None and columns > 0:
        layout["columns"] = int(max(1, columns))
    min_card = _coerce_number(raw_layout.get("minCardWidth") or raw_layout.get("min_card_width"))
    if min_card is not None and min_card > 0:
        layout["minCardWidth"] = int(max(240, min_card))
    gap = _coerce_number(raw_layout.get("gap") or raw_layout.get("gridGap"))
    if gap is not None and gap > 0:
        layout["gap"] = int(max(1, gap))
    dense = raw_layout.get("dense")
    if dense is not None:
        layout["dense"] = _coerce_bool(dense, default=True)
    return layout


def _normalize_section_layout(raw_layout) -> Optional[dict]:
    if not isinstance(raw_layout, dict):
        return None
    span = _coerce_number(raw_layout.get("span") or raw_layout.get("colSpan"))
    tone = raw_layout.get("tone")
    layout = {}
    if span is not None and span > 0:
        layout["span"] = int(max(1, span))
    if isinstance(tone, str) and tone.strip():
        normalized = tone.strip().lower()
        if normalized in {"accent", "soft"}:
            layout["tone"] = normalized
    return layout or None


def _extract_media_slots(layout_config: dict) -> List[dict]:
    if not isinstance(layout_config, dict):
        return []
    sections = layout_config.get("sections")
    if not isinstance(sections, list):
        return []
    for section in sections:
        if not isinstance(section, dict):
            continue
        components = section.get("components")
        if not isinstance(components, list):
            continue
        for comp in components:
            if isinstance(comp, dict) and comp.get("type") == "media-uploader":
                slots = comp.get("slots")
                if isinstance(slots, list) and slots:
                    return slots
    return []


def _normalize_slots_payload(obj, base_prompt: str, selected_options: dict, fallback_slots: List[dict]) -> List[dict]:
    raw_slots = None
    if isinstance(obj, dict):
        raw_slots = obj.get("slots")
        if raw_slots is None:
            raw_slots = obj.get("media_slots")
    selected_style = _pick_selected_first(selected_options, ["风格", "风格方向"])
    style_default = selected_style or _infer_style(base_prompt)
    selected_moods = _pick_selected_list(selected_options, ["氛围", "情绪氛围", "情绪", "氛围关键词"])
    mood_defaults = selected_moods or _infer_moods(base_prompt)

    slots = []
    if isinstance(raw_slots, list) and raw_slots:
        slots = _sanitize_slots(raw_slots, base_prompt, style_default, mood_defaults)
    if not slots and fallback_slots:
        slots = _sanitize_slots(fallback_slots, base_prompt, style_default, mood_defaults)
    return slots


def _apply_slots_to_layout(layout_config: dict, slots: List[dict]) -> dict:
    if not isinstance(layout_config, dict):
        return layout_config
    if not slots:
        return layout_config
    layout_copy = json.loads(json.dumps(layout_config))
    sections = layout_copy.get("sections")
    if not isinstance(sections, list):
        return layout_copy
    applied = False
    for section in sections:
        if not isinstance(section, dict):
            continue
        components = section.get("components")
        if not isinstance(components, list):
            continue
        for comp in components:
            if isinstance(comp, dict) and comp.get("type") == "media-uploader":
                comp["slots"] = slots
                applied = True
    if not applied:
        sections.append(
            {
                "id": "materials",
                "title": "素材准备",
                "layout": {"span": 2},
                "components": [
                    {
                        "id": "materials-uploader",
                        "type": "media-uploader",
                        "title": "上传或生成素材",
                        "required": True,
                        "max": max(1, len(slots)),
                        "slots": slots,
                    }
                ],
            }
        )
    return layout_copy


def _sanitize_slots(raw_slots, base_prompt: str, style: str, moods: List[str]) -> List[dict]:
    if not isinstance(raw_slots, list):
        return []
    slots = []
    used_ids: set = set()
    slot_prompt = _compose_slot_prompt(base_prompt, style, moods)
    for idx, slot in enumerate(raw_slots):
        if not isinstance(slot, dict):
            continue
        slot_id = _ensure_unique_id(slot.get("id") or f"slot-{idx + 1}", used_ids, "slot")
        label = slot.get("label")
        label = label.strip() if isinstance(label, str) and label.strip() else f"素材 {idx + 1}"
        prompt = slot.get("prompt")
        prompt = prompt.strip() if isinstance(prompt, str) and prompt.strip() else slot_prompt
        layer_type = _normalize_layer_type(slot.get("layerType"), label)
        slots.append(
            {
                "id": str(slot_id),
                "label": label,
                "layerType": layer_type,
                "prompt": prompt,
            }
        )
    if slots and not any(slot["layerType"] == "background" for slot in slots):
        background_id = _ensure_unique_id("background", used_ids, "slot")
        slots.insert(
            0,
            {
                "id": background_id,
                "label": "背景",
                "layerType": "background",
                "prompt": f"{slot_prompt}，背景",
            },
        )
    return slots


def _sanitize_components(raw_components, base_prompt: str, style: str, moods: List[str], used_ids: Optional[set] = None) -> List[dict]:
    if not isinstance(raw_components, list):
        return []
    allowed = {
        "text-input",
        "textarea",
        "select",
        "multi-select",
        "media-uploader",
        "number-input",
        "slider",
        "toggle",
        "ratio-select",
        "color-palette",
        "prompt-editor",
    }
    components = []
    used_ids = used_ids or set()
    for idx, comp in enumerate(raw_components):
        if not isinstance(comp, dict):
            continue
        comp_type = comp.get("type")
        if comp_type not in allowed:
            continue
        comp_id = _ensure_unique_id(comp.get("id") or f"{comp_type}-{idx + 1}", used_ids, "component")
        base = {"id": comp_id, "type": comp_type}
        label = comp.get("label")
        if isinstance(label, str) and label.strip():
            base["label"] = label.strip()
        title = comp.get("title")
        if isinstance(title, str) and title.strip():
            base["title"] = title.strip()
        helper_text = comp.get("helperText")
        if isinstance(helper_text, str) and helper_text.strip():
            base["helperText"] = helper_text.strip()

        if comp_type in ("text-input", "textarea"):
            placeholder = comp.get("placeholder")
            if isinstance(placeholder, str) and placeholder.strip():
                base["placeholder"] = placeholder.strip()
            default = comp.get("default")
            if default is not None:
                base["default"] = str(default)
        elif comp_type in ("select", "multi-select", "ratio-select"):
            options = _coerce_options(comp.get("options"))
            if not options:
                continue
            base["options"] = options[:8]
            display = comp.get("display")
            if isinstance(display, str) and display.strip():
                display_value = display.strip().lower()
                if display_value in {"chips"}:
                    base["display"] = display_value
            default = comp.get("default")
            if comp_type in ("select", "ratio-select"):
                if isinstance(default, str) and default in options:
                    base["default"] = default
                else:
                    base["default"] = options[0]
            else:
                if isinstance(default, list):
                    selected = [str(v).strip() for v in default if str(v).strip() and str(v).strip() in options]
                    base["default"] = selected
        elif comp_type == "color-palette":
            options = _coerce_palette_options(comp.get("options"))
            if not options:
                continue
            base["options"] = options
            allow_multiple = _coerce_bool(comp.get("allowMultiple"), default=False)
            if allow_multiple:
                base["allowMultiple"] = True
            default = comp.get("default")
            if allow_multiple:
                if isinstance(default, list):
                    base["default"] = [str(v).strip() for v in default if str(v).strip()]
            else:
                if isinstance(default, str):
                    base["default"] = default
                else:
                    base["default"] = options[0].get("value")
        elif comp_type in ("number-input", "slider"):
            range_config = _normalize_number_range(
                comp.get("min"),
                comp.get("max"),
                comp.get("step"),
                comp.get("default"),
            )
            base.update(range_config)
            unit = comp.get("unit")
            if isinstance(unit, str) and unit.strip():
                base["unit"] = unit.strip()
        elif comp_type == "toggle":
            base["default"] = _coerce_bool(comp.get("default"), default=False)
        elif comp_type == "prompt-editor":
            fields = _sanitize_prompt_fields(comp.get("fields"), base_prompt)
            if not fields:
                continue
            base["fields"] = fields
        elif comp_type == "media-uploader":
            slots = _sanitize_slots(comp.get("slots"), base_prompt, style, moods)
            if not slots:
                continue
            base["slots"] = slots
            max_val = _coerce_number(comp.get("max"))
            if isinstance(max_val, float) and max_val > 0:
                base["max"] = int(max_val)
            if _coerce_bool(comp.get("required"), default=False):
                base["required"] = True
        components.append(base)
    return components


def _normalize_layout_payload(obj: dict, fallback: dict, base_prompt: str, selected_options: dict) -> dict:
    if not isinstance(obj, dict):
        return fallback
    text = obj.get("text_response")
    if not isinstance(text, str) or not text.strip():
        text = fallback.get("text_response", "")
    layout = obj.get("layout_config")
    if not isinstance(layout, dict):
        return fallback
    sections = layout.get("sections")
    if not isinstance(sections, list) or not sections:
        return fallback

    selected_style = _pick_selected_first(selected_options, ["风格", "风格方向"])
    style_default = selected_style or _infer_style(base_prompt)
    selected_moods = _pick_selected_list(selected_options, ["氛围", "情绪氛围", "情绪", "氛围关键词"])
    mood_defaults = selected_moods or _infer_moods(base_prompt)

    sanitized_sections = []
    has_media = False
    used_section_ids: set = set()
    used_component_ids: set = set()
    ratio_component_ids: List[str] = []
    for idx, section in enumerate(sections):
        if not isinstance(section, dict):
            continue
        section_id = _ensure_unique_id(section.get("id") or f"section-{idx + 1}", used_section_ids, "section")
        title = section.get("title")
        title = title.strip() if isinstance(title, str) and title.strip() else None
        description = section.get("description")
        description = description.strip() if isinstance(description, str) and description.strip() else None
        components = _sanitize_components(section.get("components"), base_prompt, style_default, mood_defaults, used_ids=used_component_ids)
        if not components:
            continue
        if any(comp.get("type") == "media-uploader" for comp in components):
            has_media = True
        ratio_component_ids.extend([comp["id"] for comp in components if comp.get("type") == "ratio-select"])
        entry = {"id": str(section_id), "components": components}
        if title:
            entry["title"] = title
        if description:
            entry["description"] = description
        section_layout = _normalize_section_layout(section.get("layout"))
        if section_layout:
            entry["layout"] = section_layout
        sanitized_sections.append(entry)

    if not sanitized_sections or not has_media:
        return fallback

    meta = layout.get("meta") if isinstance(layout.get("meta"), dict) else {}
    meta.setdefault("sourcePrompt", base_prompt)
    meta["schema_version"] = LAYOUT_SCHEMA_VERSION
    if not isinstance(meta.get("summary"), str) or not meta.get("summary", "").strip():
        meta["summary"] = fallback.get("layout_config", {}).get("meta", {}).get("summary", "")
    aspect = meta.get("aspect_ratio")
    if not isinstance(aspect, str) or not aspect.strip():
        meta["aspect_ratio"] = _infer_aspect_ratio(base_prompt, selected_options)
    aspect_field = meta.get("aspect_ratio_field")
    if isinstance(aspect_field, str) and aspect_field in used_component_ids:
        meta["aspect_ratio_field"] = aspect_field
    elif ratio_component_ids:
        meta["aspect_ratio_field"] = ratio_component_ids[0]
    else:
        meta.pop("aspect_ratio_field", None)
    layout_meta = _normalize_layout_meta(meta.get("layout"))
    if not layout_meta:
        fallback_layout = fallback.get("layout_config", {}).get("meta", {}).get("layout")
        layout_meta = _normalize_layout_meta(fallback_layout)
    if layout_meta:
        meta["layout"] = layout_meta
    return {"text_response": text.strip(), "layout_config": {"meta": meta, "sections": sanitized_sections}}


def generate_llm_layout(req: LayoutRequest) -> dict:
    fallback = _build_rule_layout_payload(req.prompt, req.chat_history, req.selected_options)
    if not _siliconflow_enabled():
        fallback["text_response"] += "（当前未配置 SILICONFLOW_API_KEY，使用规则兜底布局）"
        return fallback

    layout_system = (
        "你是 AIGC 配置页生成器。请根据用户需求生成“专属编辑布局”。\n"
        "输出必须是单个 JSON 对象，仅包含字段：text_response(string), layout_config(object)。\n"
        f"layout_config.meta 包含 sourcePrompt, summary, aspect_ratio, schema_version(固定为 {LAYOUT_SCHEMA_VERSION})，可选 aspect_ratio_field 与 layout。\n"
        "meta.layout 可包含 columns/minCardWidth/gap/dense。\n"
        "layout_config.sections 为数组，每项包含 id, title, components，可选 description 与 layout(span,tone)。tone 仅可为 accent/soft。\n"
        "components 的 type 仅可为：text-input, textarea, select, multi-select, media-uploader, number-input, slider, toggle, ratio-select, color-palette, prompt-editor。\n"
        "select/multi-select/ratio-select 需要 options(string[])，可选 display:'chips' 以使用按钮样式。\n"
        "number-input/slider 可包含 min/max/step/default/unit/helperText。\n"
        "toggle 可包含 default(boolean)/helperText。\n"
        "color-palette 需要 options，支持 string 或 {value,label,color}。\n"
        "prompt-editor 需要 fields，fields 为 {id,label,placeholder,default,helperText}。\n"
        "media-uploader 组件可以先不填 slots 或仅给出占位，slots 会在第二步生成。\n"
        "建议输出 4-6 个 sections，使用 3 列布局（columns=3, minCardWidth≈280, gap=2），并给出不同的 span 以形成仪表盘布局。\n"
        "请使用中文，不要包含多余解释或 Markdown。"
    )
    slots_system = (
        "你是 AIGC 图层规划器。请根据用户需求返回生图图层拆分。\n"
        "输出必须是单个 JSON 对象，仅包含字段：slots(array)。\n"
        "slots 中每一项为 {id,label,layerType,prompt}，layerType 只能是 background/subject/decor。\n"
        "必须包含一个 background，至少一个 subject；可根据需求补充 decor。\n"
        "prompt 需简洁描述该层要生成的内容。请使用中文。"
    )
    selected_json = json.dumps(req.selected_options or {}, ensure_ascii=False)
    user = (
        f"用户需求：{req.prompt}\n"
        f"对话记录：{req.chat_history or ''}\n"
        f"已选表单项：{selected_json}\n"
        "请据此生成布局。"
    )
    try:
        layout_content = _call_siliconflow_chat(
            [
                {"role": "system", "content": layout_system},
                {"role": "user", "content": user},
            ]
        )
        layout_obj = _safe_parse_json_object(layout_content)
    except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, ValueError) as e:
        layout_obj = None
        fallback["text_response"] += f"（布局生成失败：{e}，使用规则兜底布局）"
    except Exception as e:
        layout_obj = None
        fallback["text_response"] += f"（布局生成未知错误：{e}，使用规则兜底布局）"

    try:
        slots_content = _call_siliconflow_chat(
            [
                {"role": "system", "content": slots_system},
                {"role": "user", "content": user},
            ]
        )
        slots_obj = _safe_parse_json_object(slots_content)
    except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError, ValueError) as e:
        slots_obj = None
        fallback["text_response"] += f"（图层拆分失败：{e}，使用规则兜底图层）"
    except Exception as e:
        slots_obj = None
        fallback["text_response"] += f"（图层拆分未知错误：{e}，使用规则兜底图层）"

    layout_config = None
    text_response = None
    if isinstance(layout_obj, dict):
        text_response = layout_obj.get("text_response")
        layout_config = layout_obj.get("layout_config")
    if not isinstance(text_response, str) or not text_response.strip():
        text_response = fallback.get("text_response", "")
    if not isinstance(layout_config, dict):
        layout_config = fallback.get("layout_config", {})

    fallback_slots = _extract_media_slots(fallback.get("layout_config", {}))
    slots = _normalize_slots_payload(slots_obj, req.prompt, req.selected_options or {}, fallback_slots)
    layout_with_slots = _apply_slots_to_layout(layout_config, slots)
    combined = {"text_response": text_response, "layout_config": layout_with_slots}
    return _normalize_layout_payload(combined, fallback, req.prompt, req.selected_options or {})

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

def _build_placeholder_image(prompt: str, width: int = 512, height: int = 512) -> str:
    """Generate a lightweight placeholder image (base64) when diffusion is unavailable."""
    img = Image.new("RGB", (width, height), color=(240, 243, 247))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = (prompt or "LiteDraw").strip()[:80]
    lines = [text[i:i+20] for i in range(0, len(text), 20)]
    y = height // 2 - (len(lines) * 12)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.text(((width - w) / 2, y), line, fill=(60, 60, 70), font=font)
        y += h + 6
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _parse_aspect_ratio(ratio_str: str) -> float:
    try:
        if ":" in ratio_str:
            w, h = ratio_str.split(":")
            return float(w) / float(h)
        return float(ratio_str)
    except Exception:
        return 3 / 4


def _create_layer(name: str, layer_type: str, prompt: str, width: int, height: int, x: int, y: int, z_index: int):
    img_b64 = _build_placeholder_image(f"{layer_type}:{prompt}", width=width, height=height)
    return {
        "id": f"{layer_type}-{name}",
        "name": name,
        "layer_type": layer_type,
        "image_base64": img_b64,
        "width": width,
        "height": height,
        "placement": {"x": x, "y": y, "z_index": z_index},
    }

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


@app.post("/generate-text")
def generate_text(req: TextRequest):
    """Lightweight placeholder: echo prompt into positive prompt."""
    if not req.prompt:
        return JSONResponse(status_code=400, content={"error": "prompt 不能为空"})
    response = {
        "text_response": f"收到你的想法：{req.prompt}",
        "positive": req.prompt.strip(),
        "negative": "",
    }
    return response


@app.post("/generate-layout")
def generate_layout(req: LayoutRequest):
    """Return a layout configuration for the editor step (LLM or rule-based)."""
    if not req.prompt:
        return JSONResponse(status_code=400, content={"error": "prompt 不能为空"})
    return generate_llm_layout(req)


@app.post("/remove-background")
def remove_background(payload: dict):
    """Placeholder background removal; returns the original image as transparent PNG base64."""
    image_b64 = payload.get("image_base64")
    if not image_b64:
        return JSONResponse(status_code=400, content={"error": "缺少 image_base64"})
    try:
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        image_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        return {"image_base64": base64.b64encode(buffer.getvalue()).decode("utf-8")}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"抠图失败: {e}"})


@app.post("/generate-image")
def generate_image(req: ImageRequest):
    """Generate an image and return base64 PNG."""
    if not req.prompt:
        return JSONResponse(status_code=400, content={"error": "prompt 不能为空"})
    width = max(256, min(req.width or 512, 2048))
    height = max(256, min(req.height or 512, 2048))
    # Diffusers requires multiples of 8.
    width = max(256, (width // 8) * 8)
    height = max(256, (height // 8) * 8)
    steps = max(10, min(req.num_inference_steps or 30, 100))

    pipe = None
    lora_models: List[str] = []
    try:
        provider = _resolve_image_provider(req.image_provider)
        if provider == "siliconflow":
            image_b64 = _call_siliconflow_image(req.prompt, width, height, steps)
            return {"image_base64": image_b64}

        base_model_id = req.base_model or BASE_MODEL_DEFAULT
        lora_models = req.lora_models or []
        lora_models = [name for name in lora_models if name and name != "None"]
        lora_models = list(dict.fromkeys(lora_models))

        with _pipe_lock:
            pipe = _ensure_pipe(base_model_id)

            adapter_names = []
            supports_multi = hasattr(pipe, "set_adapters")
            if lora_models:
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
                if supports_multi and adapter_names:
                    pipe.set_adapters(adapter_names, adapter_weights=[1.0] * len(adapter_names))

        with torch.inference_mode():
            image = pipe(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_inference_steps=steps,
                guidance_scale=req.guidance_scale or 7.5,
                width=width,
                height=height,
            ).images[0]

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        image_b64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
        return {"image_base64": image_b64}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if pipe is not None and lora_models:
            with _pipe_lock:
                try:
                    pipe.unload_lora_weights()
                except Exception:
                    pass


@app.post("/generate-layered")
def generate_layered(req: LayeredRequest):
    """Return placeholder background + subject layers with simple layout metadata."""
    if not req.prompt:
        return JSONResponse(status_code=400, content={"error": "prompt 不能为空"})

    ratio = _parse_aspect_ratio(req.aspect_ratio or "3:4")
    base_w = 768
    bg_w = base_w
    bg_h = int(base_w / ratio)

    layers = []
    # background
    bg = _create_layer("背景", "background", req.prompt, bg_w, bg_h, x=0, y=0, z_index=0)
    layers.append(bg)

    subjects = max(1, (req.count or 3) - 1)
    positions = [
        (int(bg_w * 0.2), int(bg_h * 0.3)),
        (int(bg_w * 0.55), int(bg_h * 0.35)),
        (int(bg_w * 0.35), int(bg_h * 0.65)),
        (int(bg_w * 0.6), int(bg_h * 0.6)),
    ]
    for idx in range(subjects):
        x, y = positions[idx % len(positions)]
        layer = _create_layer(f"主体{idx+1}", "subject", req.prompt, 512, 512, x=x, y=y, z_index=idx + 1)
        layers.append(layer)

    return {
        "aspect_ratio": req.aspect_ratio,
        "layers": layers,
    }


@app.post("/compose-materials")
def compose_materials(payload: dict):
    """Auto-layout given materials with simple stacking."""
    materials = payload.get("materials") or []
    aspect_ratio = payload.get("aspect_ratio") or "3:4"
    ratio = _parse_aspect_ratio(aspect_ratio)
    base_w = 768
    bg_w = base_w
    bg_h = int(base_w / ratio)

    layout_layers = []
    positions = [
        (int(bg_w * 0.2), int(bg_h * 0.25)),
        (int(bg_w * 0.55), int(bg_h * 0.25)),
        (int(bg_w * 0.2), int(bg_h * 0.6)),
        (int(bg_w * 0.55), int(bg_h * 0.6)),
    ]
    for idx, mat in enumerate(materials):
        name = mat.get("name") or f"素材{idx+1}"
        layer_type = mat.get("layer_type") or "subject"
        img_b64 = mat.get("image_base64") or _build_placeholder_image(name, 512, 512)
        x, y = positions[idx % len(positions)]
        layer = {
            "id": f"compose-{idx}",
            "name": name,
            "layer_type": layer_type,
            "image_base64": img_b64,
            "width": 512,
            "height": 512,
            "placement": {"x": x, "y": y, "z_index": idx + 1},
        }
        layout_layers.append(layer)

    return {
        "aspect_ratio": aspect_ratio,
        "layers": layout_layers,
    }


@app.post("/analyze-layer-plan")
def analyze_layer_plan(payload: dict):
    """Placeholder for LiteDraw layer plan analysis."""
    chat_history = payload.get("chat_history") or ""
    selected = payload.get("selected_options") or {}
    ui_state = payload.get("ui_state") or {}
    prompt_hint = (chat_history.splitlines()[-1] if chat_history else "") or "scene"
    plan = {
        "estimated_layers": 3,
        "estimated_time_seconds": 10,
        "layer_plan": [
            {
                "layer_id": "background",
                "layer_name": "背景",
                "layer_type": "background",
                "enabled": True,
                "order": 0,
                "prompt": prompt_hint,
                "placement": {"x": 0.0, "y": 0.0},
                "generation_params": {"needs_transparent_bg": False},
            },
            {
                "layer_id": "subject-1",
                "layer_name": "主体 1",
                "layer_type": "subject",
                "enabled": True,
                "order": 1,
                "prompt": prompt_hint,
                "placement": {"x": 0.35, "y": 0.45},
                "generation_params": {"needs_transparent_bg": True},
            },
            {
                "layer_id": "subject-2",
                "layer_name": "主体 2",
                "layer_type": "subject",
                "enabled": True,
                "order": 2,
                "prompt": prompt_hint,
                "placement": {"x": 0.65, "y": 0.55},
                "generation_params": {"needs_transparent_bg": True},
            },
        ],
        "meta": {
            "selected_options": selected,
            "ui_state": ui_state,
        },
    }
    return plan

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
