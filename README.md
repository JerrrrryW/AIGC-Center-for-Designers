# AIGC 中心

## 项目概述

本项目是一个基于 Web 的平台，旨在实现本地训练 LoRA（低秩自适应）模型、运行 Stable Diffusion 推理，并在 M1 阶段提供简化的“AI 场景生成”体验。主要任务包括：

1. **LoRA 训练**：上传图像、配置训练参数并运行训练流程，无需直接使用命令行。
2. **Stable Diffusion 推理**：提供简洁的界面，通过文本提示词生成图像，并支持加载已训练的 LoRA 模型。
3. **模型管理**：提供专用页面，用于查看、下载和删除已训练的 LoRA 模型。
4. **AI 场景生成（M1/M2）**：参考 LiteDraw 的 Chat→Edit→Preview 流程：先通过聊天与选择题澄清需求，再生成编辑界面并准备素材，最后将背景、主体、装饰、文字与色块图层落到 Web 画布（Canvas）进行预览与导出。

### 系统架构

应用程序采用客户端-服务器架构：

* **前端**：基于 **React (使用 Vite)** 和 **TypeScript** 构建的单页应用。使用 **Material-UI (MUI)** 作为组件库，并使用 **React Router** 处理页面导航。前端现集成了训练、推理、模型管理、画布以及新的“AI 场景生成（M1/M2）”页面，采用统一的侧边栏导航。
* **后端**：基于 **FastAPI** 构建的 Python 服务器，向前端暴露 REST API。
* LoRA 训练：在 `/train` 后台执行，生成元数据与缩略图，若训练失败会自动清理空目录。
* 推理：`/generate` 接口支持基础模型与多 LoRA 组合，按需加载管线。
* 模型管理：通过 `/models` 接口进行列表展示、下载和删除操作。
* 场景生成（M1）：新增 `/generate-text`、`/generate-layout`、`/generate-image`、`/remove-background`、`/analyze-layer-plan` 和 `/compose-materials` 接口，为桌面端工作流提供表单建议、图层规划、透明底生成、抠图、排版预览与落画布数据。



### 核心技术栈

* **前端**：React, Vite, TypeScript, Material-UI, Axios, React Router
* **后端**：Python, FastAPI, PyTorch (利用 MPS 进行 GPU 加速), `diffusers`, `peft`, `accelerate`, `transformers`

---

## 构建与运行

本项目需要同时运行两个独立的进程：后端服务器和前端开发服务器。

### 1. 后端设置

后端依赖于特定的 Conda 环境和一系列 Python 软件包。

* **环境要求**：需要一个名为 `aigc` 的 Conda 环境。
* **安装依赖**：所有 Python 依赖项均列在 `requirements.txt` 中。您可以通过以下命令安装：
```bash
pip install -r requirements.txt

```


* **启动服务器**：
在项目根目录（`AIGC-Training`）下执行以下命令：
```bash
conda run -n aigc uvicorn Server.main:app --reload --host 0.0.0.0 --port 8000

```



### 2. 前端设置

前端是一个标准的基于 Vite 的 React 应用程序。

* **安装依赖**：
进入 `frontend` 目录并运行：
```bash
cd frontend
npm install

```


* **启动开发服务器**：
在 `frontend` 目录下运行：
```bash
npm run dev -- --host 0.0.0.0

```


应用程序通常可以通过 `http://localhost:5173` 访问。

### 局域网访问（可选）

如需从局域网内的其他设备访问 UI：

1. 查找您 Mac 的局域网 IP（例如 `192.168.x.x`）并访问 `http://<您的-IP>:5173`。
2. 确保后端运行在 `--host 0.0.0.0` 模式下（如上所示）。
3. 如果您使用自定义的 API 基础 URL，请在执行 `npm run dev` 前设置 `VITE_API_BASE_URL`（例如 `http://<您的-IP>:8000`）。
4. 如果需要自定义 CORS 源，请在后端设置 `AIGC_CORS_ORIGINS`（逗号分隔）或 `AIGC_CORS_ORIGIN_REGEX`。

### LLM 配置（SiliconFlow / Gemini，可选）

“AI 场景生成 · Chat”支持通过 LLM 生成选择题表单，并支持多轮对话。如果未配置，会自动使用本地兜底表单/回复。
`/generate-image` 默认使用 SiliconFlow 生图 API；如需切回本地 Diffusers，可设置 `IMAGE_GENERATION_PROVIDER=local`（或 `auto` 自动选择）。该接口现支持 `layer_type`、`transparent_background`、`remove_background` 参数，用于非背景层透明底生成与后处理。

* **SiliconFlow（OpenAI 兼容）**：
  * **必需**：`SILICONFLOW_API_KEY`
  * **可选（聊天）**：
    * `SILICONFLOW_BASE_URL`（默认：`https://api.siliconflow.cn/v1/chat/completions`）
    * `SILICONFLOW_MODEL`（默认：`Qwen/Qwen3-Next-80B-A3B-Instruct`）
  * **可选（生图）**：
    * `IMAGE_GENERATION_PROVIDER`（`siliconflow`/`local`/`auto`，默认：有 `SILICONFLOW_API_KEY` 时为 `siliconflow`）
    * `SILICONFLOW_IMAGE_BASE_URL`（默认：`https://api.siliconflow.cn/v1`）
    * `SILICONFLOW_IMAGE_MODEL`（默认：`Qwen/Qwen-Image`）

* **Gemini（Google Generative Language API）**：
  * **必需**：`GEMINI_API_KEY`
  * **可选**：
    * `GEMINI_BASE_URL`（例如：`https://generativelanguage.googleapis.com/v1beta`，也可填写完整 `.../models/<model>:generateContent`）
    * `GEMINI_MODEL`（例如：`gemini-1.5-pro`）

也可以在前端侧边栏的 “LLM 配置” 中填写 API Key / Base URL / Model。Gemini 仅用于文本对话与表单/布局生成，生图仍需 SiliconFlow 或本地 Diffusers。

---

## 开发规范

* **后端 API**：
* 主 FastAPI 应用程序位于 `Server/main.py`。
* 耗时较长的任务（训练、推理）使用 `BackgroundTasks` 在后台线程中执行。
* `/generate` 接口支持 `lora_model` 参数，以便在推理时应用训练好的 LoRA 模型。
* 新增了 `/models`、`/models/download/{model_name}` 和 `/models/delete/{model_name}` 接口用于模型管理。
* 场景生成（LiteDraw-like）相关接口：`/generate-json`、`/chat`、`/generate-layout`、`/generate-image`、`/remove-background`、`/analyze-layer-plan`、`/compose-materials`。
* `/generate-image` 可根据 `layer_type` 和 `transparent_background` 对非背景层做透明底后处理。
* `/remove-background` 现为启发式软边抠图，不再原样返回原图。
* `/analyze-layer-plan` 会根据对话、表单选项与画幅推断背景、主体、装饰和文字安全区。
* `/compose-materials` 会根据画幅做自适应排版，并返回 `image`、`text`、`shape` 三类图层供画布导入。
* 系统现支持自动清理空模型目录，以防止失败的训练任务占用空间。


* **前端 UI**：
* 主要应用组件为 `frontend/src/App.tsx`，现已包含所有页面的路由配置。
* `Layout.tsx` 组件提供了统一的侧边栏和应用栏。
* 训练界面位于 `frontend/src/components/TrainingPage.tsx`。
* 推理界面位于 `frontend/src/components/InferencePage.tsx`，现包含一个用于选择已训练 LoRA 模型的下拉菜单。
* 模型管理界面位于 `frontend/src/components/ModelsPage.tsx`。
* 当前主要的“AI 场景生成（M1/M2）”入口为 `/scene`，核心页面为 `frontend/src/components/scene/SceneFlowPage.tsx`，将聊天、意图板、编辑配置、素材准备与预览整合到单页工作流中。
* 画布页面位于 `frontend/src/components/CanvasPage.tsx`，现支持 `image`、`text`、`shape` 三类图层的导入、拖拽、缩放和导出 PNG。


* **模型与数据存储**：
* 上传用于训练的图像临时存储在 `temp_training_images` 目录中。
* 训练完成的 LoRA 模型保存至 `lora_models` 目录。每次训练运行都会生成一个带有时间戳的子文件夹。

---

## 手动验证

部署启动命令无需调整，仍按上文分别启动后端与前端。

建议按以下路径做最小验收：

1. 打开 `http://localhost:5173/scene`。
2. 在聊天区输入一个明确场景需求，点击“生成编辑界面”。
3. 确认页面出现编辑配置、图层规划 JSON，以及长时间操作时的加载遮罩。
4. 在素材区分别生成背景层与至少一个主体层。
5. 确认主体层会显示“建议透明底”或“透明底”，并可点击“抠图”。
6. 填写标题、副标题或其他文本字段，并选择一个主色调。
7. 确认右侧“实时预览”会随着素材和文案变化自动同步，并在同步中保留最近一次成功结果。
8. 点击“进入画布”，确认背景、图片层、文字层和色块层都已导入。
9. 在画布中拖动、缩放若干图层。
10. 导出 PNG，确认导出结果包含全部图层。

已知环境问题：

* 如果本机 `npm run build` 无法启动且报 `libicui18n.*.dylib` 缺失，这通常是本机 Node 运行时依赖缺失，不是前端代码本身错误。
