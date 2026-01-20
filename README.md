# AIGC 中心

## 项目概述

本项目是一个基于 Web 的平台，旨在实现本地训练 LoRA（低秩自适应）模型、运行 Stable Diffusion 推理，并在 M1 阶段提供简化的“AI 场景生成”体验。主要任务包括：

1. **LoRA 训练**：上传图像、配置训练参数并运行训练流程，无需直接使用命令行。
2. **Stable Diffusion 推理**：提供简洁的界面，通过文本提示词生成图像，并支持加载已训练的 LoRA 模型。
3. **模型管理**：提供专用页面，用于查看、下载和删除已训练的 LoRA 模型。
4. **AI 场景生成（M1/M2）**：参考 LiteDraw 的 Chat→Edit→Preview 流程：先通过聊天与选择题澄清需求，再生成编辑界面并准备素材，最后将图层落到 Web 画布（Canvas）进行预览与导出（当前为占位生成）。

### 系统架构

应用程序采用客户端-服务器架构：

* **前端**：基于 **React (使用 Vite)** 和 **TypeScript** 构建的单页应用。使用 **Material-UI (MUI)** 作为组件库，并使用 **React Router** 处理页面导航。前端现集成了训练、推理、模型管理、画布以及新的“AI 场景生成（M1/M2）”页面，采用统一的侧边栏导航。
* **后端**：基于 **FastAPI** 构建的 Python 服务器，向前端暴露 REST API。
* LoRA 训练：在 `/train` 后台执行，生成元数据与缩略图，若训练失败会自动清理空目录。
* 推理：`/generate` 接口支持基础模型与多 LoRA 组合，按需加载管线。
* 模型管理：通过 `/models` 接口进行列表展示、下载和删除操作。
* 场景生成（M1）：新增 `/generate-text`、`/generate-layout`、`/generate-image`（占位图）和 `/remove-background` 接口，为桌面端工作流提供提示词、表单建议与占位图。



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

### SiliconFlow（可选：用于 LLM 生成表单/多轮对话）

“AI 场景生成 · Chat”支持通过 SiliconFlow（OpenAI 兼容的 Chat Completions）生成选择题表单，并支持多轮对话。如果未配置，会自动使用本地兜底表单/回复。

* **必需**：`SILICONFLOW_API_KEY`
* **可选**：
  * `SILICONFLOW_BASE_URL`（默认：`https://api.siliconflow.cn/v1/chat/completions`）
  * `SILICONFLOW_MODEL`（默认：`Qwen/Qwen2.5-7B-Instruct`）

---

## 开发规范

* **后端 API**：
* 主 FastAPI 应用程序位于 `Server/main.py`。
* 耗时较长的任务（训练、推理）使用 `BackgroundTasks` 在后台线程中执行。
* `/generate` 接口支持 `lora_model` 参数，以便在推理时应用训练好的 LoRA 模型。
* 新增了 `/models`、`/models/download/{model_name}` 和 `/models/delete/{model_name}` 接口用于模型管理。
* 场景生成（LiteDraw-like）相关接口：`/generate-json`、`/chat`、`/generate-layout`、`/analyze-layer-plan`、`/compose-materials`。
* 系统现支持自动清理空模型目录，以防止失败的训练任务占用空间。


* **前端 UI**：
* 主要应用组件为 `frontend/src/App.tsx`，现已包含所有页面的路由配置。
* `Layout.tsx` 组件提供了统一的侧边栏和应用栏。
* 训练界面位于 `frontend/src/components/TrainingPage.tsx`。
* 推理界面位于 `frontend/src/components/InferencePage.tsx`，现包含一个用于选择已训练 LoRA 模型的下拉菜单。
* 模型管理界面位于 `frontend/src/components/ModelsPage.tsx`。
* 新的“AI 场景生成（M1/M2）”路由为 `/scene/chat`（聊天+表单）与 `/scene/edit`（编辑+素材），核心页面在 `frontend/src/components/scene/SceneChatPage.tsx` 与 `frontend/src/components/scene/SceneEditPage.tsx`；旧的快速页保留在 `/scene/quick`（`frontend/src/components/SceneGenerationPage.tsx`）。


* **模型与数据存储**：
* 上传用于训练的图像临时存储在 `temp_training_images` 目录中。
* 训练完成的 LoRA 模型保存至 `lora_models` 目录。每次训练运行都会生成一个带有时间戳的子文件夹。
