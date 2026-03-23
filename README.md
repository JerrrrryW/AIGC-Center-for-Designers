# AIGC 中心

一个面向设计场景的本地 Web 工具。它可以帮助您用对话和表单快速生成场景方案、准备素材、做简单排版预览，并把结果放进画布继续微调后导出 PNG。

## 这个工具能做什么

从“想法”到“出图”的完整工作流：

1. **描述需求**：先用聊天或选择题明确场景需求。
2. **生成素材**：生成背景、主体、装饰等图像素材。
3. **整理版式**：自动生成一个基础排版预览。
4. **进入画布**：把图片、文字和色块放进画布继续调整。
5. **导出结果**：导出 PNG 给评审、汇报或继续设计。

除了AI场景生成外，系统也支持：

* **LoRA 训练**：上传图像并训练 LoRA 模型。
* **Stable Diffusion 推理**：输入提示词生成图片，并支持加载已训练的 LoRA。
* **模型管理**：查看、下载和删除已训练的 LoRA 模型。

## 适合谁

适合：

* 想快速把创意变成视觉草图的设计师
* 需要先出一版场景稿，再进入精修流程的团队
* 需要在本地运行、希望保留更多素材控制权的用户

不太适合：

* 完全不接触终端、也没有同事能协助完成首次环境安装的用户

> 这个项目目前还不是“下载即用”的桌面软件，第一次启动仍然需要运行几条命令。

## 第一次使用前，需要准备什么

最低要求：

* 一台可用的 Mac
* 能打开“终端（Terminal）”
* 已安装项目运行所需环境：Conda、Python 依赖、Node.js 和 npm
* 前端建议使用 `Node.js 20.19+` 或 `22.12+`

如果这是您第一次配置环境，可以从这里开始一步一步操作。

### 第 0 步：打开终端

在 Mac 上打开“终端（Terminal）”应用。

打开后，您会看到一个可以输入命令的窗口。

### 第 1 步：进入项目文件夹

在终端里输入：

```bash
cd /Users/你的用户名/Code/AIGC-Center
```

如果您的项目不在这个位置，也可以把 `AIGC-Center` 文件夹直接拖进终端窗口，终端会自动补全路径。

### 第 2 步：检查 Node.js 和 npm 是否已经安装

在终端里输入：

```bash
node -v
npm -v
```

如果终端显示版本号，请确认：

* `Node.js` 版本为 `20.19+` 或 `22.12+`
* `npm` 能正常输出版本号

建议直接安装较新的 LTS 版本，避免因为版本过低导致前端无法启动。

如果提示 `command not found`：

* 先去 Node.js 官网安装最新版 LTS
* 安装完成后，关闭终端再重新打开一次
* 再重复执行上面的 `node -v` 和 `npm -v`

### 第 3 步：检查 Conda 是否已经安装

在终端里输入：

```bash
conda --version
```

如果终端显示版本号，说明已经安装好了。

如果提示 `command not found`：

* 先安装 Miniconda
* 安装完成后，关闭终端再重新打开一次
* 再执行一次 `conda --version`

### 第 4 步：创建项目需要的 Python 环境

在终端里输入：

```bash
conda create -n aigc python=3.10 -y
```

说明：

* 这条命令会创建一个名为 `aigc` 的环境
* 如果终端提示这个环境已经存在，可以直接进入下一步

### 第 5 步：进入 `aigc` 环境

在终端里输入：

```bash
conda activate aigc
```

如果命令执行成功，终端前面通常会多出 `(aigc)`。

### 第 6 步：安装后端依赖

确认您现在仍在项目根目录 `AIGC-Center` 下，然后输入：

```bash
pip install -r requirements.txt
```

这一步会安装后端需要的 Python 包。

### 第 7 步：安装前端依赖

在终端里输入：

```bash
cd frontend
npm install
```

这一步会安装页面需要的依赖。

### 第 8 步：配置完成

上面这些步骤通常只需要做一次。完成后，您日常使用时只需要按下面的启动步骤操作。

可选准备：

* 如果您想让“AI 场景生成”使用在线大模型能力，可以准备 `SiliconFlow` 或 `Gemini` 的 API Key
* 如果不配置，聊天和表单能力会回退到本地兜底逻辑，但效果会很差。

## 最短上手路径

这个项目需要同时启动两个窗口：

* 一个窗口运行后端服务
* 一个窗口运行前端页面

### 第 1 步：启动后端

在项目根目录 `AIGC-Center` 下运行：

```bash
pip install -r requirements.txt
conda run -n aigc uvicorn Server.main:app --reload --host 0.0.0.0 --port 8000
```

说明：

* `pip install -r requirements.txt` 只需要在第一次安装依赖时执行
* `conda run -n aigc ...` 是启动后端服务的命令
* 如果终端里持续显示服务日志、没有立刻退出，通常就表示后端已启动

环境要求：

* 需要一个名为 `aigc` 的 Conda 环境

### 第 2 步：启动前端

新开一个终端窗口，进入 `frontend` 目录并运行：

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

说明：

* `npm install` 只需要在第一次安装依赖时执行
* `npm run dev -- --host 0.0.0.0` 是启动前端页面的命令
* 启动成功后，终端通常会显示一个本地访问地址

### 第 3 步：打开页面

浏览器访问：

```text
http://localhost:5173/scene
```

如果页面能正常打开，您就可以直接开始使用“AI 场景生成”流程。

## 建议的首次体验流程

建议第一次先只体验 `/scene` 页面：

1. 打开 `http://localhost:5173/scene`。
2. 在聊天区输入一个明确需求，例如“做一张春季促销海报，主视觉是花束和礼盒，风格清新明亮”。
3. 点击“生成编辑界面”。
4. 等待系统生成编辑配置、图层规划和素材建议。
5. 先生成背景层，再生成至少一个主体层。
6. 如果主体素材需要透明底，点击“抠图”。
7. 填写标题、副标题等文字内容，并选择一个主色调。
8. 观察右侧实时预览是否同步变化。
9. 点击“进入画布”。
10. 在画布中拖动、缩放图层，最后导出 PNG。

如果您只想确认系统是否“跑通了”，走完上面 10 步即可。

## 常见问题

### 1. 为什么要开两个终端窗口？

因为这个项目不是单一程序，而是：

* 一个后端服务负责生成、处理和管理数据
* 一个前端页面负责提供可视化界面

缺少任何一个，页面都无法完整工作。

### 2. 打开页面了，但功能没有反应，怎么办？

先检查两件事：

* 后端终端是否还在运行，没有报错退出
* 前端终端是否还在运行，并且访问的是它显示的地址

### 3. 没有配置 API Key 能不能用？

可以，但能力会受限：

* 文本对话和表单生成会回退到本地兜底逻辑
* 生图能力是否可用，取决于您是否配置了在线服务或本地 Diffusers

### 4. Node 版本需要多少？

当前前端使用的是 `Vite 7`，需要：

* `Node.js 20.19+`
* 或 `Node.js 22.12+`

如果版本过低，即使已经安装了 `node` 和 `npm`，前端也可能无法正常启动或构建。

### 5. `npm run build` 报 `libicui18n.*.dylib` 缺失怎么办？

这通常是本机 Node 运行时依赖缺失，不是前端代码本身错误。

## 可选配置

### LLM 配置（SiliconFlow / Gemini，可选）

“AI 场景生成 · Chat”支持通过大模型生成选择题表单，并支持多轮对话。

如果未配置，会自动使用本地兜底表单/回复。

`/generate-image` 默认使用 SiliconFlow 生图 API；如需切回本地 Diffusers，可设置：

```text
IMAGE_GENERATION_PROVIDER=local
```

也可以使用 `auto` 自动选择。

该接口还支持：

* `layer_type`
* `transparent_background`
* `remove_background`

这些参数主要用于非背景层透明底生成与后处理。

#### SiliconFlow（OpenAI 兼容）

必需：

* `SILICONFLOW_API_KEY`

可选（聊天）：

* `SILICONFLOW_BASE_URL`，默认：`https://api.siliconflow.cn/v1/chat/completions`
* `SILICONFLOW_MODEL`，默认：`Qwen/Qwen3-Next-80B-A3B-Instruct`

可选（生图）：

* `IMAGE_GENERATION_PROVIDER`，可选值：`siliconflow` / `local` / `auto`
* `SILICONFLOW_IMAGE_BASE_URL`，默认：`https://api.siliconflow.cn/v1`
* `SILICONFLOW_IMAGE_MODEL`，默认：`Qwen/Qwen-Image`

#### Gemini（Google Generative Language API）

必需：

* `GEMINI_API_KEY`

可选：

* `GEMINI_BASE_URL`，例如：`https://generativelanguage.googleapis.com/v1beta`
* `GEMINI_MODEL`，例如：`gemini-1.5-pro`

也可以在前端侧边栏的“LLM 配置”中填写 API Key / Base URL / Model。

注意：

* Gemini 仅用于文本对话与表单/布局生成
* 生图仍需 SiliconFlow 或本地 Diffusers

### 局域网访问（可选）

如果您想让同一局域网中的其他设备访问当前页面：

1. 查找您 Mac 的局域网 IP，例如 `192.168.x.x`。
2. 在其他设备上访问 `http://<您的-IP>:5173`。
3. 确保后端运行在 `--host 0.0.0.0` 模式下。
4. 如果您使用自定义 API 地址，请在执行 `npm run dev` 前设置 `VITE_API_BASE_URL`，例如 `http://<您的-IP>:8000`。
5. 如果需要自定义 CORS 源，请在后端设置 `AIGC_CORS_ORIGINS` 或 `AIGC_CORS_ORIGIN_REGEX`。

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

---

## 技术架构（供开发者或高级用户参考）

### 项目概述

本项目是一个基于 Web 的平台，旨在实现本地训练 LoRA（低秩自适应）模型、运行 Stable Diffusion 推理，并在 M1 阶段提供简化的“AI 场景生成”体验。主要任务包括：

1. **LoRA 训练**：上传图像、配置训练参数并运行训练流程，无需直接使用命令行。
2. **Stable Diffusion 推理**：提供简洁的界面，通过文本提示词生成图像，并支持加载已训练的 LoRA 模型。
3. **模型管理**：提供专用页面，用于查看、下载和删除已训练的 LoRA 模型。
4. **AI 场景生成（M1/M2）**：参考 LiteDraw 的 Chat→Edit→Preview 流程：先通过聊天与选择题澄清需求，再生成编辑界面并准备素材，最后将背景、主体、装饰、文字与色块图层落到 Web 画布（Canvas）进行预览与导出。

### 系统架构

应用程序采用客户端-服务器架构：

* **前端**：基于 **React（使用 Vite）** 和 **TypeScript** 构建的单页应用。使用 **Material-UI（MUI）** 作为组件库，并使用 **React Router** 处理页面导航。前端现集成了训练、推理、模型管理、画布以及新的“AI 场景生成（M1/M2）”页面，采用统一的侧边栏导航。
* **后端**：基于 **FastAPI** 构建的 Python 服务器，向前端暴露 REST API。
* LoRA 训练：在 `/train` 后台执行，生成元数据与缩略图，若训练失败会自动清理空目录。
* 推理：`/generate` 接口支持基础模型与多 LoRA 组合，按需加载管线。
* 模型管理：通过 `/models` 接口进行列表展示、下载和删除操作。
* 场景生成（M1）：新增 `/generate-text`、`/generate-layout`、`/generate-image`、`/remove-background`、`/analyze-layer-plan` 和 `/compose-materials` 接口，为桌面端工作流提供表单建议、图层规划、透明底生成、抠图、排版预览与落画布数据。

### 核心技术栈

* **前端**：React, Vite, TypeScript, Material-UI, Axios, React Router
* **后端**：Python, FastAPI, PyTorch（利用 MPS 进行 GPU 加速）, `diffusers`, `peft`, `accelerate`, `transformers`

## 开发规范（供开发者参考）

### 后端 API

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

### 前端 UI

* 主要应用组件为 `frontend/src/App.tsx`，现已包含所有页面的路由配置。
* `Layout.tsx` 组件提供了统一的侧边栏和应用栏。
* 训练界面位于 `frontend/src/components/TrainingPage.tsx`。
* 推理界面位于 `frontend/src/components/InferencePage.tsx`，现包含一个用于选择已训练 LoRA 模型的下拉菜单。
* 模型管理界面位于 `frontend/src/components/ModelsPage.tsx`。
* 当前主要的“AI 场景生成（M1/M2）”入口为 `/scene`，核心页面为 `frontend/src/components/scene/SceneFlowPage.tsx`，将聊天、意图板、编辑配置、素材准备与预览整合到单页工作流中。
* 画布页面位于 `frontend/src/components/CanvasPage.tsx`，现支持 `image`、`text`、`shape` 三类图层的导入、拖拽、缩放和导出 PNG。

### 模型与数据存储

* 上传用于训练的图像临时存储在 `temp_training_images` 目录中。
* 训练完成的 LoRA 模型保存至 `lora_models` 目录。每次训练运行都会生成一个带有时间戳的子文件夹。
