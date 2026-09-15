# Multimodal Video Highlight Pipeline

一个结合传统计算机视觉与视觉语言模型的多阶段视频高光画面筛选工具。

项目从视频中抽取候选帧，去除近重复和低质量画面，分别计算 CV 客观质量分数与 Qwen3-VL 主观审美分数，融合后选出具有时间多样性的 Top-K 高光画面。

项目同时提供 Gradio 交互界面。用户可以从高光结果中选择图片、绘制局部蒙版，并使用 Stable Diffusion Inpainting 在本地 NVIDIA GPU 上进一步完善画面。

## 主要功能
### 高光筛选

- 视频抽帧与近似帧分组
- 组内清晰帧选择
- CV 质量过滤和客观评分
- Qwen3-VL 主观评分
- 多源分数融合与 Top-K 排序
- JSON 结果输出

### 局部重绘

- Gradio 视频上传和 Gallery 展示
- 高光图片选择与 ImageEditor 蒙版绘制
- Stable Diffusion GPU 局部重绘
- 保留蒙版外原始图像
- 保存结果、蒙版和参数 JSON

## 快速开始
| 文件 | 用途 |
|---|---|
| `requirements.txt` | 高光筛选核心依赖 |
| `requirements-inpaint.txt` | Stable Diffusion 局部重绘依赖 |
| `requirements-app.txt` | 完整 Gradio 应用依赖 |
安装完整依赖并配置 `DASHSCOPE_API_KEY` 后运行：

```powershell
python app.py
```

然后打开：

```text
http://127.0.0.1:7860
```

仅运行高光筛选 CLI：

```powershell
python -m src data/test.mp4 --output runs
```

## Pipeline

```text
输入视频
    ↓
按时间或帧间隔抽帧
    ↓
pHash 连续近似帧分组
    ↓
每组保留最清晰帧
    ↓
计算完整 CV 指标
    ↓
CV 过滤纯黑纯白等极端画面
    ↓
CV 客观评分
    ↓
Qwen3-VL 主观评分
    ↓
CV / VLM 分数融合
    ↓
按最终分数排序
    ↓
时间去重
    ↓
Top-K 高光画面
    ↓
result.json
    ↓
Gradio Gallery 展示
    ↓
用户选择高光图片
    ↓
ImageEditor 绘制蒙版
    ↓
输入 prompt 与生成参数
    ↓
Stable Diffusion Inpainting
    ↓
蒙版外与原图重新合成
    ↓
重绘图片 + 蒙版 + 参数 JSON
```

## CV 指标

当前计算的客观指标包括：

| 指标 | 含义 |
|---|---|
| `sharpness` | Laplacian 方差，衡量画面清晰度 |
| `brightness` | 灰度平均值 |
| `contrast` | 灰度标准差 |
| `saturation` | HSV 饱和度平均值 |
| `overexposure_ratio` | 过曝像素比例 |
| `underexposure_ratio` | 欠曝像素比例 |
| `black_clip_ratio` | 接近纯黑的像素比例 |
| `white_clip_ratio` | 接近纯白的像素比例 |

CV filter 只负责移除明显不可用的画面。未达到硬过滤条件的质量问题会在后续 `cv_score` 中以软惩罚形式体现。

## VLM 评分维度

Qwen3-VL 当前评价以下维度，每项范围为 0～10：

- `composition`：构图
- `subject_prominence`：主体突出程度
- `pose_expression`：人物姿态与表情
- `background_cleanliness`：背景干扰程度
- `lighting_aesthetic`：光影观感
- `moment_quality`：瞬间质量
- `aesthetic`：整体审美

模型还会返回：

- `keep`：是否建议保留
- `reason`：评分理由

`vl_score` 当前为上述七项分数的平均值。

## 最终评分

默认融合公式：

```text
final_score = cv_score × 0.4 + vl_score × 0.6
```

两项输入分数和最终分数范围均为 0～10。

权重可以通过命令行参数修改。

## 项目结构

```text
multimodal-video-highlight-pipeline/
├── app.py
├── requirements.txt
├── requirements-inpaint.txt
├── requirements-app.txt
│
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── models.py
│   ├── extraction.py
│   ├── frame_selector.py
│   ├── image_quality.py
│   ├── attach.py
│   ├── frame_filter.py
│   ├── objective_scorer.py
│   ├── vl_scorer.py
│   ├── fusion.py
│   ├── ranking.py
│   ├── pipeline.py
│   └── inpaint.py
│
├── data/          # 本地输入视频，不提交
├── runs/          # 运行输出，不提交
├── playground/    # 本地实验内容，不提交
├── .env.example
├── .gitignore
└── README.md
```

各模块职责：

| 模块 | 职责 |
|---|---|
| `models.py` | 定义视频、帧、CV 和 VLM 结果的数据模型 |
| `extraction.py` | 视频抽帧与视频元数据读取 |
| `frame_selector.py` | pHash 分组及组内最清晰帧选择 |
| `image_quality.py` | 单张图片的 CV 指标计算 |
| `attach.py` | 将 CV 指标写入帧结构体 |
| `frame_filter.py` | 黑白屏、曝光异常和严重模糊过滤 |
| `objective_scorer.py` | 将 CV 指标转换为 0～10 客观分数 |
| `vl_scorer.py` | 调用 Qwen3-VL 并解析结构化评分 |
| `fusion.py` | 融合 CV 和 VLM 分数 |
| `ranking.py` | 排序、时间去重和 Top-K 选择 |
| `pipeline.py` | 编排完整处理流程 |
| `inpaint.py` | 加载本地重绘模型，执行局部重绘并保存结果 |
| `app.py` | Gradio 交互界面，连接高光筛选与局部重绘 |
| `__main__.py` | 命令行入口 |

## 环境要求

### 高光筛选 CLI

- Python 3.10 或更高版本
- 可被 OpenCV 正常读取的视频文件
- 可访问 Qwen3-VL 接口的 API Key
- 网络连接

### Gradio 与局部重绘

- Python 3.10 或更高版本
- NVIDIA GPU
- 推荐至少 8 GB 显存
- 支持 CUDA 的 PyTorch
- Diffusers、Transformers、Accelerate 和 Gradio
- 已存在于本机 Hugging Face 缓存中的 Stable Diffusion Inpainting 模型

## 安装

克隆仓库：

```powershell
git clone https://github.com/Kyri710/multimodal-video-highlight-pipeline.git
cd multimodal-video-highlight-pipeline
```

创建虚拟环境：

```powershell
python -m venv .venv
```

在 Windows PowerShell 中激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

### 仅安装高光筛选功能

```powershell
python -m pip install -r requirements.txt
```

### 安装完整 Gradio 与 GPU 重绘功能

首先根据显卡和驱动，从 PyTorch 官方渠道安装支持 CUDA 的 PyTorch。当前开发环境使用 CUDA 12.6：

```powershell
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
```

然后安装应用依赖：

```powershell
python -m pip install -r requirements-app.txt
```

验证 CUDA：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

`torch.cuda.is_available()` 应输出 `True`。

## API Key 配置

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

编辑本地 `.env`：

```dotenv
DASHSCOPE_API_KEY=你的APIKey
```

`.env` 已被 `.gitignore` 忽略，不应提交到 Git。

可以安全验证 Key 是否已被加载：

```powershell
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(bool(os.getenv('DASHSCOPE_API_KEY')))"
```

正确结果应为：

```text
True
```

该命令不会输出 Key 内容。

## 使用方法

### 使用命令行

查看所有命令行参数：

```powershell
python -m src --help
```

基本运行：

```powershell
python -m src data/test.mp4 --output runs
```

第一次测试建议增大抽帧间隔，减少 API 请求数量：

```powershell
python -m src data/test.mp4 `
  --output runs `
  --interval 30 `
  --top-k 1
```

完整示例：

```powershell
python -m src data/test.mp4 `
  --output runs `
  --mode seconds `
  --interval 1 `
  --jpg-quality 95 `
  --hash-threshold 10 `
  --top-k 5 `
  --min-gap 3 `
  --cv-weight 0.4 `
  --vl-weight 0.6
```

#### 命令行参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `video` | 必填 | 输入视频路径 |
| `--output` | `runs` | 输出根目录 |
| `--mode` | `seconds` | 抽帧间隔单位，可选 `seconds` 或 `frames` |
| `--interval` | `1.0` | 抽帧间隔 |
| `--jpg-quality` | `95` | JPG 保存质量，范围 0～100 |
| `--hash-threshold` | `10` | pHash 相似帧分组阈值 |
| `--top-k` | `5` | 最终选择数量 |
| `--min-gap` | `3.0` | 高光画面之间的最小时间间隔，单位为秒 |
| `--cv-weight` | `0.4` | CV 分数融合权重 |
| `--vl-weight` | `0.6` | VLM 分数融合权重 |

当 `--mode frames` 时，`--interval` 必须是整数值：

```powershell
python -m src data/test.mp4 `
  --mode frames `
  --interval 30
```

### 启动 Gradio 应用

```powershell
python app.py
```

应用默认只监听：

```text
http://127.0.0.1:7860
```

启动后：

1. 上传视频。
2. 设置抽帧间隔、Top-K 和最小时间间隔。
3. 点击“开始高光筛选”。
4. 从 Gallery 中选择一张高光图片。
5. 使用白色画笔涂出需要重绘的区域。
6. 输入重绘提示词和生成参数。
7. 点击“开始局部重绘”。
8. 查看结果并下载参数 JSON。

视频分析会调用 Qwen3-VL，可能产生 API 用量。局部重绘在本地 NVIDIA GPU 上运行。

## 输出

每次运行会创建独立目录：

```text
runs/
└── test_YYYYMMDD_HHMMSS_microseconds/
    ├── frame_00000000_000000.000s.jpg
    ├── frame_00000030_000001.000s.jpg
    ├── result.json
    └── edits/
        ├── frame_xxxxxxxx.png
        ├── frame_xxxxxxxx_mask.png
        └── frame_xxxxxxxx.json
```

`result.json` 包含：

- 视频元数据
- 所有候选帧
- pHash 分组编号
- CV 原始指标
- CV filter 结果及拒绝原因
- CV 各评分分量
- `cv_score`
- VLM 各维度评分及理由
- `vl_score`
- `final_score`
- 最终选中的高光画面
每次重绘都会生成唯一文件名，不会覆盖原始高光图片。重绘参数 JSON 包含源图片、模型名称、prompt、negative prompt、seed、strength、推理步数和 guidance scale。

简化示例：

```json
{
  "video_path": "data/test.mp4",
  "metadata": {
    "fps": 30.0,
    "total_frames": 719,
    "duration_sec": 23.97,
    "width": 1080,
    "height": 1920
  },
  "frames": [
    {
      "frame_id": "frame_00000060_000002.000s",
      "image_path": "runs/.../frame_00000060_000002.000s.jpg",
      "frame_index": 60,
      "timestamp_sec": 2.0,
      "group_id": 1,
      "passed_filter": true,
      "reject_reasons": [],
      "cv_score": 8.7,
      "vl_score": 8.4,
      "final_score": 8.52
    }
  ],
  "selected_frames": [
  {
    "frame_id": "frame_00000060_000002.000s",
    "timestamp_sec": 2.0,
    "cv_score": 8.7,
    "vl_score": 8.4,
    "final_score": 8.52
  }
]
}
```

由于时间间隔限制、CV filter 或 VLM 的 `keep` 判断，最终结果可能少于 `top-k`。

## 当前限制

- 当前输出的是高光图片，不是完整视频片段
- 尚未实现片段截取和高光视频合并
- VLM 请求按候选帧顺序执行，尚未并发
- 尚未实现 VLM 响应缓存，重复分析可能产生额外 API 用量
- CV 阈值和评分权重仍需通过更多类型的视频校准
- 尚未使用音频、字幕、动作识别或物体检测
- 当前提供的是本地 Gradio 界面，尚未按生产环境部署
- 重绘模型需要 NVIDIA GPU，目前没有 CPU 降级方案
- 当前重绘模型以最长边 512 像素进行推理，大图重绘区域可能略软
- 当前没有用户认证，不应直接暴露到公网
- 尚未建立完整自动化测试


## Roadmap

- [x] 多阶段视频高光画面筛选
- [x] CV 与 VLM 多源评分融合
- [x] Gradio 本地交互界面
- [x] 高光图片选择与蒙版绘制
- [x] NVIDIA GPU 局部重绘
- [x] 保存重绘结果、蒙版和参数
- [ ] 缓存 CV 与 VLM 评分结果
- [ ] 支持中断恢复
- [ ] 根据高光时间点截取视频片段
- [ ] 合并 Top-K 片段为高光视频
- [ ] 增加场景切换和转场检测
- [ ] 增加自动化测试
- [ ] 使用配置文件统一管理阈值
- [ ] 支持批量视频处理
- [ ] 完善模型下载与离线缓存管理

## 隐私与费用说明

传统 CV 处理和 Stable Diffusion Inpainting 在本地完成。

只有通过前置筛选的候选图片会发送到配置的 Qwen3-VL 服务进行主观评分。重绘图片和蒙版不会由本项目主动上传到第三方服务。

请勿将以下内容提交到 Git：

- `.env`
- API Key 或 Hugging Face Token
- 用户上传的视频和图片
- 重绘蒙版
- `runs/` 中的生成结果
- 本地模型权重

> 当前开发版本默认从 Hugging Face 本地缓存加载模型，不会自动下载。运行重绘功能前，需要确保 `stable-diffusion-v1-5/stable-diffusion-inpainting` 已存在于本机 Hugging Face 缓存中。

## License

本项目目前尚未添加开源许可证。

在明确添加许可证之前，默认不授予复制、修改或分发代码的权利。