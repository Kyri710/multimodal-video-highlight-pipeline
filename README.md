# Multimodal Video Highlight Pipeline

一个结合传统计算机视觉与视觉语言模型的多阶段视频高光画面筛选工具。

项目从视频中抽取候选帧，去除近重复和低质量画面，分别计算 CV 客观质量分数与 Qwen3-VL 主观审美分数，融合后选出具有时间多样性的 Top-K 高光画面。

## 主要功能

- 按秒或帧间隔从视频中抽帧
- 使用 perceptual hash 对连续近似帧分组
- 使用 Laplacian variance 保留组内最清晰画面
- 计算多项传统 CV 质量指标
- 过滤黑屏、白屏、严重曝光异常和严重模糊画面
- 计算 0～10 的 CV 客观质量分数
- 使用 Qwen3-VL 评价构图、人物状态、光影和整体审美
- 融合 CV 与 VLM 两类分数
- 通过最小时间间隔约束选出 Top-K 高光画面
- 将完整评分结果写入 JSON

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
│   └── pipeline.py
│
├── data/
├── runs/
├── playground/
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
| `__main__.py` | 命令行入口 |

## 环境要求

- Python 3.10 或更高版本
- 可被 OpenCV 正常读取的视频文件
- 可访问 Qwen3-VL 接口的 API Key
- 网络连接

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

安装项目依赖：

```powershell
python -m pip install -r requirements.txt

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

## 命令行参数

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

## 输出

每次运行会创建独立目录：

```text
runs/
└── test_YYYYMMDD_HHMMSS_microseconds/
    ├── frame_00000000_000000.000s.jpg
    ├── frame_00000030_000001.000s.jpg
    ├── ...
    └── result.json
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
  "selected_frames": []
}
```

由于时间间隔限制、CV filter 或 VLM 的 `keep` 判断，最终结果可能少于 `top-k`。

## 当前限制

- 当前输出的是高光图片，不是完整视频片段
- 尚未实现片段截取和高光视频合并
- VLM 请求按候选帧顺序执行，尚未并发
- 尚未实现 VLM 响应缓存，重复运行可能重复产生 API 用量
- CV 阈值和评分权重仍需要通过更多类型的视频校准
- 尚未使用音频、字幕、动作识别或物体检测
- 尚未提供 GUI
- 尚未建立完整自动化测试

## Roadmap

计划中的后续功能：

- [ ] 缓存 CV 与 VLM 评分结果
- [ ] 支持中断恢复
- [ ] 根据高光时间点截取视频片段
- [ ] 合并 Top-K 片段为高光视频
- [ ] 增加场景切换和转场检测
- [ ] 增加进度显示与日志
- [ ] 增加单元测试和集成测试
- [ ] 使用配置文件管理各项阈值
- [ ] 增加批量视频处理
- [ ] 完善依赖和打包配置

## 隐私与费用说明

传统 CV 处理在本地完成。

只有通过前置筛选的候选图片会发送到配置的 Qwen3-VL 服务进行评分。使用者应自行确认：

- API 服务的费用
- 图片上传和数据保留政策
- 视频内容是否允许上传到第三方服务
- API Key 的权限和安全性

请勿将 `.env` 或真实 API Key 提交到仓库。

## License

本项目目前尚未添加开源许可证。

在明确添加许可证之前，默认不授予复制、修改或分发代码的权利。