# clef-ane

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-Click-yellow" alt="English"></a>
  <a href="README-tw.md"><img src="https://img.shields.io/badge/繁體中文-點擊查看-orange" alt="繁體中文"></a>
  <a href="README-cn.md"><img src="https://img.shields.io/badge/简体中文-点击查看-orange" alt="简体中文"></a>
  <a href="README-ja.md"><img src="https://img.shields.io/badge/日本語-クリック-blue" alt="日本語"></a>
  <a href="README-ko.md"><img src="https://img.shields.io/badge/한국어-클릭-yellow" alt="한국어"></a>
</p>

clef-ane 让你在 Mac 上运行 Cloudflare 的 [Clef-Flash](https://huggingface.co/Cloudflare/clef-flash)。Clef-Flash 是一个帮你做判断的模型：你描述一个情况、提出问题，它会告诉你每个答案的可能性有多大。情况可以用文字、图片或短视频来描述。

它在 Mac 的神经网络引擎（ANE）上运行，这是专门处理 AI 计算的芯片，所以显卡（GPU）可以留给其他工作。所有计算都在你的 Mac 上完成，不需要联网。

这是一个实验性项目。它用到了 Apple 没有正式开放的 macOS 功能，macOS 更新后可能无法使用。

## 能做什么

- 回答三种问题：是非题、选择题、等级评分题。
- 同一个请求里可以放文字、JSON、最多 4 张图片和一段短视频。
- 完全不使用 GPU。
- 下载一次后就能离线使用。
- 请求格式与 Cloudflare 的 SystemOne API 相同。

## 需要准备

- Apple Silicon（M1 或更新）的 Mac，以及 macOS 14 或更高版本。目前只在 M5 Pro、macOS 27 上测试过。
- 约 11 GiB 的可用磁盘空间：程序约 1 GiB，模型约 10 GiB。

## 安装

在“终端”中运行：

```sh
curl -fsSL https://github.com/yanun0323/clef-ane/releases/latest/download/install.sh | sh
```

它会下载程序（约 260 MB）、确认文件完整，再把 `clef-ane` 命令放到 `~/.local/bin`。不需要另外安装 Python 或 Xcode。如果安装程序提示 `~/.local/bin` 不在 PATH 中，按它显示的那一行操作，再打开一个新的终端窗口。

需要更新时，再运行一次同样的命令。要卸载的话，删除 `~/.local/share/clef-ane` 和 `~/.local/bin/clef-ane`；下载的模型在 `~/Library/Caches/clef-ane`。

如果想改为从源代码安装（需要 Python 3.12 或更高版本，并运行 `xcode-select --install`），在项目文件夹中运行：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

## 开始使用

先下载模型（约 9.74 GiB，只需一次），再启动服务：

```sh
clef-ane download
clef-ane serve
```

- 不需要 Hugging Face 账号。下载中断的话，再运行一次同样的命令就会接着下载。
- 启动约需 1.5 分钟。看到 `Listening on http://127.0.0.1:8080` 就可以开始用了。
- 第一次发送图片或视频时会多等约 10 秒，之后就很快。
- 按 `Ctrl-C` 停止。

> [!WARNING]
> 服务没有密码保护，默认只有你自己的 Mac 能连接。除非是可信的网络，否则不要开放给其他电脑。

## 提出问题

在另一个终端窗口运行：

```sh
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request.json
```

[`examples/request.json`](examples/request.json) 描述了一位被重复扣款的客户，并提出三个问题。你会收到：

```json
{
  "model": "clef",
  "answers": {
    "team": {"type": "choice", "choice": "billing", "confidence": 0.9647,
             "probabilities": {"billing": 0.9647, "technical": 0.0142, "other": 0.0211}},
    "refund": {"type": "noul", "noul": 0.8871},
    "urgency": {"type": "score", "score": 1.6367, "confidence": 0.7115,
                "legend": {"0": "Routine", "1": "Soon", "2": "Urgent"},
                "probabilities": {"0": 0.0748, "1": 0.2137, "2": 0.7115}}
  },
  "usage": {"input_tokens": 343, "output_tokens": 0}
}
```

意思是：这张工单应该交给账务部门（billing，可能性 96%），客户有要求退款（89%），而且最可能是紧急（Urgent）。

### 怎么写请求

请求包含 `state`（情况，可以是文字或 JSON）和最多 64 个 `questions`。每个问题有：

| 字段 | 含义 |
|---|---|
| `type` | `noul` 是非题、`choice` 选择题、`score` 等级评分题 |
| `instructions` | 问题本身。可以省略 |
| `criteria` | 选项。`choice` 填选项名称和说明；`score` 填一个从低到高的列表 |

`noul` 的答案是“是”的概率；`choice` 和 `score` 会给出每个选项的概率。

## 图片和视频

示例程序会读取你 Mac 上的文件并发送：

```sh
python examples/image_request.py receipt.png --question "Is the receipt marked as paid?"
python examples/video_request.py clip.mp4 --question "Does the person enter the room?"
```

如果要自己发送文件，请先用 base64 把文件转成文字，再放进请求：

```json
{
  "state": "Decide from the attached image.",
  "images": ["data:image/png;base64,iVBORw0..."],
  "questions": {"paid": {"type": "noul", "instructions": "Is the receipt marked as paid?"}}
}
```

视频使用 `"videos": ["data:video/mp4;base64,..."]`。MOV 用 `video/quicktime`，WebM 用 `video/webm`。

| | 图片 | 视频 |
|---|---|---|
| 文件类型 | PNG、JPEG、WebP | MP4、MOV、WebM |
| 数量 | 最多 4 张 | 1 段 |
| 文件上限 | 每张 4 MiB，合计 8 MiB | 16 MiB |
| 尺寸或时长上限 | 每张 1600 万像素 | 30 秒 |
| 模型实际看到的 | 每张缩小到约 512×512 | 每秒约 2 帧画面，每帧缩小到约 256×256 |

**文件太大会直接报错。** 在上限内的图片和视频会自动缩小（不会裁剪）；超过上限就会被拒绝，系统不会帮你剪短或压缩。请先把长视频剪短、大文件压缩后再发送。

另外需要知道：

- 视频的声音会被忽略。
- 图片会被缩小、视频只取部分画面，所以小字和很快的动作可能会被漏看。
- 不能发送网址或文件路径，要把文件内容直接放进请求。

## 选项

| 选项 | 默认值 | 作用 |
|---|---|---|
| `--port` | `8080` | 使用的端口 |
| `--host` | `127.0.0.1` | 接收连接的地址。默认只允许你自己的 Mac |
| `--max-tokens` | `16384` | 可接受的最长输入。数字越小，占用内存越少 |
| `--max-body-mb` | `32` | 请求大小上限（MiB） |
| `--vision-device` | `ane` | 图片和视频在哪里处理。`ane` 不占用 GPU；`mps` 使用 GPU，稍微快一点；`cpu` 最慢 |
| `--model-dir` | 在 `~/Library/Caches/clef-ane` 中 | 模型存放位置 |

全部选项请运行 `clef-ane --help`。也可以设置 `CLEF_ANE_HOME` 来指定模型的存放位置。

旧的纯文字模型（约 8.90 GiB）仍可使用：

```sh
clef-ane download --revision text-v1 --model-dir ~/clef-text-model
clef-ane serve --model-dir ~/clef-text-model
```

## 当前限制

- 一次只处理一个请求，另外一个可以排队，更多的会收到“忙碌中”的错误。
- 在我们的 27 道测试题中，答案都与原版模型相同。但我们没有运行 Clef-Flash 的官方评测，整体准确率可能不同。
- 图片和视频的分辨率比原版能使用的低，所以细节丰富的图片可能不够准确。
- 它只会为你给出的选项打分，不会生成文字。

## 待办事项

原版 Clef-Flash 能做到、但 clef-ane 还做不到的：

- [ ] 同时处理多个请求。
- [ ] 使用更清晰的图片，以及超过 4 张图片。
- [ ] 使用更长的视频、更多画面，以及超过一段视频。
- [ ] 让你调整图片和视频的处理方式（`media_kwargs`）。
- [ ] 让你限制读取多少情况描述文字（`max_state_tokens`）。
- [ ] 返回每个答案背后的原始分数（logits）。
- [ ] 文件太大时可选择自动缩小或剪短，而不是直接拒绝。
- [ ] 支持更大的 Clef 模型。
- [ ] 运行 Decision Index 官方评测。

## 开发者

```sh
make install   # 安装开发版
make test      # 快速测试，不需要模型
make run       # 启动服务
make request   # 把 examples/request.json 发送到服务
make package   # 把安装文件打包到 dist/
make help      # 列出所有命令
```

与原版模型比对结果（`make test-model`、`make test-images`、`make test-videos`）需要原版模型文件。构建和上传模型的脚本在 [`scripts/`](scripts/) 中。

## 许可

clef-ane 使用 [Apache-2.0 许可](LICENSE)。安装包中还附带其他软件，各自有自己的许可，列在安装文件夹的 `THIRD_PARTY_NOTICES.txt` 中；其中包括 GPL 许可的 FFmpeg 编码库（x264 和 x265）。

Clef-Flash 是 Cloudflare 基于 Qwen 的 Qwen3.5-9B 制作的，两者都使用 Apache-2.0 许可。下载的模型附带 `LICENSE` 和 `NOTICE`，[`src/clef_ane/_vendor/`](src/clef_ane/_vendor/) 中的 Cloudflare 代码也保留原有许可。本项目不是 Cloudflare、Qwen 或 Apple 的官方项目。
