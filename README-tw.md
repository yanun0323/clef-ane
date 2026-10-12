# clef-ane

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-Click-yellow" alt="English"></a>
  <a href="README-tw.md"><img src="https://img.shields.io/badge/繁體中文-點擊查看-orange" alt="繁體中文"></a>
  <a href="README-cn.md"><img src="https://img.shields.io/badge/简体中文-点击查看-orange" alt="简体中文"></a>
  <a href="README-ja.md"><img src="https://img.shields.io/badge/日本語-クリック-blue" alt="日本語"></a>
  <a href="README-ko.md"><img src="https://img.shields.io/badge/한국어-클릭-yellow" alt="한국어"></a>
</p>

clef-ane 讓你在 Mac 上執行 Cloudflare 的 [Clef-Flash](https://huggingface.co/Cloudflare/clef-flash)，以及較大的 [Clef](https://huggingface.co/Cloudflare/clef)。它們是幫你做判斷的模型：你描述一個情況、提出問題，模型會告訴你每個答案的可能性有多高。情況可以用文字、圖片或短影片來描述。

它在 Mac 的神經網路引擎（ANE）上執行，這是專門處理 AI 運算的晶片，所以顯示卡（GPU）可以留給其他工作使用。所有運算都在你的 Mac 上完成，不需要連網路。

這是實驗性專案。它用到 Apple 沒有正式開放的 macOS 功能，macOS 更新後可能會無法使用。

## 可以做什麼

- 回答三種問題：是非題、選擇題、等級評分題。
- 同一個請求裡可以放文字、JSON、最多 4 張圖片和一段短影片。
- 完全不使用 GPU。
- 下載一次後就能離線使用。
- 請求格式和 Cloudflare 的 SystemOne API 相同。

## 需要準備

- Apple Silicon（M1 以上）的 Mac，以及 macOS 14 以上。目前只在 M5 Pro、macOS 27 上測試過。
- 約 11 GiB 的可用磁碟空間：程式約 1 GiB，模型約 10 GiB。
- 如果要用較大的 Clef 模型：約 24 GiB 的可用磁碟空間，以及至少 48 GB 記憶體的 Mac。目前只在 64 GB 上測試過。

## 安裝

在「終端機」執行：

```sh
curl -fsSL https://github.com/yanun0323/clef-ane/releases/latest/download/install.sh | sh
```

它會下載程式（約 260 MB）、確認檔案完整，再把 `clef-ane` 指令放到 `~/.local/bin`。不需要另外安裝 Python 或 Xcode。如果安裝程式提示 `~/.local/bin` 不在 PATH 裡，照著它顯示的那一行做，再開一個新的終端機視窗。

要更新時，再執行一次同樣的指令。要移除的話，刪掉 `~/.local/share/clef-ane` 和 `~/.local/bin/clef-ane`；下載的模型在 `~/Library/Caches/clef-ane`。

如果想改從原始碼安裝（需要 Python 3.12 以上，並執行 `xcode-select --install`），在專案資料夾中執行：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

## 開始使用

先下載模型（約 9.74 GiB，只需一次），再啟動服務：

```sh
clef-ane download
clef-ane serve
```

- 不需要 Hugging Face 帳號。下載中斷的話，再執行一次同樣的指令就會接著下載。
- 啟動約需 2 分鐘。看到 `Listening on http://127.0.0.1:8080` 就可以開始用了。
- 第一次傳圖片或影片時會多等約 10 秒，之後就很快。
- 按 `Ctrl-C` 停止。

### 較大的 Clef 模型

預設使用 Clef-Flash。想改用 Clef，在兩個指令都加上 `--model clef`：

```sh
clef-ane download --model clef
clef-ane serve --model clef
```

| | Clef-Flash | Clef |
|---|---|---|
| 原版模型大小 | 9B | 27B |
| 下載大小 | 9.74 GiB | 22.94 GiB |
| 啟動時間 | 約 2 分鐘 | 約 7 分鐘 |
| 執行時使用的記憶體 | 約 14 GiB | 約 27 GiB |
| 下面的範例問題 | 約 0.5 秒 | 約 1.6 秒 |

以上是在 64 GB 記憶體的 M5 Pro 上量到的。兩個模型可以都下載，但一次只能執行一個：神經網路引擎放不下兩個。根據 Cloudflare 自己的評測，有些任務 Clef 較好，有些是 Clef-Flash 較好，詳見 [Clef 的模型說明](https://huggingface.co/Cloudflare/clef)。

> [!WARNING]
> 服務沒有密碼保護，預設只有你自己的 Mac 能連線。除非是信任的網路，否則不要開放給其他電腦。

## 提出問題

在另一個終端機視窗執行：

```sh
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request.json
```

[`examples/request.json`](examples/request.json) 描述一位被重複扣款的客戶，並提出三個問題。你會收到：

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

意思是：這張客服單應該交給帳務部門（billing，可能性 96%），客戶有要求退款（89%），而且最可能是緊急（Urgent）。

### 怎麼寫請求

請求包含 `state`（情況，可以是文字或 JSON）和最多 64 個 `questions`。每個問題有：

| 欄位 | 意思 |
|---|---|
| `type` | `noul` 是非題、`choice` 選擇題、`score` 等級評分題 |
| `instructions` | 問題本身。可以省略 |
| `criteria` | 選項。`choice` 填選項名稱和說明；`score` 填一個從低到高的清單 |

`noul` 的答案是「是」的機率；`choice` 和 `score` 會給你每個選項的機率。

## 圖片和影片

範例程式會讀取你 Mac 上的檔案並送出：

```sh
python examples/image_request.py receipt.png --question "Is the receipt marked as paid?"
python examples/video_request.py clip.mp4 --question "Does the person enter the room?"
```

如果要自己送檔案，請先用 base64 把檔案轉成文字，再放進請求：

```json
{
  "state": "Decide from the attached image.",
  "images": ["data:image/png;base64,iVBORw0..."],
  "questions": {"paid": {"type": "noul", "instructions": "Is the receipt marked as paid?"}}
}
```

影片使用 `"videos": ["data:video/mp4;base64,..."]`。MOV 用 `video/quicktime`，WebM 用 `video/webm`。

| | 圖片 | 影片 |
|---|---|---|
| 檔案類型 | PNG、JPEG、WebP | MP4、MOV、WebM |
| 數量 | 最多 4 張 | 1 段 |
| 檔案上限 | 每張 4 MiB，合計 8 MiB | 16 MiB |
| 尺寸或長度上限 | 每張 1,600 萬像素 | 30 秒 |
| 模型實際看到的 | 每張縮小到約 512×512 | 每秒約 2 張畫面，每張縮小到約 256×256 |

**檔案太大會直接報錯。** 在上限內的圖片和影片會自動縮小（不會裁切）；超過上限就會被拒絕，系統不會幫你剪短或壓縮。請先把長影片剪短、大檔案壓縮後再傳。

另外要知道：

- 影片的聲音會被忽略。
- 圖片會縮小、影片只取部分畫面，所以小字和很快的動作可能會漏看。
- 不能傳網址或檔案路徑，要把檔案內容直接放進請求。

## 選項

| 選項 | 預設值 | 用途 |
|---|---|---|
| `--model` | `clef-flash` | 要用哪個模型：`clef-flash` 或 `clef`。`download` 和 `serve` 要用同一個 |
| `--port` | `8080` | 使用的連接埠 |
| `--host` | `127.0.0.1` | 接收連線的位址。預設只允許你自己的 Mac |
| `--max-tokens` | `16384` | 可接受的最長輸入。數字越小，用的記憶體越少 |
| `--max-body-mb` | `32` | 請求的大小上限（MiB） |
| `--vision-device` | `ane` | 圖片和影片在哪裡處理。`ane` 不占用 GPU；`mps` 用 GPU，稍微快一點；`cpu` 最慢 |
| `--model-dir` | 在 `~/Library/Caches/clef-ane` 裡 | 模型存放位置。每個模型有自己的資料夾 |

所有選項請執行 `clef-ane --help`。也可以設定 `CLEF_ANE_HOME` 來指定模型的存放位置。

舊的純文字模型（約 8.90 GiB）仍可使用：

```sh
clef-ane download --revision text-v1 --model-dir ~/clef-text-model
clef-ane serve --model-dir ~/clef-text-model
```

## 目前的限制

- 一次只處理一個請求，另外一個可以排隊，再多的會收到「忙碌中」的錯誤。
- 在我們的 27 題測試中，兩個模型的答案都和各自的原版相同。但我們沒有跑官方評測，整體準確度可能不同。
- macOS 讓所有 App 加起來最多在神經網路引擎上放 127 個程式。含圖片功能時，Clef-Flash 會用掉 123 個，Clef 會用掉 124 個。如果有其他 App 正在使用神經網路引擎，啟動可能會失敗；把 `--max-tokens` 設小一點可以少用一些。
- 圖片和影片的解析度比原版能用的低，所以細節很多的圖片可能比較不準。
- 它只會替你給的選項打分數，不會寫出文字。

## 待辦事項

原版模型做得到、但 clef-ane 還做不到的：

- [ ] 同時處理多個請求。
- [ ] 使用更清楚的圖片，以及超過 4 張圖片。
- [ ] 使用更長的影片、更多畫面，以及超過一段影片。
- [ ] 讓你調整圖片和影片的處理方式（`media_kwargs`）。
- [ ] 讓你限制要讀多少情況描述文字（`max_state_tokens`）。
- [ ] 回傳每個答案背後的原始分數（logits）。
- [ ] 檔案太大時可選擇自動縮小或剪短，而不是直接拒絕。
- [ ] 跑 Decision Index 官方評測。

## 開發者

```sh
make install   # 安裝開發版
make test      # 快速測試，不需要模型
make run       # 啟動服務
make request   # 把 examples/request.json 送到服務
make package   # 把安裝檔打包到 dist/
make help      # 列出所有指令
```

和原版模型比對結果（`make test-model`、`make test-images`、`make test-videos`）需要原版模型檔。建置和上傳模型的腳本在 [`scripts/`](scripts/)。這些指令預設處理 Clef-Flash，加上 `MODEL=clef` 才會處理 Clef，例如 `make fetch MODEL=clef`。

## 授權

clef-ane 使用 [Apache-2.0 授權](LICENSE)。安裝檔裡也附帶其他軟體，各自有自己的授權，列在安裝資料夾的 `THIRD_PARTY_NOTICES.txt`；其中包含 GPL 授權的 FFmpeg 編碼函式庫（x264 和 x265）。

Cloudflare 以 Qwen 的 Qwen3.5-9B 為基礎做出 Clef-Flash，以 Qwen3.8-27B 為基礎做出 Clef，它們都使用 Apache-2.0 授權。下載的模型附有 `LICENSE` 和 `NOTICE`，[`src/clef_ane/_vendor/`](src/clef_ane/_vendor/) 裡的 Cloudflare 程式碼也保留原本的授權。本專案不是 Cloudflare、Qwen 或 Apple 的官方專案。
