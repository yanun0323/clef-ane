# clef-ane

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-Click-yellow" alt="English"></a>
  <a href="README-tw.md"><img src="https://img.shields.io/badge/繁體中文-點擊查看-orange" alt="繁體中文"></a>
  <a href="README-cn.md"><img src="https://img.shields.io/badge/简体中文-点击查看-orange" alt="简体中文"></a>
  <a href="README-ja.md"><img src="https://img.shields.io/badge/日本語-クリック-blue" alt="日本語"></a>
  <a href="README-ko.md"><img src="https://img.shields.io/badge/한국어-클릭-yellow" alt="한국어"></a>
</p>

clef-ane 讓你在 Mac 上執行 Cloudflare 的 [Clef-Flash](https://huggingface.co/Cloudflare/clef-flash)。Clef-Flash 是一個幫你做判斷的模型：你描述一個情況、提出問題，它會告訴你每個答案的可能性有多高。情況可以用文字、圖片或短影片來描述。

它在 Mac 的神經網路引擎（ANE）上執行，這是專門處理 AI 運算的晶片，所以顯示卡（GPU）可以留給其他工作使用。所有運算都在你的 Mac 上完成，不需要連網路。

這是實驗性專案。它用到 Apple 沒有正式開放的 macOS 功能，macOS 更新後可能會無法使用。

## 可以做什麼

- 回答三種問題：是非題、選擇題、等級評分題。
- 同一個請求裡可以放文字、JSON、最多 4 張圖片和一段短影片。
- 完全不使用 GPU。
- 下載一次後就能離線使用。
- 請求格式和 Cloudflare 的 SystemOne API 相同。

## 需要準備

- Apple Silicon 的 Mac。目前只在 M5 Pro、macOS 27 上測試過。
- Python 3.12 以上。
- Apple 的命令列工具：執行 `xcode-select --install`。
- 約 10 GiB 的可用磁碟空間。

## 安裝

在專案資料夾中執行：

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
- 啟動約需 1.5 分鐘。看到 `Listening on http://127.0.0.1:8080` 就可以開始用了。
- 第一次傳圖片或影片時會多等約 10 秒，之後就很快。
- 按 `Ctrl-C` 停止。

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
| `--port` | `8080` | 使用的連接埠 |
| `--host` | `127.0.0.1` | 接收連線的位址。預設只允許你自己的 Mac |
| `--max-tokens` | `16384` | 可接受的最長輸入。數字越小，用的記憶體越少 |
| `--max-body-mb` | `32` | 請求的大小上限（MiB） |
| `--vision-device` | `ane` | 圖片和影片在哪裡處理。`ane` 不占用 GPU；`mps` 用 GPU，稍微快一點；`cpu` 最慢 |
| `--model-dir` | 在 `~/Library/Caches/clef-ane` 裡 | 模型存放位置 |

所有選項請執行 `clef-ane --help`。也可以設定 `CLEF_ANE_HOME` 來指定模型的存放位置。

舊的純文字模型（約 8.90 GiB）仍可使用：

```sh
clef-ane download --revision text-v1 --model-dir ~/clef-text-model
clef-ane serve --model-dir ~/clef-text-model
```

## 目前的限制

- 一次只處理一個請求，另外一個可以排隊，再多的會收到「忙碌中」的錯誤。
- 在我們的 27 題測試中，答案都和原版模型相同。但我們沒有跑 Clef-Flash 的官方評測，整體準確度可能不同。
- 圖片和影片的解析度比原版能用的低，所以細節很多的圖片可能比較不準。
- 它只會替你給的選項打分數，不會寫出文字。

## 待辦事項

原版 Clef-Flash 做得到、但 clef-ane 還做不到的：

- [ ] 同時處理多個請求。
- [ ] 使用更清楚的圖片，以及超過 4 張圖片。
- [ ] 使用更長的影片、更多畫面，以及超過一段影片。
- [ ] 讓你調整圖片和影片的處理方式（`media_kwargs`）。
- [ ] 讓你限制要讀多少情況描述文字（`max_state_tokens`）。
- [ ] 回傳每個答案背後的原始分數（logits）。
- [ ] 檔案太大時可選擇自動縮小或剪短，而不是直接拒絕。
- [ ] 支援較大的 Clef 模型。
- [ ] 跑 Decision Index 官方評測。
- [ ] 提供不需要 Python 的安裝程式。

## 開發者

```sh
make install   # 安裝開發版
make test      # 快速測試，不需要模型
make run       # 啟動服務
make request   # 把 examples/request.json 送到服務
make help      # 列出所有指令
```

和原版模型比對結果（`make test-model`、`make test-images`、`make test-videos`）需要原版模型檔。建置和上傳模型的腳本在 [`scripts/`](scripts/)。

## 授權

Clef-Flash 是 Cloudflare 以 Qwen 的 Qwen3.5-9B 為基礎做成的，兩者都使用 Apache-2.0 授權。下載的模型附有 `LICENSE` 和 `NOTICE`，[`src/clef_ane/_vendor/`](src/clef_ane/_vendor/) 裡的 Cloudflare 程式碼也保留原本的授權。本專案不是 Cloudflare、Qwen 或 Apple 的官方專案。
