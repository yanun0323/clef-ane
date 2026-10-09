# clef-ane

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-Click-yellow" alt="English"></a>
  <a href="README-tw.md"><img src="https://img.shields.io/badge/繁體中文-點擊查看-orange" alt="繁體中文"></a>
  <a href="README-cn.md"><img src="https://img.shields.io/badge/简体中文-点击查看-orange" alt="简体中文"></a>
  <a href="README-ja.md"><img src="https://img.shields.io/badge/日本語-クリック-blue" alt="日本語"></a>
  <a href="README-ko.md"><img src="https://img.shields.io/badge/한국어-클릭-yellow" alt="한국어"></a>
</p>

clef-ane は、Cloudflare の [Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) を Mac で動かすためのツールです。Clef-Flash は判断をするモデルです。状況を説明して質問すると、それぞれの答えがどのくらいありそうかを教えてくれます。状況はテキスト、画像、短い動画で伝えられます。

Mac の Neural Engine（ANE）という AI 処理用のチップで動くので、グラフィックス（GPU）はほかの作業のために空いたままです。処理はすべて Mac の中で行われ、インターネット接続は不要です。

これは実験的なプロジェクトです。Apple が正式にサポートしていない macOS の機能を使っているため、macOS のアップデートで動かなくなる可能性があります。

## できること

- 3 種類の質問に答えます：はい/いいえ、選択肢から 1 つ選ぶ、段階で評価する。
- 1 つのリクエストに、テキスト、JSON、最大 4 枚の画像、短い動画 1 本を入れられます。
- GPU はまったく使いません。
- 一度ダウンロードすれば、オフラインで使えます。
- リクエストの形式は Cloudflare の SystemOne API と同じです。

## 必要なもの

- Apple Silicon の Mac。今のところ M5 Pro と macOS 27 でしか試していません。
- Python 3.12 以降。
- Apple のコマンドラインツール：`xcode-select --install` を実行します。
- 約 10 GiB の空きディスク容量。

## インストール

プロジェクトのフォルダで次を実行します。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

## 使い始める

最初に一度だけモデル（約 9.74 GiB）をダウンロードし、サーバーを起動します。

```sh
clef-ane download
clef-ane serve
```

- Hugging Face のアカウントは不要です。ダウンロードが止まったら、同じコマンドをもう一度実行すると続きから再開します。
- 起動には約 1.5 分かかります。`Listening on http://127.0.0.1:8080` と表示されたら使えます。
- 画像や動画を含む最初のリクエストは、約 10 秒長くかかります。その後は速くなります。
- 止めるときは `Ctrl-C` を押します。

> [!WARNING]
> サーバーにはパスワードがありません。初期設定では、あなたの Mac からしか接続できません。信頼できるネットワーク以外では、ほかのコンピューターに公開しないでください。

## 質問する

別のターミナルウィンドウで次を実行します。

```sh
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request.json
```

[`examples/request.json`](examples/request.json) は、二重に請求されたお客様について 3 つの質問をします。返ってくる結果は次のとおりです。

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

つまり、このチケットは請求担当（billing）に回すべきで（可能性 96%）、お客様は返金を求めていて（89%）、緊急度は「Urgent」の可能性が最も高い、という結果です。

### リクエストの書き方

リクエストには `state`（状況。テキストまたは JSON）と、最大 64 個の `questions` を入れます。各質問の項目は次のとおりです。

| 項目 | 意味 |
|---|---|
| `type` | `noul` ははい/いいえ、`choice` は選択肢から 1 つ、`score` は段階評価 |
| `instructions` | 質問の文章。省略できます |
| `criteria` | 選択肢。`choice` では名前と説明、`score` では低い順から高い順のリスト |

`noul` の答えは「はい」である確率です。`choice` と `score` では、すべての選択肢の確率が返ります。

## 画像と動画

サンプルスクリプトは、Mac 上のファイルを読み込んで送信します。

```sh
python examples/image_request.py receipt.png --question "Is the receipt marked as paid?"
python examples/video_request.py clip.mp4 --question "Does the person enter the room?"
```

自分でファイルを送る場合は、base64 でファイルを文字に変換してリクエストに入れます。

```json
{
  "state": "Decide from the attached image.",
  "images": ["data:image/png;base64,iVBORw0..."],
  "questions": {"paid": {"type": "noul", "instructions": "Is the receipt marked as paid?"}}
}
```

動画は `"videos": ["data:video/mp4;base64,..."]` を使います。MOV は `video/quicktime`、WebM は `video/webm` です。

| | 画像 | 動画 |
|---|---|---|
| ファイルの種類 | PNG、JPEG、WebP | MP4、MOV、WebM |
| 数 | 最大 4 枚 | 1 本 |
| ファイルサイズの上限 | 1 枚 4 MiB、合計 8 MiB | 16 MiB |
| 大きさ・長さの上限 | 1 枚 1600 万画素 | 30 秒 |
| モデルが実際に見るもの | 各画像を約 512×512 に縮小 | 1 秒あたり約 2 コマ、各コマを約 256×256 に縮小 |

**ファイルが大きすぎるとエラーになります。** 上限内の画像や動画は自動で縮小されます（切り抜きはしません）。上限を超えると受け付けられず、自動で短くしたり圧縮したりはしません。長い動画は短くし、大きなファイルは圧縮してから送ってください。

ほかにも知っておくとよいこと：

- 動画の音声は使いません。
- 画像は縮小され、動画は一部のコマだけを使うため、小さな文字や速い動きは見落とすことがあります。
- URL やファイルのパスは使えません。ファイルの中身をリクエストに入れて送ってください。

## オプション

| オプション | 初期値 | 内容 |
|---|---|---|
| `--port` | `8080` | 使うポート番号 |
| `--host` | `127.0.0.1` | 接続を受け付けるアドレス。初期値ではあなたの Mac からのみ接続できます |
| `--max-tokens` | `16384` | 受け付ける入力の最大の長さ。小さくするとメモリ使用量が減ります |
| `--max-body-mb` | `32` | リクエストサイズの上限（MiB） |
| `--vision-device` | `ane` | 画像と動画をどこで処理するか。`ane` は GPU を使いません。`mps` は GPU を使い、少し速くなります。`cpu` は最も遅くなります |
| `--model-dir` | `~/Library/Caches/clef-ane` 内 | モデルの保存場所 |

すべてのオプションは `clef-ane --help` で確認できます。`CLEF_ANE_HOME` を設定して、モデルの保存場所を変えることもできます。

以前のテキスト専用モデル（約 8.90 GiB）も引き続き使えます。

```sh
clef-ane download --revision text-v1 --model-dir ~/clef-text-model
clef-ane serve --model-dir ~/clef-text-model
```

## 現在の制限

- リクエストは 1 つずつ処理します。もう 1 つまで待機でき、それ以上は「混雑中」のエラーになります。
- 27 問のテストでは、元のモデルと同じ答えになりました。ただし Clef-Flash 公式のベンチマークは実行していないため、全体的な精度は異なる可能性があります。
- 画像と動画は元のモデルが扱える解像度より低くして使うため、細かい部分の多い画像では精度が下がることがあります。
- 指定した選択肢に点数をつけるだけで、文章は書きません。

## やることリスト

元の Clef-Flash にはできて、clef-ane にはまだできないこと：

- [ ] 複数のリクエストを同時に処理する。
- [ ] より鮮明な画像と、4 枚を超える画像を使う。
- [ ] より長い動画、より多くのコマ、複数の動画を使う。
- [ ] 画像と動画の処理方法を調整できるようにする（`media_kwargs`）。
- [ ] 状況説明のテキストをどこまで読むか制限できるようにする（`max_state_tokens`）。
- [ ] 各答えのもとになる生のスコア（logits）を返す。
- [ ] 大きすぎるファイルを拒否する代わりに、自動で縮小・短縮できるようにする（任意）。
- [ ] より大きな Clef モデルに対応する。
- [ ] Decision Index の公式ベンチマークを実行する。
- [ ] Python が不要なインストーラーを用意する。

## 開発者向け

```sh
make install   # 開発用にインストール
make test      # 短時間のテスト（モデル不要）
make run       # サーバーを起動
make request   # examples/request.json をサーバーに送信
make help      # すべてのコマンドを表示
```

元のモデルとの結果比較（`make test-model`、`make test-images`、`make test-videos`）には、元のモデルのファイルが必要です。モデルの作成とアップロード用のスクリプトは [`scripts/`](scripts/) にあります。

## ライセンス

Clef-Flash は、Cloudflare が Qwen の Qwen3.5-9B をもとに作ったモデルで、どちらも Apache-2.0 ライセンスです。ダウンロードしたモデルには `LICENSE` と `NOTICE` が含まれ、[`src/clef_ane/_vendor/`](src/clef_ane/_vendor/) にある Cloudflare のコードも元のライセンスのままです。このプロジェクトは Cloudflare、Qwen、Apple の公式プロジェクトではありません。
