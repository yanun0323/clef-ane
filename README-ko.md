# clef-ane

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-Click-yellow" alt="English"></a>
  <a href="README-tw.md"><img src="https://img.shields.io/badge/繁體中文-點擊查看-orange" alt="繁體中文"></a>
  <a href="README-cn.md"><img src="https://img.shields.io/badge/简体中文-点击查看-orange" alt="简体中文"></a>
  <a href="README-ja.md"><img src="https://img.shields.io/badge/日本語-クリック-blue" alt="日本語"></a>
  <a href="README-ko.md"><img src="https://img.shields.io/badge/한국어-클릭-yellow" alt="한국어"></a>
</p>

clef-ane는 Cloudflare의 [Clef-Flash](https://huggingface.co/Cloudflare/clef-flash)를 Mac에서 실행하는 도구입니다. Clef-Flash는 판단을 내려 주는 모델입니다. 상황을 설명하고 질문하면, 각 답이 얼마나 그럴듯한지 알려 줍니다. 상황은 텍스트, 이미지, 짧은 동영상으로 설명할 수 있습니다.

Mac의 Neural Engine(ANE)이라는 AI 전용 칩에서 실행되므로, 그래픽 칩(GPU)은 다른 작업에 그대로 쓸 수 있습니다. 모든 처리는 Mac 안에서 이루어지며 인터넷 연결이 필요 없습니다.

실험적인 프로젝트입니다. Apple이 공식적으로 지원하지 않는 macOS 기능을 사용하므로, macOS 업데이트 후 작동하지 않을 수 있습니다.

## 할 수 있는 일

- 세 가지 질문에 답합니다: 예/아니요, 하나 고르기, 단계로 평가하기.
- 한 요청에 텍스트, JSON, 이미지 최대 4장, 짧은 동영상 1개를 넣을 수 있습니다.
- GPU를 전혀 사용하지 않습니다.
- 한 번 내려받으면 오프라인으로 쓸 수 있습니다.
- 요청 형식이 Cloudflare의 SystemOne API와 같습니다.

## 준비물

- Apple Silicon Mac. 지금까지는 macOS 27을 실행하는 M5 Pro에서만 테스트했습니다.
- Python 3.12 이상.
- Apple 명령줄 도구: `xcode-select --install`을 실행하세요.
- 약 10 GiB의 여유 디스크 공간.

## 설치

프로젝트 폴더에서 다음을 실행합니다.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

## 시작하기

모델(약 9.74 GiB)을 한 번 내려받은 뒤 서버를 시작합니다.

```sh
clef-ane download
clef-ane serve
```

- Hugging Face 계정은 필요 없습니다. 다운로드가 멈추면 같은 명령을 다시 실행해 이어서 받으세요.
- 시작하는 데 약 1.5분이 걸립니다. `Listening on http://127.0.0.1:8080`이 보이면 준비된 것입니다.
- 이미지나 동영상이 들어간 첫 요청은 약 10초 더 걸립니다. 그 뒤로는 빠릅니다.
- `Ctrl-C`를 누르면 멈춥니다.

> [!WARNING]
> 서버에는 비밀번호가 없습니다. 기본적으로 내 Mac에서만 접속할 수 있습니다. 믿을 수 있는 네트워크가 아니라면 다른 컴퓨터에 열지 마세요.

## 질문하기

다른 터미널 창에서 다음을 실행합니다.

```sh
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request.json
```

[`examples/request.json`](examples/request.json)은 요금이 두 번 청구된 고객에 대해 질문 세 개를 보냅니다. 다음과 같은 결과를 받습니다.

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

즉, 이 티켓은 결제 담당(billing)에게 보내야 하고(가능성 96%), 고객은 환불을 요청하고 있으며(89%), 긴급도는 "Urgent"일 가능성이 가장 높다는 뜻입니다.

### 요청 작성 방법

요청에는 `state`(상황. 텍스트 또는 JSON)와 최대 64개의 `questions`를 넣습니다. 각 질문에는 다음 항목이 있습니다.

| 항목 | 의미 |
|---|---|
| `type` | `noul`은 예/아니요, `choice`는 하나 고르기, `score`는 단계 평가 |
| `instructions` | 질문 내용. 생략할 수 있습니다 |
| `criteria` | 선택지. `choice`는 이름과 설명, `score`는 낮은 것부터 높은 것 순서의 목록 |

`noul`의 답은 "예"일 확률입니다. `choice`와 `score`는 모든 선택지의 확률을 알려 줍니다.

## 이미지와 동영상

예제 스크립트는 Mac에 있는 파일을 읽어서 보냅니다.

```sh
python examples/image_request.py receipt.png --question "Is the receipt marked as paid?"
python examples/video_request.py clip.mp4 --question "Does the person enter the room?"
```

직접 파일을 보내려면 base64로 파일을 글자로 바꾼 뒤 요청에 넣으세요.

```json
{
  "state": "Decide from the attached image.",
  "images": ["data:image/png;base64,iVBORw0..."],
  "questions": {"paid": {"type": "noul", "instructions": "Is the receipt marked as paid?"}}
}
```

동영상은 `"videos": ["data:video/mp4;base64,..."]`를 사용합니다. MOV는 `video/quicktime`, WebM은 `video/webm`입니다.

| | 이미지 | 동영상 |
|---|---|---|
| 파일 형식 | PNG, JPEG, WebP | MP4, MOV, WebM |
| 개수 | 최대 4장 | 1개 |
| 파일 크기 한도 | 장당 4 MiB, 합계 8 MiB | 16 MiB |
| 크기·길이 한도 | 장당 1,600만 화소 | 30초 |
| 모델이 실제로 보는 것 | 각 이미지를 약 512×512로 축소 | 초당 약 2장면, 각 장면을 약 256×256으로 축소 |

**파일이 너무 크면 오류가 납니다.** 한도 안의 이미지와 동영상은 자동으로 축소됩니다(잘라 내지 않습니다). 한도를 넘으면 받지 않으며, 자동으로 줄이거나 압축해 주지 않습니다. 긴 동영상은 잘라 내고 큰 파일은 압축한 뒤 보내세요.

그 밖에 알아 둘 점:

- 동영상의 소리는 사용하지 않습니다.
- 이미지는 축소되고 동영상은 일부 장면만 쓰기 때문에, 작은 글씨나 빠른 움직임은 놓칠 수 있습니다.
- 링크나 파일 경로는 쓸 수 없습니다. 파일 내용을 요청에 넣어 보내세요.

## 옵션

| 옵션 | 기본값 | 하는 일 |
|---|---|---|
| `--port` | `8080` | 사용할 포트 번호 |
| `--host` | `127.0.0.1` | 접속을 받을 주소. 기본값은 내 Mac에서만 접속을 허용합니다 |
| `--max-tokens` | `16384` | 받을 수 있는 가장 긴 입력. 숫자가 작을수록 메모리를 덜 씁니다 |
| `--max-body-mb` | `32` | 요청 크기 한도(MiB) |
| `--vision-device` | `ane` | 이미지와 동영상을 처리할 곳. `ane`은 GPU를 쓰지 않고, `mps`는 GPU를 써서 조금 더 빠르며, `cpu`가 가장 느립니다 |
| `--model-dir` | `~/Library/Caches/clef-ane` 안 | 모델 저장 위치 |

모든 옵션은 `clef-ane --help`로 볼 수 있습니다. `CLEF_ANE_HOME`을 설정해 모델 저장 위치를 바꿀 수도 있습니다.

이전의 텍스트 전용 모델(약 8.90 GiB)도 계속 쓸 수 있습니다.

```sh
clef-ane download --revision text-v1 --model-dir ~/clef-text-model
clef-ane serve --model-dir ~/clef-text-model
```

## 현재 한계

- 요청은 한 번에 하나씩 처리합니다. 하나 더 기다릴 수 있고, 그 이상은 "사용 중" 오류를 받습니다.
- 27개 테스트 질문에서는 원래 모델과 같은 답을 냈습니다. 하지만 Clef-Flash 공식 벤치마크는 돌려 보지 않았으므로 전체 정확도는 다를 수 있습니다.
- 이미지와 동영상은 원래 모델이 다룰 수 있는 것보다 낮은 해상도로 사용하므로, 세부 내용이 많은 사진은 덜 정확할 수 있습니다.
- 주어진 선택지에 점수를 매길 뿐, 글을 쓰지는 않습니다.

## 할 일

원래 Clef-Flash는 할 수 있지만 clef-ane는 아직 못 하는 것:

- [ ] 여러 요청을 한꺼번에 처리하기.
- [ ] 더 선명한 이미지와 4장을 넘는 이미지 사용하기.
- [ ] 더 긴 동영상, 더 많은 장면, 여러 개의 동영상 사용하기.
- [ ] 이미지와 동영상 처리 방식을 조정할 수 있게 하기(`media_kwargs`).
- [ ] 상황 설명 텍스트를 어디까지 읽을지 제한할 수 있게 하기(`max_state_tokens`).
- [ ] 각 답의 바탕이 되는 원시 점수(logits) 돌려주기.
- [ ] 너무 큰 파일을 거부하는 대신 원하면 자동으로 줄이거나 자르기.
- [ ] 더 큰 Clef 모델 지원하기.
- [ ] Decision Index 공식 벤치마크 돌리기.
- [ ] Python 없이 쓸 수 있는 설치 프로그램 만들기.

## 개발자용

```sh
make install   # 개발용으로 설치
make test      # 빠른 테스트, 모델 필요 없음
make run       # 서버 시작
make request   # examples/request.json을 서버로 보내기
make help      # 모든 명령 보기
```

원래 모델과 결과를 비교하려면(`make test-model`, `make test-images`, `make test-videos`) 원래 모델 파일이 필요합니다. 모델을 만들고 올리는 스크립트는 [`scripts/`](scripts/)에 있습니다.

## 라이선스

Clef-Flash는 Cloudflare가 Qwen의 Qwen3.5-9B를 바탕으로 만든 모델이며, 둘 다 Apache-2.0 라이선스입니다. 내려받은 모델에는 `LICENSE`와 `NOTICE`가 들어 있고, [`src/clef_ane/_vendor/`](src/clef_ane/_vendor/)의 Cloudflare 코드도 원래 라이선스를 그대로 따릅니다. 이 프로젝트는 Cloudflare, Qwen, Apple의 공식 프로젝트가 아닙니다.
