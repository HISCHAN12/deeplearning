# 다른 데스크탑에서 실행하는 방법

이 문서는 TFT가 이미 설치된 다른 컴퓨터에서 프로젝트를 실행하고, 게임 실행 상태를 확인하는 절차이다.

## 1. 저장소 받기

```bash
git clone https://github.com/HISCHAN12/deeplearning.git
cd deeplearning
```

## 2. Python 의존성 설치

```bash
python3 -m pip install -r requirements.txt
```

필요한 외부 라이브러리는 `numpy`뿐이다.

## 3. 기본 딥러닝 데모 실행

```bash
python3 -m src.tft_advisor.train_demo
```

정상 실행되면 다음 정보가 출력된다.

- Autoencoder reconstruction loss
- Placement prediction accuracy
- Top 4 accuracy
- 추천 action
- `outputs/demo_result.json`

## 4. 로컬 실시간 advisor 실행

```bash
python3 -m src.tft_advisor.live_advisor
```

브라우저에서 다음 주소를 연다.

```text
http://127.0.0.1:8000
```

화면에서 meta-deck과 board positioning을 선택하면 2초마다 추천 결과가 갱신된다.

정상 실행되면 터미널에 다음과 비슷하게 출력된다.

```text
Starting TFT live advisor...
Training local prototype model. This usually takes a few seconds.
Model ready.
TFT live advisor running at http://127.0.0.1:8000
Press Ctrl+C to stop.
```

만약 8000번 포트가 이미 사용 중이면 다음처럼 다른 포트로 실행할 수 있다.

```bash
TFT_ADVISOR_PORT=8001 python3 -m src.tft_advisor.live_advisor
```

이 경우 브라우저에서는 `http://127.0.0.1:8001`을 연다.

## 5. TFT 실행 상태 확인

1. Riot Client 또는 League of Legends/TFT를 실행한다.
2. 위의 live advisor 서버를 실행한다.
3. `http://127.0.0.1:8000`에서 상단 상태 메시지를 확인한다.

상태 메시지가 다음처럼 보이면 클라이언트 감지가 된 것이다.

```text
Riot client detected
```

클라이언트가 감지되지 않으면 다음처럼 표시된다.

```text
Manual demo mode
```

## 6. 중요한 구현 범위

현재 프로젝트는 과제 제출용 prototype이다.

가능한 것:

- 로컬 웹 advisor 실행
- TFT/League client lockfile 감지 시도
- meta-deck 선택 기반 placement 예측
- board positioning 선택 기반 추천 결과 갱신
- OBS browser source로 overlay처럼 표시

아직 자동화하지 않은 것:

- 실제 게임 화면에서 챔피언/아이템/증강체 자동 인식
- 현재 내 보드를 OCR/screen recognition으로 읽기
- TFT 클라이언트 내부 상태와 완전 자동 연동

따라서 발표에서는 다음처럼 설명하면 정확하다.

> 본 프로젝트는 실제 TFT 클라이언트와 완전 자동으로 결합된 상용 오버레이가 아니라, 로컬 웹 기반 실시간 advisor prototype이다. Riot 클라이언트 실행 여부는 감지하고, 추천 입력은 안정적인 데모를 위해 수동 선택값을 사용한다. 향후 screen recognition 또는 Riot API 연동으로 자동 보드 인식을 확장할 수 있다.

## 7. 포트가 이미 사용 중일 때

기본 주소는 `127.0.0.1:8000`이다. 만약 다른 프로그램이 8000번 포트를 사용 중이면 아래처럼 다른 포트로 실행한다.

```bash
TFT_ADVISOR_PORT=8001 python3 -m src.tft_advisor.live_advisor
```
