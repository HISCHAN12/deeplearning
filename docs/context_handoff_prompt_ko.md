# 다른 컴퓨터·새 세션 전달용 프롬프트

아래 내용을 새 Codex 또는 ChatGPT 세션에 그대로 붙여 넣으면 현재 프로젝트 맥락을 이어갈 수 있다.

---

다음 GitHub 저장소를 기준으로 작업해줘.

`https://github.com/HISCHAN12/deeplearning`

로컬에서 clone한 뒤 반드시 저장소 루트에서 작업해야 한다.

```powershell
git clone https://github.com/HISCHAN12/deeplearning.git
cd deeplearning
python -m pip install -r requirements.txt
```

## 프로젝트 목적

Deep Learning 수업 Term Project이며 주제는 TFT 게임 데이터와 녹화 영상을 이용한 최종 등수 및 Top 4 확률 예측이다.

현재 가장 중요한 기능은 **TFT 녹화 영상 자동 분석**이다. 라이브 게임 분석이 아니라 플레이가 끝난 녹화 영상을 처음부터 끝까지 일정 간격으로 분석한다.

```text
0초 프레임 분석
5초 프레임 분석
10초 프레임 분석
...
영상 종료
시간대별 예상 등수와 Top 4 확률 HTML/JSON 보고서 생성
```

녹화 영상 모드에서는 각 시점의 시각적 보드 강도와 변화량을 이용한 참고 행동 추천도 표시한다. 이 추천은 실제 체력·골드·아이템 OCR 결과가 아니라 시각적 상태 proxy를 사용한다.

## 현재 구현

### 녹화 영상 분석

- 실행 UI:

```powershell
python -m src.tft_advisor.video_analysis_app
```

- CLI:

```powershell
python -m src.tft_advisor.video_analyzer "C:\path\to\tft_video.mp4" --interval 5
```

- 지원 영상: MP4, MKV, AVI, MOV, WEBM
- OpenCV로 일정 간격의 프레임 추출
- 화면 중앙을 TFT 보드 영역으로 crop
- 보드를 4×7 grid로 분할
- HSV saturation, brightness, Canny edge density 계산
- 활성 칸 수와 프레임 변화량 계산
- 1~8등 확률과 Top 4 확률 계산
- `outputs/video_analysis` 아래 JSON과 HTML 그래프 저장

### 수동 advisor prototype

- `python -m src.tft_advisor.live_advisor`
- `python -m src.tft_advisor.overlay_advisor`
- 사용자가 조합, 보드 배치, 스테이지, 체력, 골드, 레벨, 연승/연패, 보드 강도, 대기 아이템을 입력
- 직전 상태와 현재 상태의 체력·골드·레벨·보드 강도 변화량 추적
- 예상 등수, Top 4 확률, 참고 행동 출력

수동 advisor는 보조 prototype이고 최종 발표에서는 녹화 영상 자동 분석을 중심으로 설명한다.

## 사용 알고리즘

- Multi-hot encoding
- NumPy Autoencoder
- K-means clustering
- 고정 3×3 convolution kernel 기반 CNN-style board features
- OpenCV HSV/brightness/Canny edge features
- Feedforward neural network
- ReLU
- Softmax
- Cross-entropy
- Mini-batch gradient descent
- Backpropagation
- L2 weight decay
- 80/20 hold-out evaluation

## 중요 한계

- 실제 빅데이터가 아니라 1,800개 합성 데이터로 학습
- exact placement 약 23.1%, 합성 Top 4 약 67.8%
- Top 4와 Bottom 4 데이터가 모두 포함되도록 합성 점수 분포를 보정함
- 해당 성능은 합성 규칙을 복원한 결과이며 실제 TFT 성능이 아님
- 영상에서 정확한 챔피언과 아이템을 인식하지 않음
- 체력과 골드 OCR이 없음
- 실제 학습형 CNN/YOLO가 아니라 OpenCV 시각 특징과 고정 CNN-style feature 사용
- 영상 중앙이 TFT 보드라는 가정
- 라이브 게임 자동 분석 또는 라이브 추천은 구현하지 않음

## 핵심 파일

```text
README.md
docs/video_analysis_ko.md
docs/video_script_ko.md
src/tft_advisor/game_state.py
src/tft_advisor/models.py
src/tft_advisor/recommender.py
src/tft_advisor/synthetic_data.py
src/tft_advisor/video_analyzer.py
src/tft_advisor/video_analysis_app.py
src/tft_advisor/overlay_advisor.py
tests/test_advisor.py
```

## 검증 명령

```powershell
python -m unittest discover -s tests
python -m src.tft_advisor.train_demo
```

테스트에는 영상 생성, 프레임 분석, JSON/HTML 보고서 생성까지 포함되어 있다.

## 과제 제출 조건

- public GitHub 링크 제출
- GitHub Pages 프로젝트 설명
- 약 7분 YouTube 발표 영상
- introduction, task and method, experiments, result analysis, conclusion, references 형식

현재 GitHub Pages와 YouTube 링크는 실제 활성화·업로드 여부를 다시 확인해야 한다. `README.md`와 `index.html`의 `YOUR_VIDEO_ID`는 영상 업로드 후 실제 ID로 교체해야 한다.

## 새 세션에서 먼저 할 일

1. `git status`와 `git log --oneline -5` 확인
2. README와 위 핵심 파일 읽기
3. `python -m pip install -r requirements.txt`
4. `python -m unittest discover -s tests`
5. 녹화 영상 분석 앱 실행
6. 실제 TFT 녹화 영상으로 crop 영역과 분석 결과 확인
7. 과장 없이 현재 한계를 유지하면서 개선

현재 구현이 실제 챔피언, 아이템, 체력, 골드를 정확히 자동 인식하는 것처럼 설명하면 안 된다. 정확한 표현은 **“OpenCV 시각 특징 기반 TFT 녹화 영상 자동 분석 및 시간대별 승부 예측 prototype”**이다.

---
