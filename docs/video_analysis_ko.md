# TFT 녹화 영상 승부 예측 실행 방법

이 기능은 실제 게임 중 실시간으로 동작하지 않는다. TFT 플레이를 녹화한 뒤 영상 파일을 오프라인으로 분석하여 시간대별 예상 등수와 Top 4 확률을 출력한다.

## 1. 의존성 설치

```powershell
cd C:\Users\user\Desktop\DB\deeplearning
python -m pip install -r requirements.txt
```

## 2. TFT 녹화

OBS Studio, Windows Game Bar 또는 다른 화면 녹화 프로그램으로 TFT 플레이를 녹화한다.

권장 조건:

- 해상도: 1920×1080 또는 1280×720
- TFT 게임 화면이 영상 중앙에 보이도록 녹화
- 화면 비율이 영상 중간에 바뀌지 않도록 설정
- MP4 형식 권장

## 3. 영상 선택 앱 실행

```powershell
python -m src.tft_advisor.video_analysis_app
```

1. `영상 선택`을 누른다.
2. 녹화한 MP4, MKV, AVI, MOV 또는 WEBM 파일을 선택한다.
3. 프레임 분석 간격을 입력한다.
   - `5초`: 일반적인 전체 게임 분석
   - `2초`: 더 자세하지만 분석 시간이 증가
   - `10초`: 빠른 데모
4. `녹화 영상 분석`을 누른다.
5. 완료 후 `결과 보고서 열기`를 누른다.

## 4. 명령줄 실행

```powershell
python -m src.tft_advisor.video_analyzer "C:\Videos\tft_game.mp4" --interval 5
```

결과:

```text
outputs/video_analysis/video_analysis.json
outputs/video_analysis/video_analysis.html
```

## 5. 분석 결과

- 영상 시점
- 예상 최종 등수
- Top 4 확률
- 시각적 보드 강도
- 감지된 활성 보드 칸 수
- 1~8등 확률 분포
- 시간대별 Top 4 확률 그래프

다음 행동 추천은 녹화 영상 모드에서 표시하지 않는다.

## 6. 사용 알고리즘

1. OpenCV로 영상 프레임 추출
2. 화면 중앙 보드 영역 crop
3. 4×7 grid 분할
4. 각 칸의 HSV saturation, brightness, Canny edge density 계산
5. 프레임 차이로 전투·화면 변화량 계산
6. CNN-style board feature와 시각 보드 강도 생성
7. Autoencoder embedding, feedforward placement model로 1~8등 확률 계산
8. 1~4등 확률을 합산하여 Top 4 확률 계산

## 7. 현재 한계

- 정확한 챔피언 종류를 인식하지 않는다.
- 아이템과 증강체를 인식하지 않는다.
- 체력과 골드를 OCR로 읽지 않는다.
- 영상 중앙이 항상 보드라는 가정을 사용한다.
- 실제 영상으로 학습한 객체 탐지 모델이 아니라 시각 특징 기반 prototype이다.
- 모델 학습 데이터는 합성 데이터이므로 실제 승부 확률로 해석하면 안 된다.

정확도를 높이려면 실제 TFT 영상 프레임에 챔피언, 아이템, 체력, 골드 라벨을 붙이고 YOLO/CNN 및 OCR 모델을 별도로 학습해야 한다.
