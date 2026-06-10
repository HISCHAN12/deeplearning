# TFT 딥러닝 Term Project 제출 계획

## 1. 추천하는 것

최종 프로젝트 주제는 다음으로 잡는 것을 추천한다.

**TFT 게임 상태 기반 승부 예측 및 다음 행동 추천 AI**

핵심 추천 기능은 다음 3개로 좁히는 것이 좋다.

1. **Top 4 / 최종 등수 예측**
   현재 조합, 아이템, 증강체, 보드 배치를 입력하면 최종 등수 확률을 예측한다.

2. **메타 덱 클러스터링**
   Autoencoder로 고차원 덱 정보를 저차원 embedding으로 압축하고, 비슷한 덱들을 meta-deck cluster로 묶는다.

3. **다음 행동 추천**
   가장 가까운 클러스터 안에서 과거에 좋은 성적을 낸 행동을 찾고, `level_up`, `roll_down`, `hold_economy`, `slam_item`, `reposition_carry` 중 하나를 추천한다.

수업 과제 기준으로는 실시간 완성형 게임 보조 프로그램보다, 딥러닝 개념을 명확히 보여주는 프로토타입이 더 안전하다. 구현 범위는 “데이터 수집 구조 + 합성 데이터 실험 + 추천 데모”까지가 적절하다.

## 2. 내가 해야 할 것

제출 전 해야 할 일은 다음 순서가 좋다.

1. GitHub 저장소를 public으로 만든다.
2. `README.md`의 이름, GitHub Pages URL, YouTube URL을 실제 값으로 바꾼다.
3. GitHub Pages를 켠다.
   저장소 설정에서 Pages source를 root 또는 GitHub Actions로 설정한다.
4. 데모를 실행한다.

```bash
python3 -m src.tft_advisor.train_demo
```

실시간 오버레이형 데모는 다음 명령으로 실행한다.

```bash
python3 -m src.tft_advisor.live_advisor
```

브라우저에서 `http://127.0.0.1:8000`을 열면 2초마다 추천 결과가 갱신된다.

5. 출력 결과와 `outputs/demo_result.json`을 영상에서 보여준다.
6. 7분 발표 영상을 녹화하고 YouTube에 업로드한다.
7. README와 `index.html`에 YouTube 링크를 넣는다.
8. 최종적으로 public GitHub 링크를 제출한다.

## 3. 실제 구현 방식 결정

### 구현 결정: Windows 항상 위 수동 입력 오버레이

현재 구현은 `tkinter` 기반의 반투명 always-on-top 창을 TFT 위에 표시한다. 사용자는 오버레이에서 meta-deck과 board positioning을 직접 선택하고, 모델은 예측 등수, Top 4 확률, 군집, 추천 행동을 갱신한다.

이유는 다음과 같다.

- 별도 외부 GUI 패키지 없이 Python 기본 `tkinter`로 실행할 수 있다.
- Riot API key나 실시간 보드 접근이 없어도 안정적인 데모가 가능하다.
- TFT를 창 모드 또는 테두리 없는 창 모드로 실행하면 게임 위에 계속 표시된다.
- 상단 드래그, 접기, 종료, 수동 상태 선택을 지원한다.

실행 명령은 `python -m src.tft_advisor.overlay_advisor`이다. Riot/League 클라이언트 lockfile을 감지하면 연결 상태를 표시한다. 하지만 추천에 필요한 세부 보드 상태는 안정적인 과제 데모를 위해 수동 선택값을 사용한다. 완전 자동 보드 인식은 현재 구현 범위가 아니다.

### 발표에서 보여줄 데모 흐름

1. 현재 보드 상태 예시를 보여준다.
2. Autoencoder가 덱 벡터를 embedding으로 압축한다고 설명한다.
3. 비슷한 덱들이 cluster로 묶이는 구조를 설명한다.
4. CNN-style board encoder가 배치 정보를 feature로 바꾼다고 설명한다.
5. Feedforward network가 등수 확률을 예측한다고 설명한다.
6. 같은 cluster의 과거 match 결과를 분석해 다음 행동을 추천한다고 설명한다.
7. 최종 출력 예시를 보여준다.

## 4. 7분 영상 구성안

- 0:00-0:40: 프로젝트 소개와 문제 정의
- 0:40-1:30: TFT에서 왜 딥러닝이 필요한지 설명
- 1:30-2:30: 데이터 표현 방식 설명
- 2:30-3:30: Autoencoder와 meta-deck clustering 설명
- 3:30-4:20: CNN-style board feature와 feedforward placement predictor 설명
- 4:20-5:20: action recommendation 방식 설명
- 5:20-6:20: 코드 실행 및 결과 시연
- 6:20-7:00: 한계점, 실제 Riot API 확장, 결론

## 5. 한계점과 향후 개선

현재 구현은 합성 데이터를 사용한다. 이는 Riot Developer API key 없이도 채점자가 실행할 수 있게 하기 위한 선택이다.

향후 개선 방향은 다음과 같다.

- Riot API로 실제 match log 수집
- set별 champion, trait, augment schema 자동 업데이트
- PyTorch 기반 autoencoder와 classifier로 확장
- 실제 게임 중 수동 입력 또는 screen recognition과 연결
- 추천 결과를 웹 dashboard 또는 OBS overlay로 표시
