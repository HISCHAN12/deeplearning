# 다른 Windows 데스크톱에서 실행하는 방법

이 문서는 TFT가 설치된 Windows PC에서 로컬 웹 advisor prototype을 실행하는 절차다.

## 1. 준비 사항

- Python 3.10 이상
- Git
- 인터넷 연결

PowerShell에서 다음 명령으로 설치 여부를 확인한다.

```powershell
python --version
git --version
```

`python`을 실행했을 때 Microsoft Store만 열리거나 버전이 표시되지 않으면 Python을 먼저 설치한다. 설치 과정에서 `Add python.exe to PATH`를 선택한다.

## 2. 저장소 받기

```powershell
git clone https://github.com/HISCHAN12/deeplearning.git
cd deeplearning
python -m pip install -r requirements.txt
```

## 3. 기본 데모와 테스트

```powershell
python -m unittest discover -s tests
python -m src.tft_advisor.train_demo
```

기본 데모는 합성 데이터를 80% 학습용과 20% 평가용으로 분리한다. 실행 후 hold-out placement accuracy, hold-out Top 4 accuracy, 추천 행동이 출력되고 `outputs/demo_result.json`이 생성된다.

## 4. 녹화 영상 자동 분석

최종 발표에서 권장하는 기능이다.

```powershell
python -m src.tft_advisor.video_analysis_app
```

영상 선택 앱에서 TFT 녹화 파일을 선택하고 분석 간격을 설정한다. 프로그램은 영상 처음부터 끝까지 해당 간격으로 프레임을 계속 추출하여 각 시점의 예상 등수와 Top 4 확률을 계산한다. 전체 분석이 끝나면 JSON과 HTML 그래프를 생성한다.

자세한 내용은 `docs/video_analysis_ko.md`를 참고한다.

## 5. 로컬 웹 advisor 실행

```powershell
python -m src.tft_advisor.live_advisor
```

정상적으로 시작되면 다음 메시지가 표시된다.

```text
Starting TFT live advisor...
Training local prototype model. This usually takes a few seconds.
Model ready.
TFT live advisor running at http://127.0.0.1:8000
Press Ctrl+C to stop.
```

브라우저에서 다음 주소를 연다.

```text
http://127.0.0.1:8000
```

화면에서 meta-deck과 board positioning을 선택하면 predicted placement, predicted Top 4 probability, cluster, recommended action이 갱신된다.

## 6. TFT 위에 오버레이로 실행

```powershell
python -m src.tft_advisor.overlay_advisor
```

오버레이 기능:

- TFT 화면 위에 항상 표시
- 반투명·테두리 없는 창
- 상단 영역을 드래그해서 이동
- `−` 버튼으로 접기
- `×` 버튼으로 종료
- 메타 덱과 보드 배치를 수동 선택
- 현재 스테이지, 체력, 골드, 레벨, 연승/연패, 보드 강도, 대기 아이템 입력
- `현재 게임 상태 반영` 버튼으로 상태 스냅샷 저장
- 직전 입력 대비 체력·골드·레벨·보드 강도 변화 추적
- 예측 등수, Top 4 확률, 군집, 추천 행동 표시
- 1~8등 확률 분포 표시
- 2초마다 클라이언트 상태와 결과 갱신

TFT는 `창 모드` 또는 `테두리 없는 창 모드`를 권장한다. 독점 전체화면에서는 Windows 데스크톱 오버레이가 게임 뒤에 가려질 수 있다.

게임에서 상태가 바뀌어도 오버레이가 자동으로 읽지는 않는다. 실제 상태에 맞게 입력값을 변경하고 `현재 게임 상태 반영`을 눌러야 한다.

예시:

1. 3-2에서 체력 78, 골드 42, 레벨 6, 보드 강도 7을 입력하고 상태를 반영한다.
2. 3-5에서 체력 54, 골드 18, 레벨 7, 보드 강도 5로 변경하고 다시 반영한다.
3. 프로그램은 체력 -24, 골드 -24, 레벨 +1, 보드 강도 -2의 흐름을 계산한다.
4. 현재 상태와 변화량을 함께 신경망에 입력하여 예측 확률과 추천 행동을 갱신한다.

## 7. 포트 변경

8000번 포트가 사용 중이면 PowerShell에서 다음과 같이 실행한다.

```powershell
$env:TFT_ADVISOR_PORT=8001
python -m src.tft_advisor.live_advisor
```

브라우저 주소는 `http://127.0.0.1:8001`이다.

## 8. Riot Client 감지

프로그램은 다음 순서로 League Client lockfile을 찾는다.

1. `TFT_RIOT_LOCKFILE` 환경 변수
2. 일반적인 Windows 설치 경로
3. 실행 중인 League Client 프로세스의 설치 폴더

League of Legends를 다른 드라이브에 설치했다면 lockfile 경로를 직접 지정한다.

```powershell
$env:TFT_RIOT_LOCKFILE="D:\Riot Games\League of Legends\lockfile"
python -m src.tft_advisor.live_advisor
```

클라이언트를 찾으면 `Riot client detected`, 찾지 못하면 `Manual demo mode`가 표시된다.

## 9. 정확한 구현 범위

현재 가능한 기능:

- 녹화 영상 전체를 일정 간격으로 자동 분석
- 시간대별 예상 등수와 Top 4 확률 보고서 생성
- 로컬 웹 advisor 실행
- 선택한 sample meta-deck과 board style 기반 예측
- 사용자가 입력한 실제 라운드·체력·골드·레벨·연승/연패·보드 강도 기반 예측
- 직전 입력과 현재 입력 사이의 게임 흐름 반영
- 8개 placement 확률과 Top 4 probability 표시
- meta-deck cluster 기반 다음 행동 추천
- Riot/League Client 실행 상태 감지 시도
- OBS browser source를 이용한 화면 배치

현재 구현되지 않은 기능:

- 라이브 게임 화면의 실시간 자동 분석
- 실제 TFT 보드의 챔피언, 아이템, 증강체 자동 인식
- 현재 게임 상태를 Riot API에서 실시간으로 가져오기
- OCR 또는 screen recognition
- 게임 위에 직접 결합되는 상용 overlay

발표에서는 다음과 같이 설명하는 것이 정확하다.

> 이 프로젝트는 실제 TFT 보드를 자동 인식하는 상용 오버레이가 아니라 수동 상태 입력형 deep learning advisor prototype입니다. Riot Client 실행 상태는 감지할 수 있고, 사용자가 입력한 현재 게임 상태와 직전 상태 대비 흐름을 모델 입력에 반영합니다.
