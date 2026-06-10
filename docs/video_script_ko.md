# 7분 발표 스크립트 초안

안녕하세요. 제 프로젝트는 Teamfight Tactics, 즉 TFT 게임에서 현재 조합과 보드 상태를 보고 최종 등수를 예측하고, 다음 행동을 추천하는 딥러닝 기반 advisor입니다.

TFT는 단순히 강한 챔피언 하나를 고르는 게임이 아니라, 챔피언 조합, 특성, 아이템, 증강체, 골드 운영, 레벨업 타이밍, 그리고 보드 배치가 모두 함께 작용하는 게임입니다. 그래서 입력 차원이 높고 상호작용이 복잡합니다. 이 프로젝트는 이런 비선형적인 게임 데이터를 딥러닝 개념으로 분석하는 것을 목표로 합니다.

첫 번째 단계는 덱 상태를 벡터로 표현하는 것입니다. 챔피언, 특성, 아이템, 증강체를 multi-hot vector로 만들면 하나의 게임 상태가 고차원 벡터가 됩니다. 하지만 이 벡터는 그대로 쓰기에는 sparse하고 차원이 크기 때문에, Deep Learning textbook Chapter 14의 Autoencoder 개념을 사용합니다. Autoencoder는 입력 벡터를 낮은 차원의 latent embedding으로 압축한 뒤 다시 복원하도록 학습합니다. 이 latent embedding은 비슷한 덱끼리 가까워지는 표현 공간으로 사용할 수 있습니다.

두 번째 단계는 meta-deck clustering입니다. Autoencoder에서 나온 embedding에 k-means clustering을 적용해서 비슷한 조합을 같은 cluster로 묶습니다. 예를 들어 fast 8 carry 조합, reroll duelist 조합, bruiser frontline 조합처럼 운영 방식이 비슷한 덱들이 하나의 그룹으로 모이게 됩니다.

세 번째 단계는 보드 배치 정보입니다. TFT 보드는 4x7 grid로 표현할 수 있습니다. 저는 CNN Chapter 9의 아이디어를 참고해서 고정된 convolution kernel을 보드 위에 적용하고, 앞라인 밀도, 뒷라인 캐리 위치, 좌우 분포 같은 spatial feature를 추출했습니다. 이 부분은 학습 가능한 CNN layer가 아니라 CNN-style feature extractor이며, 향후 PyTorch CNN으로 확장할 수 있습니다.

네 번째 단계는 등수 예측입니다. Autoencoder embedding과 board feature를 합친 뒤, 작은 feedforward neural network로 최종 placement probability를 예측합니다. 1등부터 4등까지의 확률을 더해 Top 4 probability도 계산합니다. 또한 Chapter 7의 regularization과 Chapter 8의 optimization 개념을 반영해 weight decay와 mini-batch gradient descent를 사용했습니다.

마지막 단계는 action recommendation입니다. 현재 덱이 어느 cluster에 가까운지 찾고, 그 cluster 안에서 과거에 어떤 행동이 평균 등수와 Top 4 rate를 개선했는지 분석합니다. 추천 후보는 level up, roll down, hold economy, slam item, reposition carry입니다. 예를 들어 bruiser frontline cluster에서는 hold economy가 좋은 평균 성적을 냈다면, advisor는 hold economy를 추천합니다.

이제 데모를 보겠습니다. 터미널에서 `python3 -m src.tft_advisor.train_demo`를 실행하면 합성 match log가 생성되고, 데이터가 80퍼센트 학습용과 20퍼센트 평가용으로 분리됩니다. 출력에는 reconstruction loss, hold-out placement accuracy, hold-out Top 4 accuracy, sample archetype, predicted placement, predicted Top 4 probability, recommended action이 표시됩니다. 고정 seed 기준 hold-out Top 4 accuracy는 약 0.71이며, 정확한 등수 accuracy는 약 0.26입니다.

이 프로젝트의 실제 데모 방식은 Windows always-on-top 오버레이입니다. `python -m src.tft_advisor.overlay_advisor`를 실행하면 반투명한 오버레이가 TFT 위에 표시됩니다. 오버레이에서 현재 조합과 보드 배치를 선택하면 예측 결과가 즉시 갱신되고, 이후 2초마다 클라이언트 상태와 결과를 다시 확인합니다. Riot 클라이언트가 실행 중이면 local lockfile을 감지해 연결 상태를 표시하지만, 세부 보드 상태는 안정적인 데모를 위해 수동 선택값을 사용합니다.

한계점은 현재 데이터가 Riot API 실제 match log가 아니라 재현 가능한 합성 데이터라는 점입니다. 또한 합성 데이터에는 조합별로 미리 정의한 행동 효과가 들어 있으므로, 추천 결과는 통제된 데이터에서 알려진 구조를 복원하는지 확인하는 실험입니다. 실제 게임에서 행동이 성적을 개선한다는 인과적 증거는 아닙니다. 향후에는 Riot API로 match log를 수집하고, PyTorch 모델로 확장하고, 실제 게임 중 수동 입력 또는 화면 인식과 연결하는 방향으로 발전시킬 수 있습니다.

결론적으로 이 프로젝트는 TFT라는 복잡한 전략 게임에 Autoencoder, CNN-style spatial features, Feedforward Network, regularization, optimization을 적용해 승부 예측과 행동 추천을 수행하는 딥러닝 프레임워크입니다.
