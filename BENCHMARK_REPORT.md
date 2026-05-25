# MNIST 벤치마크 및 발표 정리

이 문서는 `REPORT.md`에 바로 합치기 전, 현재 모델 설정과 실험 결과를 따로 기록하기 위한 작업용 레포트입니다. 모델 실행과 결과 확인은 `mnist_lab.ipynb`를 기준으로 합니다.

## 1. 기준 모델 구조

현재 기준 모델은 입력 784차원 이미지를 두 개의 은닉층을 거쳐 10개 숫자 클래스 확률로 변환합니다.

| 구분 | 설정 |
| --- | --- |
| 입력 | 784차원, 28x28 이미지를 flatten, 0~1 정규화 |
| 은닉층 1 | Affine(784 -> 512) -> BatchNorm -> ReLU -> Dropout |
| 은닉층 2 | Affine(512 -> 256) -> BatchNorm -> ReLU -> Dropout |
| 출력층 | Affine(256 -> 10) -> Softmax |
| 손실 함수 | Cross Entropy Loss |
| 총 학습 파라미터 수 | 537,354 |

파라미터 수에는 `W`, `b`, BatchNorm의 `gamma`, `beta`가 포함됩니다. BatchNorm의 `running_mean`, `running_var`와 Adam의 `m`, `v`는 학습 중 사용하는 상태값이지만 모델 파라미터 수에는 포함하지 않습니다.

## 2. 현재 하이퍼파라미터

### 데이터 및 학습 루프

| 항목 | 값 |
| --- | --- |
| 실행 파일 | `mnist_lab.ipynb` |
| 학습 데이터 | MNIST train 60,000개 |
| 테스트 데이터 | MNIST test 10,000개 |
| 입력 전처리 | `float32`, flatten 후 `/ 255.0` 정규화 |
| label 형식 | one-hot이 아닌 정수 label, 0~9 |
| epochs | 20 |
| batch_size | 128 |
| batch 순서 | epoch마다 `np.random.permutation()`으로 shuffle |
| random seed | 현재 노트북 기준 명시적으로 고정하지 않음 |

### 모델 설정

| 항목 | 값 |
| --- | --- |
| hidden_size_list | `[512, 256]` |
| activation | ReLU |
| use_batchnorm | `True` |
| BatchNorm momentum | `0.9` |
| use_dropout | `True` |
| Dropout ratio | `0.5` |
| weight initialization | He 초기화, `sqrt(2 / fan_in)` |
| bias initialization | 0 |
| BatchNorm gamma/beta 초기화 | `gamma=1`, `beta=0` |
| weight_decay_lambda | `0` |

### Optimizer 설정

| 항목 | 값 |
| --- | --- |
| optimizer | Adam |
| learning rate | `0.001` |
| beta1 | `0.9` |
| beta2 | `0.999` |
| epsilon | `1e-7` |

Adam의 `beta1`, `beta2`는 gradient와 gradient 제곱의 이동평균을 얼마나 오래 유지할지 정하는 값입니다. BatchNorm의 `momentum=0.9`와 이름은 비슷하지만, Adam momentum은 optimizer 내부 상태 업데이트에 쓰이고 BatchNorm momentum은 추론용 running mean/var 누적에 쓰입니다.

## 3. 벤치마크 기록

현재 노트북 실행 결과 기준 baseline은 다음과 같습니다.

| Run | Activation | 모델/설정 | Optimizer | epochs | batch_size | Test Accuracy | Total Params |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Baseline-A | ReLU | BN=True, Dropout=True, Dropout ratio=0.5, He init | Adam(lr=0.001, beta1=0.9, beta2=0.999) | 20 | 128 | **98.44%** | **537,354** |
| Step-A | Step Function | Baseline-A에서 은닉층 activation만 Step으로 변경 | Adam(lr=0.001, beta1=0.9, beta2=0.999) | 20 | 128 | 77.29% | 537,354 |
| Sigmoid-A | Sigmoid | Baseline-A에서 은닉층 activation만 Sigmoid로 변경, He init 유지 | Adam(lr=0.001, beta1=0.9, beta2=0.999) | 20 | 128 | 97.79% | 537,354 |

손실 곡선은 `mnist_lab.ipynb`의 `plot_loss_history(loss_history)` 출력으로 확인합니다. 현재 기준 실험에서는 학습 후 테스트 정확도 98% 이상을 달성했으므로, 기본 구조와 optimizer 설정은 MNIST 분류에 충분히 안정적으로 수렴한 것으로 볼 수 있습니다.

### Activation Function 비교 관찰 포인트

| Run | Forward 출력 | Backward gradient | 주로 업데이트되는 파라미터 | 예상 차이 |
| --- | --- | --- | --- | --- |
| Baseline-A | 음수는 0, 양수는 그대로 통과 | 양수였던 위치로 gradient 전달 | 모든 Affine/BatchNorm 파라미터 | 은닉층 feature가 학습되며 높은 정확도 |
| Step-A | 0 또는 1 binary activation | 거의 모든 위치에서 0으로 가정 | 마지막 `Affine3` 중심 | 은닉 feature extractor가 고정되어 ReLU보다 크게 낮은 성능 예상 |
| Sigmoid-A | 0~1 사이의 연속값 | `sigmoid(x) * (1 - sigmoid(x))`로 gradient 전달 | 모든 Affine/BatchNorm 파라미터 | 학습은 가능하지만 포화 구간 때문에 ReLU보다 느리거나 낮은 성능 예상 |

Step Function은 실제로 `x=0`에서 미분 불가능하고 그 외 구간의 미분값은 0입니다. 이번 구현에서는 backward를 `np.zeros_like(dout)`로 두었으므로 `Affine1`, `BatchNorm1`, `Affine2`, `BatchNorm2` 쪽 gradient는 사실상 막힙니다. 다만 마지막 `Affine3`는 Step 출력 이후에 있으므로 `W3`, `b3`는 업데이트됩니다. 따라서 성능은 random guess인 10%에 반드시 고정되지는 않지만, ReLU baseline보다 크게 낮을 것으로 예상됩니다.

Sigmoid는 Step Function과 달리 미분 가능하므로 은닉층 앞쪽까지 gradient가 전달됩니다. 다만 출력이 0~1 범위로 압축되고 입력 절댓값이 커질수록 미분값이 0에 가까워지는 포화 문제가 있어, ReLU보다 수렴이 느리거나 최종 정확도가 낮아질 수 있습니다. 이번 Sigmoid-A는 activation만 바꾼 비교를 위해 Xavier 초기화로 바꾸지 않고 Baseline-A와 같은 He 초기화를 유지합니다.

## 4. 추가 실험 기록 양식

추가 실험을 진행할 경우 아래 표에 같은 형식으로 누적 기록합니다.

| Run | 변경점 | Optimizer | epochs | batch_size | Test Accuracy | 관찰 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Baseline-A | BN=True, Dropout=True, ratio=0.5 | Adam lr=0.001 | 20 | 128 | 98.44% | 기준 성능 |
| Step-A | Activation을 Step Function으로 변경 | Adam lr=0.001 | 20 | 128 | 77.29% | 은닉층 gradient 차단 영향 확인 |
| Exp-C | Dropout 제거 | Adam lr=0.001 | 20 | 128 | (기입) | regularization 영향 확인 |
| Exp-D | Optimizer를 SGD로 변경 | SGD lr=(기입) | 20 | 128 | (기입) | Adam 대비 수렴 속도 비교 |
| Sigmoid-A | Activation을 Sigmoid로 변경 | Adam lr=0.001 | 20 | 128 | 97.79% | ReLU/Step과 gradient 흐름 비교 |

## 5. Gradient Vanishing 진단 기록

이 진단 실험은 전체 학습을 다시 수행하지 않고, 같은 mini-batch 하나에서 ReLU 모델과 Sigmoid 모델을 각각 한 번 forward/backward 한 뒤 layer별 `dW`의 L2 norm을 비교합니다. Dropout은 랜덤 mask 영향을 줄이기 위해 끄고, BatchNorm은 현재 모델 구조와 맞추기 위해 유지합니다.

| Activation | `||dW1||` | `||dW2||` | `||dW3||` | 관찰 메모 |
| --- | ---: | ---: | ---: | --- |
| ReLU | 2.950720e+00 | 2.069579e+00 | 1.804833e+00 | 기준 gradient 흐름 |
| Sigmoid | 9.426143e-01 | 7.321449e-01 | 1.546690e+00 | `W1`, `W2` gradient가 ReLU보다 작게 관찰됨 |

Sigmoid는 Step Function과 달리 미분 가능하므로 gradient가 완전히 끊기지는 않습니다. 다만 `sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)) <= 0.25`이기 때문에 여러 층을 지나며 gradient가 작아질 수 있습니다. 현재 모델은 은닉층이 2개이고 BatchNorm을 사용하므로 gradient vanishing이 완화되어 극단적으로 보이지 않을 수 있으며, 이 경우에는 ReLU와 Sigmoid의 gradient norm 비율과 loss curve를 함께 해석합니다.

이번 진단에서는 Sigmoid의 `W1` gradient가 ReLU 대비 약 31.9%, `W2` gradient가 약 35.4% 수준으로 작았습니다. 반면 `W3`는 ReLU 대비 약 85.7% 수준으로 비교적 덜 줄었습니다. 이는 Sigmoid가 출력층 가까운 곳보다 앞쪽 은닉층에서 gradient를 더 약하게 전달할 수 있음을 보여줍니다.

## 6. 실험 환경 메모

| 항목 | 내용 |
| --- | --- |
| 실행 방식 | `mnist_lab.ipynb`에서 데이터 로드, 학습, 평가 순서로 실행 |
| Python | 노트북 테스트 출력 기준 Python 3.11.14 |
| 주요 라이브러리 | NumPy, Matplotlib, pytest |
| requirements | `numpy>=1.24`, `matplotlib>=3.7`, `pytest>=7.0` |
| 학습 소요 시간 | 약 4분 |

## 7. 발표용 핵심 요약

- 모델은 784 -> 512 -> 256 -> 10 구조의 MLP이며, 은닉층마다 BatchNorm, ReLU, Dropout을 적용했습니다.
- BatchNorm은 학습 중 batch 통계로 activation 분포를 안정화하고, 추론 시에는 누적된 running mean/var를 사용합니다.
- Dropout은 학습 중 일부 뉴런을 랜덤하게 꺼서 과적합을 줄이는 regularization 역할을 합니다.
- Adam은 gradient 이동평균과 제곱 이동평균을 함께 사용해 SGD보다 빠르고 안정적인 수렴을 기대할 수 있습니다.
- Baseline-A는 `epochs=20`, `batch_size=128`, `Adam(lr=0.001)` 설정에서 테스트 정확도 98.44%를 기록했습니다.
- Sigmoid-A는 같은 조건에서 97.79%를 기록했으며, ReLU보다 낮지만 Step-A보다는 훨씬 안정적으로 학습되었습니다.
- seed가 고정되어 있지 않으므로 재실행 시 정확도는 소폭 달라질 수 있습니다.

## 8. 다음 실험 제안

Baseline-A는 98% 이상의 테스트 정확도를 달성했으므로 기본 모델 구조는 MNIST 분류에 적합합니다. 다음 단계에서는 Dropout ratio, BatchNorm 사용 여부, optimizer 종류, learning rate를 바꿔가며 수렴 속도와 테스트 정확도의 차이를 비교하면 발표에서 더 설득력 있는 ablation 결과를 제시할 수 있습니다.
