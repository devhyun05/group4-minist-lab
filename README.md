# MNIST Neural Network from Scratch

> 딥러닝 프레임워크 없이 NumPy로 학습과 역전파를 구현한 손글씨 분류 프로젝트

PyTorch나 TensorFlow의 학습 API를 사용하지 않았습니다. Affine layer부터 optimizer까지 직접 구현해 각 구성 요소가 수렴과 일반화에 어떤 영향을 주는지 실험했습니다.

## 구현 범위

| 영역 | 구현 내용 |
| --- | --- |
| Layer | Affine, BatchNorm, Dropout |
| Activation | ReLU, Sigmoid, Step Function, Softmax |
| Loss | Cross Entropy |
| Optimizer | SGD, Adam |
| Network | forward, backward, parameter update |
| Training | mini-batch 학습, 평가, loss 기록 |

기본 모델은 다음 구조를 사용합니다.

```text
784
→ Affine(512) → BatchNorm → ReLU → Dropout
→ Affine(256) → BatchNorm → ReLU → Dropout
→ Affine(10) → Softmax
```

## 실험 결과

| 실험 | 테스트 정확도 | 확인한 점 |
| --- | ---: | --- |
| Baseline | 98.44% | NumPy 구현만으로 안정적인 수렴 확인 |
| Learning rate 0.005 | 98.47% | 비교한 learning rate 중 가장 높은 결과 |
| Dropout + BatchNorm | 98.48% | ablation 조합 중 가장 높은 결과 |
| BatchNorm only | 98.33% | train 정확도와 test 정확도가 같지 않음을 확인 |
| Dropout only | 98.36% | regularization 효과 비교 |
| 둘 다 제외 | 97.98% | 정규화 기법 제거 시 성능 변화 확인 |

위 수치는 한 번의 seed와 실험 조건에서 나온 결과입니다. 조합별 차이가 0.5%p 이내이므로 모든 조건에서 같은 순서가 나온다고 일반화하지 않았습니다.

## 실험 기준

- 한 번에 한 조건을 바꿔 activation, optimizer, learning rate를 비교했습니다.
- 최종 정확도만 보지 않고 loss curve와 gradient norm을 함께 확인했습니다.
- Dropout과 BatchNorm 실험에서는 나머지 모델 구조와 학습 조건을 같게 유지했습니다.
- train 정확도와 test 정확도를 함께 보아 과적합 가능성을 확인했습니다.

자세한 설정과 해석은 [`REPORT.md`](./REPORT.md)와 [`Dropout·BatchNorm ablation`](./DROPOUT_BATCHNORM_ABLATION.md)에 있습니다.

## 실행

Python 3.11 기준입니다.

```bash
git clone https://github.com/devhyun05/group4-mnist-lab.git
cd group4-mnist-lab

conda create -n mnist-nn python=3.11 -y
conda activate mnist-nn
pip install -r requirements.txt
python download_mnist.py
pytest tests -q
```

학습과 시각화는 [`mnist_lab.ipynb`](./mnist_lab.ipynb)에서 실행할 수 있습니다.

## 구조

```text
src/
  data.py          데이터 로드와 전처리
  activations.py   activation과 gradient
  layers.py        Affine, BatchNorm, Dropout
  losses.py        Cross Entropy
  optimizers.py    SGD, Adam
  network.py       신경망 조립과 역전파
  training.py      학습과 평가
tests/              구성 요소 단위 테스트
analysis/           실험 결과와 plot
```

## 팀

크래프톤 정글 12기 302반 팀 프로젝트

- 이시원
- 이지섭
- 이현성
- 양은열
