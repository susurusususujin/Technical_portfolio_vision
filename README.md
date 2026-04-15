# Technical_portfolio_vision
서울과학기술대학교_데이터사이언스학과_24장수진_졸업심사_테크니컬포트폴리오
# 해충 이미지 품질 분류 프로젝트

이 프로젝트는 해충 이미지의 품질을 자동으로 판별하여 **정상 이미지(OK)** 와 **오류 이미지(ERROR)** 를 구분하는 이진 분류 모델을 학습하는 코드입니다.  
`true_image` 폴더의 이미지는 정상 클래스, `false_image` 폴더의 이미지는 오류 클래스로 사용하며, 오류 이미지를 **positive class** 로 두고 학습 및 평가를 수행합니다.

코드는 **Stratified 5-Fold Cross Validation**, **Transfer Learning**, **Early Stopping**, 그리고 **검증 세트 기반 threshold 자동 선택**을 포함하고 있어, 실제 촬영 품질 점검이나 재촬영 여부 판단 문제에 바로 적용할 수 있도록 구성되어 있습니다.

---

## 주요 기능

- 정상 이미지와 오류 이미지를 이용한 **이진 분류 학습**
- **Stratified 5-Fold Cross Validation** 지원
- 사전학습된 CNN backbone 기반 **Transfer Learning**
- 2단계 학습
  - Stage 1: backbone freeze 후 classifier head 학습
  - Stage 2: backbone unfreeze 후 fine-tuning
- **Early Stopping** 적용
- 검증 세트에서 목표 ERROR recall을 만족하도록 **threshold 자동 탐색**
- ERROR 클래스 기준 **Precision / Recall / F1** 출력
- fold별 best model 저장

---

## 지원 모델

다음 torchvision 사전학습 모델을 사용할 수 있습니다.

- `effb0` : EfficientNet-B0
- `mobilenetv3` : MobileNetV3-Large
- `resnet18` : ResNet18

기본값은 `effb0`입니다.

---

## 폴더 구조

코드는 아래와 같은 폴더 구조를 가정합니다.

```text
NARAE_TREND/
├─ image/
│  ├─ true_image/     # 정상 이미지 (OK, label=0)
│  └─ false_image/    # 오류 이미지 (ERROR, label=1)
├─ train_quality_cv.py
└─ outputs/
```

- `true_image` → 정상 클래스 (`label=0`)
- `false_image` → 오류 클래스 (`label=1`)

---

## 설치 환경

### 1. Python 패키지 설치

```bash
pip install torch torchvision scikit-learn pillow numpy
```

GPU 환경이 있다면 CUDA가 포함된 PyTorch 설치를 권장합니다.

---

## 실행 방법

기본 실행 예시는 아래와 같습니다.

```bash
python train_quality_cv.py --data_dir image --model effb0
```

예를 들어 ResNet18을 사용하려면 다음과 같이 실행할 수 있습니다.

```bash
python train_quality_cv.py --data_dir image --model resnet18
```

---

## 주요 인자

| 인자 | 설명 | 기본값 |
|---|---|---:|
| `--data_dir` | `true_image`, `false_image`를 포함하는 데이터 루트 폴더 | `image` |
| `--model` | 사용할 backbone (`effb0`, `mobilenetv3`, `resnet18`) | `effb0` |
| `--batch_size` | 배치 크기 | `16` |
| `--img_size` | 입력 이미지 크기 | `224` |
| `--k_folds` | 교차검증 fold 수 | `5` |
| `--seed` | 랜덤 시드 | `42` |
| `--target_recall_error` | ERROR 클래스 목표 recall | `0.95` |
| `--save_dir` | 모델 저장 폴더 | `outputs` |
| `--num_workers` | DataLoader worker 수 | `2` |

---

## 학습 방식

### 1. 데이터 구성

전체 데이터셋은 `true_image` 와 `false_image` 폴더에서 이미지를 읽어 구성됩니다.  
확장자는 `jpg`, `jpeg`, `png`, `bmp`, `tif`, `tiff`, `webp`를 지원하며, 대문자 확장자도 함께 처리합니다.

손상된 이미지가 있을 경우 바로 학습이 중단되지 않도록 예외 처리가 들어가 있으며, 문제가 있는 샘플은 다른 샘플로 대체 시도합니다.

### 2. Cross Validation

- `StratifiedKFold`를 사용하여 클래스 비율을 유지한 채 `k`개 fold로 분할합니다.
- 각 fold에서 train / val / test를 구성합니다.
- train 데이터 내부에서 다시 일부를 validation 세트로 분리합니다.

### 3. 2단계 학습

#### Stage 1
- backbone의 가중치를 고정합니다.
- classifier head만 학습합니다.
- 상대적으로 큰 learning rate를 사용합니다.

#### Stage 2
- backbone 전체를 unfreeze 합니다.
- 더 작은 learning rate로 fine-tuning을 수행합니다.

### 4. Loss 함수

- `BCEWithLogitsLoss`를 사용합니다.
- 클래스 불균형을 보정하기 위해 `pos_weight`를 자동 계산하여 적용합니다.
- 여기서 positive class는 `ERROR(false_image)` 입니다.

### 5. Threshold 자동 선택

검증 세트의 예측 확률을 바탕으로 threshold를 자동 탐색합니다.

선택 기준은 다음과 같습니다.

1. `target_recall_error` 이상을 만족하는 threshold들 중에서
   - precision이 가장 높은 threshold 선택
   - precision이 같으면 F1이 더 높은 threshold 선택
2. 목표 recall을 만족하는 threshold가 없으면
   - recall이 가장 높은 threshold 선택
   - 이후 precision, F1 순으로 비교

즉, 이 코드는 단순히 `0.5` threshold를 고정 사용하지 않고, **오류 이미지를 놓치지 않는 방향** 으로 threshold를 조정합니다.

---

## 데이터 증강 및 전처리

### Train Transform

- Resize
- RandomResizedCrop
- RandomHorizontalFlip
- ColorJitter
- Random GaussianBlur
- Normalize

### Validation / Test Transform

- Resize
- CenterCrop
- Normalize

학습 시에는 데이터 증강을 적용하고, 검증 및 테스트 시에는 안정적인 평가를 위해 고정 전처리만 적용합니다.

---

## 출력 결과

각 fold마다 다음 정보가 출력됩니다.

- train / val 샘플 수
- epoch별 train loss / val loss
- validation 기준 선택된 threshold
- test set ERROR 클래스 기준 Precision / Recall / F1
- TP / FP / FN
- 저장된 모델 경로

학습이 끝나면 전체 fold에 대한 평균과 표준편차를 요약하여 출력합니다.

예시 출력 항목:

- threshold mean ± std
- precision mean ± std
- recall mean ± std
- f1 mean ± std

---

## 저장 파일

각 fold의 best model은 아래 형식으로 저장됩니다.

```text
outputs/fold{fold}_{model_name}_thr{threshold}.pt
```

저장 내용은 다음을 포함합니다.

- `model_name`
- `state_dict`
- `threshold`
- `cfg`

즉, 추론 시에는 저장된 모델 가중치뿐 아니라, 해당 fold에서 선택된 threshold도 함께 사용할 수 있습니다.

---

## 평가지표

이 프로젝트는 **ERROR 클래스(오류 이미지)** 를 중심으로 성능을 평가합니다.

- Precision
- Recall
- F1-score
- TP / FP / FN

실제 활용 관점에서는 **오류 이미지를 놓치지 않는 것** 이 중요하므로, recall 제약을 둔 threshold 선택 방식이 핵심입니다.

---

## 활용 목적

이 코드는 다음과 같은 상황에 적합합니다.

- 촬영된 해충 이미지의 품질 자동 점검
- 재촬영 필요 여부 판단
- 데이터 수집 단계에서 품질이 낮은 이미지 자동 필터링
- 현장 이미지 수집 파이프라인의 품질 관리 자동화

---

## 주의사항

- 데이터가 매우 적거나 클래스 불균형이 심한 경우 fold별 성능 편차가 커질 수 있습니다.
- `target_recall_error`를 너무 높게 설정하면 recall은 올라가지만 precision이 낮아질 수 있습니다.
- 손상 이미지에 대한 예외 처리가 포함되어 있지만, 데이터셋 정제는 별도로 진행하는 것이 좋습니다.

---

## 요약

이 프로젝트는 해충 이미지의 품질을 자동으로 분류하기 위한 **이진 이미지 분류 학습 파이프라인**입니다.  
사전학습 CNN backbone을 활용한 transfer learning, stratified 5-fold cross validation, early stopping, threshold 자동 선택을 결합하여, 오류 이미지 검출 성능을 실무적으로 평가할 수 있도록 설계되어 있습니다.
