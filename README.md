# Technical_portfolio_vision
서울과학기술대학교_데이터사이언스학과_24장수진_졸업심사_테크니컬포트폴리오
## 해충 이미지 품질 분류 프로젝트

해충 이미지의 품질을 OK / ERROR로 분류하는 이진 분류 학습 스크립트로 재촬영이 필요한 이미지(ERROR)를 detect하는 것을 목표로 함

### 주요 특징

- Stratified 5-Fold Cross Validation
- Transfer Learning (EfficientNet-B0 / MobileNetV3-Large / ResNet18)
- 2단계 학습: 백본 고정 후 헤드 학습 → 전체 파인튜닝
- Early Stopping
- Validation 기준 ERROR Recall 목표치를 만족하는 임계값 자동 탐색

### 폴더 구조

```
NARAE_TREND/
  image/
    true_image/    ← 정상 이미지 (OK)
    false_image/   ← 오류 이미지 (ERROR)
  train_quality_cv.py
  outputs/
```

### 설치

torch, torchvision, scikit-learn, Pillow 설치

### 실행 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--data_dir` | `image` | 이미지 루트 폴더 |
| `--model` | `effb0` | effb0 / mobilenetv3 / resnet18 |
| `--batch_size` | 16 | 배치 크기 |
| `--img_size` | 224 | 입력 이미지 크기 |
| `--k_folds` | 5 | CV 폴드 수 |
| `--target_recall_error` | 0.95 | ERROR 클래스 목표 재현율 |
| `--save_dir` | `outputs` | 모델 저장 경로 |

### 결과

- fold별로 최적 모델(`.pt`)이 `outputs/` 폴더에 저장되며, 각 폴드의 Precision / Recall / F1과 선택된 임계값이 출력됨
- 학습 종료 후 전체 fold의 평균 ± 표준편차가 요약됨
