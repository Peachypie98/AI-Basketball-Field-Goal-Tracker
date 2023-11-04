# 영상 기반 농구 점수 자동 기록 AI 서비스✨
![](assets/final_inference.gif) 

*Read this in other languages: [English](README.md), [한국어](README_kor.md)*

## 🏀 배경
&nbsp;농구 경기에서 선수가 얼마나 골을 잘 넣을까 생각해 보신 적이 있나요? 만약 어떤 선수의 야투율을 수작업으로 계산한다고 하면 많은 시간과 노력이 필요할 것입니다. 게다가 바쁜 현대인들은 본인 혹은 선수들에 대한 모든 경기 영상을 시청하고 분석할 시간이 없습니다. 이처럼 농구 경기에 대해 분석하고 싶지만, 시간이 없어 망설이는 사람들을 위해 저희는 딥러닝을 사용한 영상 기반 농구 점수 자동 기록 AI 서비스를 개발했습니다! 😊

<br>

## 🧠 모델
 &nbsp;우리는 목표를 달성하기 위해 두 가지의 모델을 사용했습니다. 첫 번째 모델은 선수, 공, 골대, 슛, 골을 감지하는 데 사용하였습니다. 두 번째 모델은 프레임별로 개인의 id를 추적하는 데 사용하였습니다. 즉, 우리는 야투율 추적기를 구현하기 위해 Object Detection과 Person Re-Identification 모델을 사용했습니다. Object Detection 모델은 Deci-AI에서 개발한 super-gradients의 **YOLO-NAS-L**를 사용했습니다. 또한 Person Re-Identification 모델은 더 빠른 추론 속도와 더 작은 모델 크기를 보장하기 위해 **MobileNetV3**를 사용했습니다.   

<br>

## Ⓜ️ Faiss
 &nbsp;메타가 만든 Faiss는 여러 벡터 표현 간의 유사성을 빠르게 검색할 수 있도록 해주는 라이브러리입니다. 이를 통해 어떤 선수가 슛을 쏘고 골을 넣었는지 파악할 수 있기 때문에 해당 프로젝트에 꼭 필요한 도구입니다. 처음에는 유사성을 측정하기 위해 L2 (유클리드) 거리를 사용하여 테스트하였고 좋은 결과를 얻었습니다. 하지만 추가 실험을 거쳐서 Cosine Similarity를 활용한 것이 더 나은 결과를 도출한다는 것을 발견했습니다. 따라서 우리는 최종적인 Cosine Similarity를 검색 방법으로 채택하기로 결정하였습니다

![](assets/faiss.jpg) 

<br>

## 🖼️ Object Detection + Person Re-Identification 추론 다이어그램
 &nbsp;아래는 우리 프로젝트의 흐름을 나타내는 다이어그램입니다. 입력 프레임이 주어지면, Detection 모델을 통해 선수, 공, 골대, 슛, 골 클래스를 감지합니다. 이 중 선수 클래스의 인스턴스를 추출하여 Re-ID 모델에 입력으로 제공합니다. 이어서 Re-ID 모델은 개개인의 이미지를 나타내는 임베딩 벡터를 생성합니다. 해당 벡터들은 Faiss에 추가되어, 각 임베딩 벡터에 해당하는 상위 5개의 ID를 얻을 수 있게 됩니다. 이렇게 얻어진 결과들에 대해 최종적으로 가장 높은 신뢰도를 가진 ID를 확정하기 위해 hard voting을 활용합니다.

![](assets/inference_diagram.jpg) 

<br>

## 📝 학습 실험 결과
### Object Detection 모델
| Models | Dataset[^1] | Input Dimensions | Epochs | Batch Size (Accumulate) | Optimizer | LR | Loss | Augmentations | F1<sup>val<br>0.5 | mAP<sup>val<br>0.5 | 
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| YOLO NAS-L | D1 | (1920,1088) | 50 | 8 <br> (64)  | AdamW | 0.00001 | PPYOLOE | Resize <br>Normalize <br>HorizontalFlip | 0.2811 | 0.6485 |
| **YOLO NAS-L** | **D2** | **(1920,1088)** | **215** | **8 <br> (64)** | **AdamW** | **0.0001** | **PPYOLOE** | **HSV <br> Mosaic <br> RandomAffine <br> HorizontalFlip <br> PaddedRescale <br> Standardize <br>** | **0.8709** | **0.9407** |
[^1]: D1, D2 Dataset은 저희의 Custom Dataset입니다. D2은 D1보다 약 3.1배 많은 데이터를 가지고 있습니다.

<br>

### Person Re-Identification 모델
| Models | Dataset[^2] | Embedded Dimensions | Epochs | Batch Size | Optimizer | LR | Loss | Augmentations | mAP<sup>val |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| MobileNetV3 | R1 | 1000 | 100 | 64 | AdamW | 0.001 | TripletLoss | Resize <br>Normalize <br>HorizontalFlip | 0.9829 |
| MobileVitV2 | R1 | 1000 | 100 | 64 | AdamW | 0.001 | TripletLoss | Resize <br>Normalize <br>HorizontalFlip | 0.9748 |
| ConvNextV2-A | R1 | 1000 | 100 | 64 | AdamW | 0.001 | TripletLoss | Resize <br>Normalize <br>HorizontalFlip | 0.9721 |
| SqueezeNet | R1 | 1000 | 100 | 64 | AdamW | 0.001 | TripletLoss | Resize <br>Normalize <br>HorizontalFlip | 0.9758 |
| MobileNetV3 | R2 | 1000 | 100 | 64 | AdamW | 0.001 | TripletLoss | Resize <br>Normalize <br>HorizontalFlip | 0.8743 |
| MobileNetV3 | R2 | 1000 | 100 | 64 | AdamW | 0.001 | QuadrupletLoss | Resize <br>Normalize <br>HorizontalFlip | 0.9782 |
| SqueezeNetMod[^3] | R2 | 1000 | 500 | 64 | AdamW | 0.001 | QuadrupletLoss | Resize <br>Normalize <br>HorizontalFlip | 0.9857 |
| **MobileNetV3** | **R2** | **1000** | **500** | **64** | **AdamW** | **0.001** | **QuadrupletLoss** | **Resize <br>Normalize <br>HorizontalFlip** | **0.9923** |
[^2]: R1, R2 Dataset은 저희의 Custom Dataset입니다. R2가 R1보다 조금 더 많은 데이터와 id를 가지고 있습니다.
[^3]: 8배의 채널 압축을 한 SqueezeNet + CBAM

<br>

## 🛠️ 설치
```py
git clone https://github.com/boostcampaitech5/level3_cv_finalproject-cv-07.git
cd level3_cv_finalproject-cv-07
conda env create --name <env_name> -f env.yaml
```

<br>

## 🗂️ Dataset 경로 설정
> Object Detection 경로
 
Dataset을 다음과 같은 경로에 구성해 주세요:
1. data 파일명은 학습에 영향을 미치지 않습니다. 하지만 json 파일명은  `train.json`과 `valid.json`로 설정하여야 합니다. 
```
detection
├── data
│   ├── dataset
|   |    ├── train
|   |    |   ├── <sample1>.jpg
|   |    |   ├── <sample2>.jpg
|   |    |   ...
|   |    |   └── <sample10>.jpg
|   |    ├── valid
|   |    |   ├── <sample1>.jpg
|   |    |   ├── <sample2>.jpg
|   |    |   ...
|   |    |   └── <sample10>.jpg
|   |    ├── train.json
|   |    └── valid.json
│   ├── images
│   └── video
...
```
<br>

> Person Re-Identifcation 경로

Dataset을 다음과 같은 경로에 구성해 주세요:
1. 각 데이터 항목명을 다음과 같은 형식으로 구성해야 합니다 : `xxxxx_xx.jpg` or `xxxxx_xx_xx.jpg`
2. 해당 형식에서 처음 5개의 숫자는 사람의 ID 번호를 나타냅니다.
```
re_id
├── data
│   └── custom_dataset
|       ├── gallery
|       |   ├── <00001_01>.jpg
|       |   ├── <00001_02>.jpg
|       |   ├── <00002_01>.jpg
|       |   ├── <00002_02>.jpg
|       |   ...
|       |   └── <00010_2>.jpg
|       ├── query
|       |   ├── <00001_01>.jpg
|       |   ├── <00001_02>.jpg
|       |   ├── <00002_01>.jpg
|       |   ├── <00002_02>.jpg
|       |   ...
|       |   └── <00010_2>.jpg
|       └── training
|           ├── <00001_01>.jpg
|           ├── <00001_02>.jpg
|           ├── <00002_01>.jpg
|           ├── <00002_02>.jpg
|           ...
|           └── <00010_2>.jpg
... 
```

<br>

> Magic 경로

이곳에 추론할 비디오를 저장해 주세요.
```
datasets
├── <video1>.mp4
├── <video2>.mp4
... 
```

<br>

## 👨🏻‍💻 단 하나의 Command Line으로 학습 및 추론
### Detection Model 학습
---
```
cd detection/tools
python3 train.py --exp_name exp1 --input_dim (1920,1088) --epochs 100 --lr 0.0001 --batch_size 8 --optimizer AdamW --num_workers 4 --warmup_initial_lr 0.00001 --lr_warmup_epochs 5 --score_thr 0.8 --nms_thr 0.8 -- metric F1@0.50 --fp16 True
```
* --exp_name: 실험명
* --input_dim: input dimensions
* --epochs: epoch
* --lr: learning rate
* --batch_size: batch size
* --optimizer: optimizer
* --num_workers: dataloader num workers
* --warmup_initial_lr: warmup initial learning rate
* --lr_warmup_epochs: learning rate warmup epochs
* --score_thr: score threshold
* --nms_thr: non-max suppression threshold
* --metric: evaluation metric
* --fp16: mixed precision training

<br>

### Detection Model 추론
---
```
cd detection/tools
python3 inference.py --image True --video False --file_name image1.png --conf 0.25 --iou 0.35 --model_weight <your_exp_name>/<your_detection_weight>
```
`image` 랑 `video` 는 동시에 `True` 로 설정할 수 없습니다!
* --image: image 추론
* --video: video 추론
* --file_name: 추론할 image 또는 video 파일
* --conf: confidence threshold
* --iou: iou threshold
* --model_weight: model weight file

<br>

### Person Re-Identification Model 학습
---
```
cd re_id/tools
python3 train.py --demo False --seed 1 --model mobilenetv3 --epoch 100 --train_batch 64 --valid_batch 256 --lr 0.001 --num_workers 8 --quadruplet True --scheduler False --fp16 False
```
* --demo: `True` DeepSportsRadar dataset 사용 |  `False` Custom dataset 사용
* --seed: seed number
* --model: model. 사용 가능한 모델은 `model.py` 을 참고해 주세요.
* --epoch: epoch
* --train_batch: train batch size
* --valid_batch: valid batch size
* --lr: learning rate
* --num_workers: dataloader num workers
* --quadruplet:  `True` quadruplet loss 사용 | `False` triplet loss 사용
* --scheduler: lambda scheduler with 0.95**epoch
* --fp16: mixed precision training

<br>

### Person Re-Identification Model 추론
---
```
cd re_id/tools
python3 inference.py --demo False --model mobilenetv3 --model_weight <your_reid_weight> --batch_size 256 --num_workers 8 --query_index 0
```
* --demo: `True` DeepSportsRadar dataset 사용 | `False` Custom dataset 사용
* --model: model
* --model_weight: model weight file
* --batch size: test batch size
* --num worker: dataloader num worker
* --query_index: query index

<br>

### Magic 추론 (최종 결과 도출)
---
```
python3 magic.py --detection_weight <exp_name>/<your_detection_weight> --reid_weight <your_reid_weight> --video_file <your_video_file> --reid_model mobilenetv3 --person_thr 0.5 --cosine_thr 0.5
```
* --detection_weight: detection model trained weight
* --reid_weight: re-id model trained weight
* --video_file: 추론할 video 파일
* --reid_model: re-id model
* --person_thr: person confidence threshold
* --cosine_thr: cosine similarity threshold

<br>
  
## 멤버
| 고금강 | 김동우 | 박준일 | 임재규 | 최지욱 |
|:--:|:--:|:--:|:--:|:--:|
|<img  src='https://avatars.githubusercontent.com/u/101968683?v=4'  height=80  width=80px></img>|<img  src='https://avatars.githubusercontent.com/u/113488324?v=4'  height=80  width=80px></img>|<img  src='https://avatars.githubusercontent.com/u/106866130?v=4'  height=80  width=80px></img>|<img  src='https://avatars.githubusercontent.com/u/77265704?v=4'  height=80  width=80px></img>|<img  src='https://avatars.githubusercontent.com/u/78603611?v=4'  height=80  width=80px></img>|
|[Github](https://github.com/TwinKay)|[Github](https://github.com/dwkim8155)|[Github](https://github.com/Parkjoonil)|[Github](https://github.com/Peachypie98)|[Github](https://github.com/guk98)|
|twinkay@yonsei.ac.kr|dwkim8155@gmail.com|joonil2613@gmail.com|jaekyu.1998.bliz@gmail.com|guk9898@gmail.com|