# kor-sentence-ordering

## 개요

데이콘의 [문맥 기반 문장 순서 예측 AI 경진대회](https://dacon.io/competitions/official/236489/overview/description)에서 3위를 달성한 솔루션을 소개하는 저장소입니다. 코드 공유 및 솔루션 소개 PPT는 [여기](later...)에서 확인할 수 있습니다. 

## 환경

훈련 및 추론은 모두 Colab의 L4 환경에서 이루어졌습니다. 단, GPU를 이용하지 않는 노트북은 CPU를 이용했습니다.

## 모델

[Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) 모델을 사용했습니다. Unsloth 버전은 [여기](https://huggingface.co/unsloth/Qwen3-14B)에서 확인 가능합니다.

## 접근 방식

접근 방식을 간단하게 표현하면 다음과 같습니다.

1. 원본 데이터를 정렬한 후, 증강합니다.
2. 증강한 데이터를 포함하여 무작위로 섞인 훈련 데이터를 만듭니다.
3. QLoRA, Unsloth, SFTrainer 등을 이용하여 Qwen3-14B를 훈련합니다.
4. do_sample=False로 설정하여 테스트 데이터를 추론합니다. 29개의 체크 포인트에 대해 반복합니다.
5. 가장 성능이 높은 세 개의 체크 포인트 모델의 예측을 Ensemble하여 최종 예측을 만듭니다.

## 코드

경진 대회에 사용한 모든 코드는 [notebooks](/notebooks) 폴더에서 확인할 수 있습니다. 

### 주의 사항

1. 데이터 증강이나 훈련 자체는 재현이 완벽하게 되지 않을 수 있습니다. 왜냐하면 Colab의 시간 제한 때문에 증강이나 훈련을 한 런타임에서 계속 하지 못했기 때문입니다. 하지만 코드 자체는 동일하므로 같은 접근 방식을 따른다면 결과는 크게 다르지 않을 것입니다.
2. dataset 폴더에 훈련, 테스트 데이터는 규정에 의해 포함되어 있지 않습니다. 데이터는 

## 기타 자료

가장 성능이 높았던 단일 모델은 [여기](https://huggingface.co/JuyeopDang/Qwen-3-14B-Sentence-Ordering)에서 확인할 수 있습니다. 사용 방법은 notebooks 폴더의 [5-final-evaluation(L4).ipynb](notebooks/5-final-evaluation(L4).ipynb)를 참고해 주시기 바랍니다.
