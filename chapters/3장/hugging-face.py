# hugging face 라이브러리 설치
# !pip install transformers==4.40.1 datasets==2.19.0 huggingface_hub==0.23.0 -qqq
# - transformers : 트랜스포머 모델, 토크나이저 활용
# - datasets : 데이터셋 지원

from transformers import AutoModel, AutoTokenizer

text = "What is Huggingface Transformers?"
# BERT 모델 활용
bert_model = AutoModel.from_pretrained("bert-base-uncased") # 모델 불러오기
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # 토크나이저 불러오기
encoded_input = bert_tokenizer(text, return_tensors='pt') # 입력 토큰화
bert_output = bert_model(**encoded_input) # 모델에 입력

# GPT2 모델 활용
gpt_model = AutoModel.from_pretrained("gpt2") # 모델 불러오기
gpt_tokenizer = AutoTokenizer.from_pretrained('gpt2') # 토크나이저 불러오기
encoded_input = gpt_tokenizer(text, return_tensors='pt') # 입력 토큰화
gpt_output = gpt_model(**encoded_input) # 모델에 입력

# dataset 예제
# https://huggingface.co/datasets/klue/klue

# 모델 아이디로 모델 불러오기
from transformers import Automodel # 모델의 바디를 불러오는 클래스
model_id = 'Klue/roberta-base'
model = AutoModel.from_pretrained(model_id) # RoBERTa 모델을 한국어로 학습한 모델

# 분류 헤드가 붙은 모델
from transformers import AutoModelForSequenceClassification # 텍스트 시퀀스 분류를 위한 헤드가 포함된 모델 불러오는 클래스
model_id = 'SamLowe/roberta-base-go_emotions'
classification_model = AutoModelForSequenceClassification.from_pretrained(model_id)

# 텍스트 분류를 위한 아키텍처에 모델 바디만 불러오기
from transformers import AutoModelForSequenceClassification
model_id = 'klue/roberta-base'
classification_model = AutoModelForSequenceClassification.from_pretrained(model_id)

# 토크나이저
# - 텍스트를 토큰 단위로 나누고 각 토큰을 대응하는 토큰 아이디로 변환

from transformers import AutoTokenizer
model_id = 'klue/roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenized = tokenizer("공부 그만하고 싶은데")
print(tokenized)
# {'input_ids': [0, 4244, 4416, 19521, 1335, 2073, 2147, 2], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}

print(tokenizer.convert_ids_to_tokens(tokenized['input_ids']))
# ['[CLS]', '공부', '그만', '##하고', '싶', '##은', '##데', '[SEP]']

print(tokenizer.decode(tokenized['input_ids']))
# [CLS] 공부 그만하고 싶은데 [SEP]

print(tokenizer.decode(tokenized['input_ids'], skip_special_tokens=True))
# 공부 그만하고 싶은데

