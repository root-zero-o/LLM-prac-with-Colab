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

