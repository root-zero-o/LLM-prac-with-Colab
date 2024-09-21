# Pytorch 설치
# 런타임 - 런타임 유형 변경 - GPU로 변경
#
# !pip3 install torch
# !pip3 install torchvision

# 트랜스포머 아키텍쳐 - 텍스트를 숫자형 데이터로 변환

import torch
import torch.nn as nn

# 1) 토큰화
# 텍스트를 적절한 단위로 나누고 숫자 아이디를 부여한다. 

# 띄어쓰기 단위로 분리
input_text ="나는 최근 파리 여행을 다녀왔다"
input_text_list = input_text.split()

# 토큰 -> 아이디 딕셔너리와 아이디 -> 토큰 딕셔너리 만들기
str2idx = {word:idx for idx, word in enumerate(input_text_list)}
idx2str = {idx:word for idx, word in enumerate(input_text_list)}

# 토큰을 토큰 아이디로 변환
input_ids = [str2idx[word] for word in input_text_list]
print(input_ids)

# 2) 토큰 임베딩으로 변환
# 토큰의 의미를 담기 위해 토큰을 최소 2개 이상의 숫자 집합인 벡터로 변환한다.
# 의미를 담기 위해서는 딥러닝 모델이 훈련되어야 함

embedding_dim = 16
embed_layer = nn.Embedding(len(str2idx), embedding_dim)

input_embeddings = embed_layer(torch.tensor(input_ids))
input_embeddings = input_embeddings.unsqueeze(0)
input_embeddings.shape # 출력 : torch.Size([1, 5, 16])

# 3) 위치 인코딩
# 절대적 위치 인코딩 - 입력 토큰의 위치에 따라 고정된 임베딩을 더해준다.
# 토큰 임베딩 + 위치 인코딩 = 모델에 입력할 최종 입력 임베딩
embedding_dim = 16
max_position = 12
embed_layer = nn.Embedding(len(str2idx), embedding_dim)
position_embed_layer = nn.Embedding(max_position, embedding_dim)

position_ids = torch.arange(len(input_ids), dtype=torch.long).unsqueeze(0)
position_encodings = position_embed_layer(position_ids)
token_embeddings = embed_layer(torch.tensor(input_ids))
token_embeddings = token_embeddings.unsqueeze(0)
input_embeddings = token_embeddings + position_encodings
input_embeddings.shape

# 4) 어텐션 연산 구현
# 토큰 임베딩에 쿼리, 키, 값 가중치를 도입 -> 토큰과 토큰 사이의 관계를 계산해 맥락을 반영한다

head_dim = 16

# 쿼리, 키, 값에 대한 가중치
weight_q = nn.Linear(embedding_dim, head_dim)
weight_k = nn.Linear(embedding_dim, head_dim)
weight_v = nn.Linear(embedding_dim, head_dim)

# 변환 수행
querys = weight_q(input_embeddings)
keys = weight_k(input_embeddings)
values = weight_v(input_embeddings)

# 스케일 점곱 방식의 어텐션
from math import sqrt
import torch.nn.functional as F

def compute_attention(querys, keys, values, is_causal=False):
  dim_k = querys.size(-1)
  scores = querys @ keys.transpose(-2, -1) / sqrt(dim_k)
  weights = F.softmax(scores, dim=-1)
  return weights @ values

# 어텐션 연산 이후 : 주변 토큰과의 관련도에 따라 값 벡터를 조합한 새로운 토큰 임베딩이 생성된다.
print("원본 입력 형태 : ", input_embeddings.shape)

after_attention_embeddings = compute_attention(querys, keys, values)

print("어텐션 적용 후 형태 : ", after_attention_embeddings.shape)

# 출력
# 원본 입력 형태 :  torch.Size([1, 5, 16])
# 어텐션 적용 후 형태 :  torch.Size([1, 5, 16])

# AttentionHead 클래스 - 어텐션 연산을 수행한다(위에서 한 어텐션 연산을 수행하는 클래스)
class AttentionHead(nn.Module):
  def __init__(self, token_embed_dim, head_dim, is_causal=False):
    super().__init__()
    self.is_causal = is_causal
    self.weight_q = nn.Linear(token_embed_dim, head_dim) # 쿼리 벡터 생성을 위한 선형 층
    self.weight_k = nn.Linear(token_embed_dim, head_dim) # 키 벡터 생성을 위한 선형 층
    self.weight_v = nn.Linear(token_embed_dim, head_dim) # 값 벡터 생성을 위한 선형 층
  
  def forward(self, querys, keys, values):
    outputs = compute_attention(
        self.weight_q(querys), # 쿼리 벡터
        self.weight_k(keys), # 키 벡터
        self.weight_v(values), # 값 벡터
        is_causal=self.is_causal
    )
    return outputs

attention_head = AttentionHead(embedding_dim, embedding_dim)
after_attention_embeddings = attention_head(input_embeddings, input_embeddings, input_embeddings)

# 멀티 헤드 어텐션 - 한 번에 여러 어텐션 연산을 동시에 적용하여 성능을 높인다
class MultiheadAttention(nn.Module):
  def __init__(self, token_embed_dim, d_model, n_head, is_causal=False):
    super().__init__()
    self.n_head = n_head
    self.is_causal = is_causal
    self.weight_q = nn.Linear(token_embed_dim, d_model)
    self.weight_k = nn.Linear(token_embed_dim, d_model)
    self.weight_v = nn.Linear(token_embed_dim, d_model)
    self.concat_linear = nn.Linear(d_model, d_model)
  
  def forward(self, querys, keys, values):
    B, T, C = querys.size()
    querys = self.weight_q(querys).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # n_head로 쪼갠다
    keys = self.weight_k(keys).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    values = self.weight_v(values).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    attention = compute_attention(querys, keys, values, self.is_causal) # 각각의 어텐션을 계산한다

    output = attention.transpose(1, 2).contiguous().view(B, T, C) # 입력과 같은 형태로 변환한다
    output = self.concat_linear(output) # 선형층을 통과시키고 최종 결과를 반환한다
    return output
  
n_head = 4
mh_attention = MultiheadAttention(embedding_dim, embedding_dim, n_head)
after_attention_embeddings = mh_attention(input_embeddings, input_embeddings, input_embeddings)
after_attention_embeddings.shape