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