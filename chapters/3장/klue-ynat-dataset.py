# 데이터셋 활용하기
# !pip install transformers==4.40.1 datasets==2.19.0 huggingface_hub==0.23.0 -qqq
# 인스톨 후 사용할 것

from datasets import load_dataset

klue_tc_train = load_dataset('klue', 'ynat', split='train')
klue_tc_eval = load_dataset('klue', 'ynat', split='validation')
klue_tc_train
# 출력
# Dataset({
#     features: ['guid', 'title', 'label', 'url', 'date'],
#     num_rows: 45678
# })

klue_tc_train[0]
# 출력
# {'guid': 'ynat-v1_train_00000',
#  'title': '유튜브 내달 2일까지 크리에이터 지원 공간 운영',
#  'label': 3,
#  'url': 'https://news.naver.com/main/read.nhn?mode=LS2D&mid=shm&sid1=105&sid2=227&oid=001&aid=0008508947',
#  'date': '2016.06.30. 오전 10:36'}

# label 확인
klue_tc_train.features['label'].names
# 출력
# ['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치']

# 분류모델 학습 시 guid, url, date 컬럼은 필요없으니 제거
klue_tc_train = klue_tc_train.remove_columns(['guid', 'url', 'date'])
klue_tc_eval = klue_tc_eval.remove_columns(['guid', 'url', 'date'])
klue_tc_train
# 출력
# Dataset({
#     features: ['title', 'label'],
#     num_rows: 45678
# })

# 카테고리 받아오기
klue_tc_train.features['label'].int2str(1)
# 출력
# '경제'

klue_tc_label = klue_tc_train.features['label']

def make_str_label(batch):
  batch['label_str'] = klue_tc_label.int2str(batch['label'])
  return batch

klue_tc_train = klue_tc_train.map(make_str_label, batched=True, batch_size=1000)
klue_tc_train[0]
# 출력
# {'title': '유튜브 내달 2일까지 크리에이터 지원 공간 운영', 'label': 3, 'label_str': '생활문화'}

# 10000개만 추출해 사용
train_dataset = klue_tc_train.train_test_split(test_size=10000, shuffle=True, seed=42)['test']

dataset = klue_tc_eval.train_test_split(test_size=1000, shuffle=True, seed=42)
# 학습이 잘 되고 있는지 확인할 검증 데이터
test_dataset = dataset['test']
# 성능 확인에 사용할 테스트 데이터
valid_dataset = dataset['train'].train_test_split(test_size=1000, shuffle=True, seed=42)['test']