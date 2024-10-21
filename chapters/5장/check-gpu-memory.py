import torch

# 메모리 사용량 측정을 위한 함수
def print_gpu_utilization():
  if torch.cuda.is_available():
    used_memory = torch.cuda.memory_allocated() / 1024 ** 3
    print(f"GPU 메모리 사용량 : {used_memory:.3f}GB")
  else :
    print("런타임 유형을 GPU로 변경하세요")

# 모델을 불러오고 GPU 메모리와 데이터 타입 확인
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_id, peft=None):
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  if peft is None:
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map={"":0})

  print_gpu_utilization()
  return model, tokenizer

model_id = "EleutherAI/polyglot-ko-1.3b"
model, tokenizer = load_model_and_tokenizer(model_id) # GPU 메모리 사용량 : 2.599GB
print("모델 파라미터 데이터 타입: ", model.dtype) # 모델 파라미터 데이터 타입:  torch.float16

