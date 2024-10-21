# 성능 평가 파이프라인

import os
import json
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# SQL 생성 프롬프트
def make_prompt(ddl, question, query=''):
    prompt = f"""당신은 SQL을 생성하는 SQL 봇입니다. DDL의 테이블을 활용한 Question을 해결할 수 있는 SQL 쿼리를 생성하세요.
        ### DDL:
        {ddl}

        ### Question:
        {question}

        ### SQL:
        {query}"""
    return prompt

# 요청 jsonl 작성 함수
def make_requests_for_gpt_evaluation(df, filename, dir="requests"):
    if not os.path(dir).exists():
        os.path(dir).mkdir(parents=True)
    prompts = []
    for idx, row in df.iterrows():
        prompts.append("""Based on below DDL and Question, evaluate gen_sql can resolve Question. If gen_sql and gt_sql do equal job, return "yes" else return "no". Output JSON Format: {"resolve_yn": ""}""" + f"""
                        DDL : {row['context']}
                        Question: {row['question']}
                        gt_sql: {row['answer']}
                        gen_sql: {row['get_sql']}
                        """)
        jobs = [{"model" : "gpt-4-turbo-preview", "response_format" : {"type": "json_object"}, "messages": [{"role" : "system", "content": prompt}]} for prompt in prompts]
        with open(os.path(dir, filename), "w") as f :
            for job in jobs:
                json_string = json.dumps(job)
                f.write(json_string + "\n")

# 비동기 요청 명령
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# python api_request_parallel_processor.py \
#   --requests_filepath {요청 파일 경로} \
#   --save_filepath {생성할 결과 파일 경로} \
#   --request_url https://api.openai.com/v1/chat/completions \
#   --max_requests_per_minute 300 \
#   --max_tokens_per_minute 100000 \
#   --token_encoding_name cl100k_base \
#   --max_attempts 5 \
#   --logging_level 20

# 명령어 실행 후, 반환된 평가 결과를 읽어와 csv로 변환하는 함수
def change_jsonl_to_csv(input_file, output_file, prompt_column="prompt", response_column="response"):
    prompts = []
    responses = []
    with open(input_file, 'r') as json_file:
        for data in json_file:
            prompts.append(json.loads(data)[0]['messages'][0]['content'])
            responses.append(json.loads(data)[1]['choices'][0]['message']['content'])
    df = pd.DataFrame({prompt_column: prompts, response_column: responses})
    df.to_csv(output_file, index=False)
    return df