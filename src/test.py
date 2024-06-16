from llm_agents import *
from parser import *
import time
llm_config = {
    # change these three together
    'llm_type': 'openai',  # openai, ollama, anthropic
    'api_key_link': 'api_key.txt',
    'model': "gpt-4",  # see llm_model.txt
    'temperature': 0,
}
with open(llm_config['api_key_link'], 'r') as f:
    api_key = f.read()

now = time.time()
cot_agent = LLM_agent(llm_type=llm_config['llm_type'], api_key=api_key, model=llm_config['model'],
                                      temperature=llm_config['temperature'])
cot_agent.set_prompt('prompt_templates/cot_agent.json')
cot_agent.set_parser(CoT_Agent)
arguments_dict_cot = {
    'subject': 'college_math',
    'question': 'Let y = f(x) be a solution of the differential equation x dy + (y - xe^x) dx = 0 such that y = 0 when x = 1. What is the value of f(2)? The options are: A) 1/(2e), B) 1/e, C) e^2/2, D) 2e'
}
result = cot_agent.involk(arguments_dict_cot)

after = time.time()
print(result)
print(after-now)