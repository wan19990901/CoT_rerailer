import os
import pandas as pd
import re
from collections import Counter
from llm_agents import *
from tqdm import tqdm
import argparse
PREPROCESSED_FP = '../data/preprocessed'

llm_config = {
    # change these three together
    'llm_type': 'openai',  # openai, ollama, anthropic
    'api_key_link': 'api_key.txt',
    'model': "gpt-4o",  # see llm_model.txt
    'temperature': 0,
}

with open(llm_config['api_key_link'], 'r') as f:
    api_key = f.read()

def load_df(dataset_fp):
    df = pd.read_csv(os.path.join(PREPROCESSED_FP, dataset_fp))
    print('------------------------------------------------------')
    print(f'The distribution of category in {dataset_fp} is:\n{Counter(df.Category)}')
    return df





def standardize_answer(answer):
    # Check for strict multiple choice format (single letter or letter followed by parenthesis)
    if re.match(r'^[a-zA-Z]\W.*$', answer.strip()):
        return answer.strip().lower()[0]

    # For other cases, return the answer as is
    return answer.lower()


def mad(df, num_steps= 'MULTI'):
    result_df_dict = {
        'CaseID': [],
        'Category': [],
        'Question': [],
        'Correct Answer': [],
        'ori_cot1': [],
        'ori_cot2': [],
        'ori_ans1': [],
        'ori_ans2': [],
        'cot1': [],
        'cot2': [],
        'ans1': [],
        'ans2': [],
        'MAD_answer': []
    }
    print(f'There are {len(df)} data in total')
    print(f'Category distribution is {Counter(df.Category.tolist())}')
    debate_rounds = 3
    # Add a variable to track whether the header has been written
    header_written = False
    for row_idx in tqdm(range(len(df))):
        row = df.iloc[row_idx]
        subject = row['Category']
        question = row['Question']
        correct_answer = row['Correct_Answer']
        cot1 = row['cot1']
        cot2 = row['cot2']
        answer1 = row['answer1']
        answer2 = row['answer2']

        result_df_dict['CaseID'].append(row_idx)
        result_df_dict['Category'].append(subject)
        result_df_dict['Question'].append(question)
        result_df_dict['Correct Answer'].append(standardize_answer(correct_answer))
        result_df_dict['ori_cot1'].append(cot1)
        result_df_dict['ori_cot2'].append(cot2)
        print('\n\n\n')
        print('question: ', question)
        print('correct answer: ', correct_answer)
        print('COT1: ', cot1)
        print('/n/n')
        print('COT2',cot2)
        print('ans1: ', answer1, '; ans2: ',answer2)
        print('\n\n\n')
        result_df_dict['ori_ans1'].append(standardize_answer(answer1))
        result_df_dict['ori_ans2'].append(standardize_answer(answer2))

        for i in range(debate_rounds-1):
            has_mistake = True
            counter = 0

            agent1 = LLM_agent(llm_type=llm_config['llm_type'], api_key=api_key, model=llm_config['model'],
                               temperature=llm_config['temperature'])
            agent1.set_prompt('prompt_templates/multi_agent_debate.json')
            agent1.set_parser(Multi_Agent_Debate)
            arguments = {
                'subject': subject,
                'response': cot2,
                'question': question

            }



            while (has_mistake is True) and counter < 5:
                try:
                    agent1_response = agent1.involk(arguments)
                    has_mistake = False
                except:
                    has_mistake = True
                counter += 1


            has_mistake = True
            counter = 0
            agent2 = LLM_agent(llm_type=llm_config['llm_type'], api_key=api_key, model=llm_config['model'],
                               temperature=llm_config['temperature'])
            agent2.set_prompt('prompt_templates/multi_agent_debate.json')
            agent2.set_parser(Multi_Agent_Debate)
            arguments2 = {
                'subject': subject,
                'response': cot1,
                'question': question

            }
            while (has_mistake is True) and counter < 5:
                try:
                    agent2_response = agent2.involk(arguments2)
                    has_mistake = False
                except:
                    has_mistake = True
                counter += 1
            print('---------------------')
            print(agent1_response)
            print('---------------------')
            print(agent2_response)
            try:
                cot1 = agent1_response['Updated_Response']
                ans1 = agent1_response['Final_Answer']
                cot2 = agent2_response['Updated_Response']
                ans2 = agent2_response['Final_Answer']
            except:
                cot1 = 'error'
                ans1 = 'error'
                cot2 = 'error'
                ans2 = 'error'


        result_df_dict['ans1'].append(ans1)
        result_df_dict['ans2'].append(ans2)
        result_df_dict['cot1'].append(cot1)
        result_df_dict['cot2'].append(cot2)
        result_df_dict['MAD_answer'].append(ans1)

        if len(result_df_dict['CaseID']) >= 1:
            result_df = pd.DataFrame.from_dict(result_df_dict)
            result_df.to_csv(f'../result/MAD_result_{num_steps}.csv', mode='a', header=not header_written, index=False)
            header_written = True  # Ensure header is not written again
            # Clear the buffer
            for key in result_df_dict.keys():
                result_df_dict[key].clear()

    if len(result_df_dict['CaseID']) > 0:
        result_df = pd.DataFrame.from_dict(result_df_dict)
        result_df.to_csv(f'../result/rerailer_result_{num_steps}.csv', mode='a', header=not header_written, index=False)



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--STEPS', type=str, required=True)
    # args = parser.parse_args()

    df_raw = pd.read_csv('../data/final_test_data/added_experiments/cleaned_result_gpt3.5.csv')
    df = df_raw.loc[df_raw.Consistency == False].iloc[55:]
    mad(df, num_steps='MULTI')

