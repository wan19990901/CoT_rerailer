import os
import pandas as pd
import re
from collections import Counter
from llm_agents import *
from tqdm import tqdm
from dotenv.main import load_dotenv
import random
import os
PREPROCESSED_FP = '../data/preprocessed'
llm_config = {
    # change these three together
    'llm_type': 'openai',  # openai, ollama, anthropic
    'api_key_link': 'api_key_claude.txt',
    'model': "gpt-4-0125-preview",  # see llm_model.txt
    'temperature': 0,
}



load_dotenv()
try:
    api_key = os.environ.get('OPENAI_GY_API_KEY')
except:
    raise "APIKEY Error, no API Key"

def load_df(dataset_fp):
    df = pd.read_csv(os.path.join(PREPROCESSED_FP, dataset_fp))
    print('------------------------------------------------------')
    print(f'The distribution of category in {dataset_fp} is:\n{Counter(df.Category)}')
    return df




def generate_new_response(subject, question,cot):
    success = False
    while not success:

        correct_answer_agent_partial_cot = LLM_agent(llm_type=llm_config['llm_type'], api_key=api_key, model=llm_config['model'],
                                  temperature=llm_config['temperature'])
        correct_answer_agent_partial_cot.set_prompt('llm_project/src/prompt_templates/correct_answer_agent_partial_cot.json')
        correct_answer_agent_partial_cot.set_parser(Correct_Answer_Agent_Partial_CoT)
        arguments = {
            'subject':subject,
            'question':question,
            'cot':cot
        }

        try:
            forward_result = correct_answer_agent_partial_cot.involk(arguments)
            success = True
        except:
            success = False
    print('------------------------------------------------------')
    for key, value in forward_result.items():
        print(key)
        print(value)
    print('------------------------------------------------------')
    cot, final_answer = forward_result.values()
    return cot, final_answer

def self_correct_complete(cot, steps, question,subject):
    check_list = []
    partial_cot = []
    for i in range(int(steps)):
        current_step = i + 1

        masked_cot = cot[:i+1]


        success = False
        while not success:

            root_checker_agent = LLM_agent(llm_type=llm_config['llm_type'], api_key=api_key, model=llm_config['model'],
                                  temperature=llm_config['temperature'])
            root_checker_agent.set_prompt('llm_project/src/prompt_templates/root_checker_agent.json')
            root_checker_agent.set_parser(Root_Checker_Agent)
            arguments = {
                'subject':subject,
                'current_step':current_step,
                'cot':masked_cot,
                'question':question

            }
            try:
                response = root_checker_agent.involk(arguments)
                success = True
            except:
                success = False

        print(f'Step {current_step}', response, '\n\n')
        check_list.append((response['Step_Hallucination']))
        if (response['Step_Hallucination'] == 'YES'):
            debate_response = multi_agents_debate(subject,current_step,masked_cot,question,response)
            print('Old Version: ', masked_cot[i])
            partial_cot.append(debate_response['Correction'])
            print('Corrected Version', partial_cot[i])

            break
        else:
            if not re.search(r'\*<verified>\*', masked_cot[i]):
                partial_cot.append(masked_cot[i] + ' *<verified>*')
            else:
                partial_cot.append(masked_cot[i])
    print('------------------------------------------------------')
    print(partial_cot)
    print('------------------------------------------------------')

    return check_list, partial_cot


def standardize_answer(answer):
    # Check for strict multiple choice format (single letter or letter followed by parenthesis)
    if re.match(r'^[a-zA-Z]\W.*$', answer.strip()):
        return answer.strip().lower()[0]

    # For other cases, return the answer as is
    return answer.lower()

def multi_agents_debate(subject,current_step,masked_cot,question,response):
    final_response = response

    # print('Start Debating')
    # attempts = 0
    # counter = 0
    # while (attempts < 1) and counter <=2:
    #     print('attempt:',attempts)
    #     success = False
    #     while not success:
    #         try:
    #             debate = debate_agent.json(subject=subject, current_step=current_step, cot=masked_cot,
    #                                                       question=question,response = response)
    #             response = output_repraser(debate)
    #             print('\n\n\n',response,'\n\n\n')
    #             if response['Agreement'] == 'YES':
    #                 final_response = response
    #                 attempts += 1
    #
    #             success = True
    #             counter += 1
    #         except:
    #             success = False

    return final_response





if __name__ == '__main__':

    result_df_dict = {
        'CaseID': [],
        'Category': [],
        'Question': [],
        'Correct Answer': [],
        'Raw COT Answer': [],
        'Corrected COT Answer': [],
        'Hallu Seq': [],
        'raw_cot': [],
        'corrected_cot': []
    }

    df_raw = pd.read_csv('llm_project/result/gpt4_results.csv')
    df = df_raw.loc[df_raw.Consistency == False]

    print(f'There are {len(df)} data in total')
    print(f'Category distribution is {Counter(df.Category.tolist())}')

    # Add a variable to track whether the header has been written
    header_written = False
    for row_idx in tqdm(range(119,len(df))):
        row = df.iloc[row_idx]
        subject = row['Category']
        question = row['Question']
        correct_answer = row['Correct_Answer']
        cot = row['raw_cot']
        raw_cot_answer = row['Output_Answer']

        result_df_dict['CaseID'].append(row_idx)
        result_df_dict['Category'].append(subject)
        result_df_dict['Question'].append(question)
        result_df_dict['Correct Answer'].append(standardize_answer(correct_answer))
        result_df_dict['raw_cot'].append(cot)
        print('\n\n\n')
        print('question: ',question)
        print('correct answer: ',correct_answer)
        print('COT: ',cot)
        print('raw_cot_answer: ',raw_cot_answer)
        print('\n\n\n')
        result_df_dict['Raw COT Answer'].append(standardize_answer(raw_cot_answer))

        # for i in range(config['num_agents']):
        has_mistake = True
        counter = 0
        new_answer = (standardize_answer(raw_cot_answer))
        corrected_cot = cot

        try:
            steps_list_with_indices = re.split(r'(?i)([Ss]tep \d+\s?:)', cot)

            # Reconstruct the steps list to include "step n:" with the actual step text.
            result_steps = [f"{steps_list_with_indices[i]} {steps_list_with_indices[i + 1].strip()}" for i in
                            range(1, len(steps_list_with_indices), 2)]
        except:
            result_steps = []
        if len(result_steps) == 0:
            result_steps = ['No initial thoughts proposed, start from the scratch']

        steps =  len(result_steps)


        check_list,partial_cot = self_correct_complete(result_steps, steps, question=question,subject=subject
                                                 )
        if 'YES' in check_list:
            corrected_cot, corrected_answer = generate_new_response(subject=subject,question=question,cot=partial_cot)
            new_answer = (standardize_answer(corrected_answer))
        else:
            has_mistake = False

            cot = corrected_cot
            counter += 1
        result_df_dict['Corrected COT Answer'].append(new_answer)
        result_df_dict['corrected_cot'].append(corrected_cot)
        result_df_dict['Hallu Seq'].append(check_list)
        if len(result_df_dict['CaseID']) >= 1:
            result_df = pd.DataFrame.from_dict(result_df_dict)
            result_df.to_csv('llm_project/result/df_final_no_debate.csv', mode='a', header=not header_written, index=False)
            header_written = True  # Ensure header is not written again
            # Clear the buffer
            for key in result_df_dict.keys():
                result_df_dict[key].clear()

    if len(result_df_dict['CaseID']) > 1:
        result_df = pd.DataFrame.from_dict(result_df_dict)
        result_df.to_csv('llm_project/result/df_final_no_debate.csv', mode='a', header=not header_written, index=False)
