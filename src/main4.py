import os
import pandas as pd
import re
from collections import Counter
from llm_agents import *

PREPROCESSED_FP = '../data/preprocessed'


def load_df(dataset_fp):
    df = pd.read_csv(os.path.join(PREPROCESSED_FP, dataset_fp))
    print('------------------------------------------------------')
    print(f'The distribution of category in {dataset_fp} is:\n{Counter(df.Category)}')
    return df


def select_sample(df, test_case_number=0):
    test_sample = df.iloc[test_case_number]
    print(f'\nSample {test_case_number} Question:')
    print(test_sample.Question)
    print(f'\nSample {test_case_number} Correct Answer:')
    print(test_sample['Correct Answer'])
    print('------------------------------------------------------')
    return test_sample


def generate_cot_response(subject, question):
    result = cot_agent(subject=subject, question=question)
    success = False
    while not success:
        try:
            forward_result = output_repraser(result)
            success = True
        except:
            success = False
    for key, value in forward_result.items():
        print(key)
        print(value)
    print('------------------------------------------------------')
    cot, steps, final_answer = forward_result.values()
    return cot, steps, final_answer

def self_correct_complete(old_cot, steps, question, ngram=1):

    success = False
    for i in range(3):
        while not success:
            try:
                conditional_check_result = debate_whole_agent(subject=subject, cot=old_cot,
                                                                question=question)
                response = output_repraser(conditional_check_result)

                success = True
            except:
                success = False

        print(response, '\n\n')
        old_cot = response['Corrected COT']


    print('------------------------------------------------------')

    print('------------------------------------------------------')
    print(f'The ground truth answer should be: {correct_answer}')

    return response['Step Verification'], response['Corrected COT'], response['Final Answer']



if __name__ == '__main__':
    config = {
        'dataset_fp': 'Self_Check.csv',
        'test_case_number': range(0,146),
        'ngram': 'all',
        'num_agents': 1
    }

    result_df_dict = {
        'CaseID': [],
        'Question': [],
        'Correct Answer':[],
        'Raw COT Answer':[],
        'Corrected COT Answer': [],
        'Hallu Seq':[],
        'raw_cot':[],
        'corrected_cot': []
    }

    df = pd.read_csv('../data/3-12_data.csv')

    for row_idx in range(len(df)):
        row = df.iloc[row_idx]


        subject = row['Category']
        question = row['Question']
        correct_answer = row['Correct_Answer']
        cot = row['Cot']
        raw_cot_answer = row['Output_Answer']

        result_df_dict['CaseID'].append(row_idx)
        result_df_dict['Question'].append(question)
        result_df_dict['Correct Answer'].append(correct_answer)
        result_df_dict['raw_cot'].append(cot)

        print('question: ',question)
        print('correct answer: ',correct_answer)
        print('COT: ',cot)
        print('raw_cot_answer: ',raw_cot_answer)

        result_df_dict['Raw COT Answer'].append(raw_cot_answer)
        matches = re.findall(r'(?i)([Ss]tep \d+\s?:)', cot)

        # Split the string into steps without removing the "step n:" part
        steps_list_with_indices = re.split(r'(?i)([Ss]tep \d+\s?:)', cot)

        # Reconstruct the steps list to include "step n:" with the actual step text.
        result_steps = [f"{steps_list_with_indices[i]} {steps_list_with_indices[i + 1].strip()}" for i in
                        range(1, len(steps_list_with_indices), 2)]

        steps =  len(result_steps)

        multi_checker = []
        for i in range(config['num_agents']):
            check_list,corrected_cot,corrected_answer = self_correct_complete(result_steps, steps, question=question,
                                                     ngram=config['ngram'])



        result_df_dict['corrected_cot'].append(corrected_cot)
        result_df_dict['Hallu Seq'].append(check_list)
        result_df_dict['Corrected COT Answer'].append(corrected_answer)
    result_df = pd.DataFrame.from_dict(result_df_dict)
    result_df.to_csv('../result/error_analysis_whole_attempt_3_debate.csv')
