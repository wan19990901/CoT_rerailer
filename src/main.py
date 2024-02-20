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
    forward_result = output_repraser(result)
    for key, value in forward_result.items():
        print(key)
        print(value)
    print('------------------------------------------------------')
    cot, steps, final_answer = forward_result.values()
    return cot, steps, final_answer


def geneate_ngram_step_checker(cot, steps, final_answer, ngram=1):
    check_list = []
    for i in range(int(steps)):
        current_step = i + 1

        if ngram == 'all':
            ngram = int(steps)

        if ngram < current_step:
            if current_step == int(steps):
                pattern = f'[sS]tep\s?{current_step - ngram + 1}.*'
                match = re.search(pattern, cot, re.DOTALL)
                if match:
                    masked_cot = match.group()
                else:
                    print("No match found.")
                    masked_cot = cot
            else:
                pattern = f'[sS]tep\s?{current_step - ngram + 1}.*?(?=[sS]tep\s?{current_step + 1})'
                match = re.search(pattern, cot, re.DOTALL)
                if match:
                    masked_cot = match.group()
                else:
                    print("No match found.")
                    masked_cot = cot
        else:
            pattern = f'[sS]tep\s?1.*?(?=[sS]tep\s?{current_step + 1})'
            match = re.search(pattern, cot, re.DOTALL)
            if match:
                masked_cot = match.group()
            else:
                print("No match found.")
                masked_cot = cot

        conditional_check_result = ngram_checker_agent(subject=subject, current_step=current_step, cot=masked_cot,
                                                       final_answer=final_answer, question=question)
        response = output_repraser(conditional_check_result)
        print(f'Step {current_step}', response, '\n\n')
        check_list.append((response['Step Correctness'], response['Logic Consistency']))
    print('------------------------------------------------------')
    print(check_list)
    print('------------------------------------------------------')
    print(f'The ground truth answer should be: {correct_answer}')

    return check_list


if __name__ == '__main__':
    config = {
        'dataset_fp':'combined_df.csv',
        'test_case_number':145,
        'ngram': 3
    }

    df = load_df(config['dataset_fp'])
    test_sample = select_sample(df, config['test_case_number'])

    subject = test_sample['Category']
    question = test_sample['Question']
    correct_answer = test_sample['Correct Answer']

    cot, steps, final_answer = generate_cot_response(subject, question)
    check_list = geneate_ngram_step_checker(cot, steps, final_answer, config['ngram'])
