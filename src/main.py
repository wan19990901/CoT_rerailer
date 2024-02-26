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

def generate_variable_extractor(cot, steps, ngram=1):
    check_list = []
    previous_variables = 'N/A'
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
        success= False
        while not success:
            try:
                conditional_check_result = variable_agent(subject=subject, current_step=current_step, cot=masked_cot,
                                                          question=question,previous_variables=previous_variables)
                response = output_repraser(conditional_check_result)

                print(f'Step {current_step}:')
                for key,value in response.items():
                    print(f'{key}: {value}\n')

                previous_variables = (response['Current Variable'])
                response2 = output_repraser(ngram_checker_agent2(subject=subject, current_step=current_step, cot=masked_cot,
                                                               extracted_var=previous_variables, question=question))
                print(f'Step {current_step}', response2, '\n\n')
                check_list.append((response2['Step Correctness'], response2['Logic Consistency']))
                success = True
            except:
                success = False
    print('------------------------------------------------------')
    print(check_list)
    print('------------------------------------------------------')
    print(f'The ground truth answer should be: {correct_answer}')

    return check_list

def majority_vote(checker_seq_list):
    transposed_list = list(zip(*checker_seq_list))

    # Majority vote for each tuple position
    majority_vote_list = []
    for tuples in transposed_list:
        # Count occurrences of each tuple and find the most common
        counter = Counter(tuples)
        most_common_tuple, _ = counter.most_common(1)[0]
        majority_vote_list.append(most_common_tuple)
    return majority_vote_list


if __name__ == '__main__':
    config = {
        'dataset_fp': 'Self_Check.csv',
        'test_case_number': 145,
        'ngram': 1,
        'num_agents': 3
    }

    df = load_df(config['dataset_fp'])
    test_sample = select_sample(df, config['test_case_number'])

    subject = test_sample['Category']
    question = test_sample['Question']
    correct_answer = test_sample['Correct Answer']

    cot, steps, final_answer = generate_cot_response(subject, question)


    for ngram in [3,'all']:
        multi_checker = []
        for i in range(config['num_agents']):
            check_list = generate_variable_extractor(cot, steps,
                                                     ngram=ngram)

            multi_checker.append(check_list)

        for index, content in enumerate(multi_checker):
            print(f'{index}: {content}')

        majority_vote_list = majority_vote(multi_checker)
        print('\n\nMajority Vote: ', majority_vote_list)
