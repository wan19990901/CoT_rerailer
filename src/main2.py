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
def generate_new_response(subject, question,cot):
    result = correct_answer_agent(subject=subject, question=question,cot=cot)
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
    cot, final_answer = forward_result.values()
    return cot, final_answer

def self_correct_complete(cot, steps, question, ngram=1):
    check_list = []
    for i in range(int(steps)):
        current_step = i + 1

        masked_cot = cot[:i+1]


        success = False
        while not success:
            try:
                conditional_check_result = root_checker_agent(subject=subject, current_step=current_step, cot=masked_cot,
                                                                question=question)
                response = output_repraser(conditional_check_result)

                success = True
            except:
                success = False

        print(f'Step {current_step}', response, '\n\n')
        check_list.append((response['Step Hallucination']))
        if (response['Step Hallucination'] == 'YES'):
            debate_response = multi_agents_debate(subject,current_step,masked_cot,question,response)
            print('Old Version: ', masked_cot[i])
            masked_cot[i] = debate_response['Correction']
            print('Corrected Version', masked_cot[i])

    print('------------------------------------------------------')
    print(check_list)
    print('------------------------------------------------------')
    print(f'The ground truth answer should be: {correct_answer}')

    return check_list, masked_cot

def multi_agents_debate(subject,current_step,masked_cot,question,response):
    final_response = response
    print('Start Debating')
    attempts = 0
    counter = 0
    while (attempts < 2) and counter <=4:
        print('attempt:',attempts)
        success = False
        while not success:
            try:
                debate = debate_agent(subject=subject, current_step=current_step, cot=masked_cot,
                                                          question=question,response = response)
                response = output_repraser(debate)
                print('\n\n\n',response,'\n\n\n')
                if response['Agreement'] == 'YES':
                    final_response = response
                    attempts += 1

                success = True
                counter += 1
            except:
                success = False
    return final_response


def majority_vote(checker_seq_list):

    # Majority vote for each tuple position
    majority_vote = []
    for i in range(len(checker_seq_list[0])):
        # Get the i-th element from each list and count occurrences
        ith_elements = [lst[i] for lst in checker_seq_list]
        # Count occurrences of each answer for both elements in the tuple
        count_first = Counter([elem[0] for elem in ith_elements])
        count_second = Counter([elem[1] for elem in ith_elements])
        # Get the most common element with majority vote
        majority_first = count_first.most_common(1)[0][0]
        majority_second = count_second.most_common(1)[0][0]
        # Add the majority vote tuple to the result list
        majority_vote.append((majority_first, majority_second))

    return majority_vote


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
        'Hallu Seq':[]
    }

    df = load_df(config['dataset_fp'])
    for sample_id in config['test_case_number']:


        test_sample = select_sample(df, sample_id)

        subject = test_sample['Category']
        question = test_sample['Question']
        correct_answer = test_sample['Correct Answer']

        result_df_dict['CaseID'].append(sample_id)
        result_df_dict['Question'].append(question)
        result_df_dict['Correct Answer'].append(correct_answer)


        cot, steps, final_answer = generate_cot_response(subject, question)
        result_df_dict['Raw COT Answer'].append(final_answer)
        matches = re.findall(r'(?i)([Ss]tep \d+\s?:)', cot)

        # Split the string into steps without removing the "step n:" part
        steps_list_with_indices = re.split(r'(?i)([Ss]tep \d+\s?:)', cot)

        # Reconstruct the steps list to include "step n:" with the actual step text.
        result_steps = [f"{steps_list_with_indices[i]} {steps_list_with_indices[i + 1].strip()}" for i in
                        range(1, len(steps_list_with_indices), 2)]

        assert len(result_steps) == int(steps)

        multi_checker = []
        for i in range(config['num_agents']):
            check_list,corrected_cot = self_correct_complete(result_steps, steps, question=question,
                                                     ngram=config['ngram'])
            if 'YES' in check_list:
                _, corrected_answer = generate_new_response(subject=subject,question=question,cot=corrected_cot[:-1])
                result_df_dict['Corrected COT Answer'].append(corrected_answer)
            else:
                result_df_dict['Corrected COT Answer'].append(final_answer)
            multi_checker.append(check_list)

        for index, content in enumerate(multi_checker):
            print(f'{index}: {content}')

        majority_vote_list = majority_vote(multi_checker)
        # print('\n\nMajority Vote: ', majority_vote_list)
        result_df_dict['Hallu Seq'].append(check_list)

    result_df = pd.DataFrame.from_dict(result_df_dict)
    result_df.to_csv('../result/error_analysis.csv')
