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
    result = correct_answer_agent_partial_cot(subject=subject, question=question,cot=cot)
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
            break
    print('------------------------------------------------------')
    print(masked_cot)
    print('------------------------------------------------------')
    print(f'The ground truth answer should be: {correct_answer}')

    return check_list, masked_cot


def standardize_answer(answer):
    # Check for strict multiple choice format (single letter or letter followed by parenthesis)
    if re.match(r'^[a-zA-Z]\W.*$', answer.strip()):
        return answer.strip().lower()[0]

    # For other cases, return the answer as is
    return answer.lower()

def multi_agents_debate(subject,current_step,masked_cot,question,response):
    final_response = response
    print('Start Debating')
    attempts = 0
    counter = 0
    while (attempts < 1) and counter <=2:
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
        result_df_dict['Correct Answer'].append(standardize_answer(correct_answer))
        result_df_dict['raw_cot'].append(cot)

        print('question: ',question)
        print('correct answer: ',correct_answer)
        print('COT: ',cot)
        print('raw_cot_answer: ',raw_cot_answer)

        result_df_dict['Raw COT Answer'].append(standardize_answer(raw_cot_answer))
        matches = re.findall(r'(?i)([Ss]tep \d+\s?:)', cot)

        # Split the string into steps without removing the "step n:" part
        steps_list_with_indices = re.split(r'(?i)([Ss]tep \d+\s?:)', cot)

        # Reconstruct the steps list to include "step n:" with the actual step text.
        result_steps = [f"{steps_list_with_indices[i]} {steps_list_with_indices[i + 1].strip()}" for i in
                        range(1, len(steps_list_with_indices), 2)]

        steps =  len(result_steps)

        multi_checker = []
        for i in range(config['num_agents']):
            check_list,partial_cot = self_correct_complete(result_steps, steps, question=question,
                                                     ngram=config['ngram'])
            if 'YES' in check_list:
                corrected_cot, corrected_answer = generate_new_response(subject=subject,question=question,cot=partial_cot)
                result_df_dict['Corrected COT Answer'].append(standardize_answer(corrected_answer))
            else:
                result_df_dict['Corrected COT Answer'].append(standardize_answer(raw_cot_answer))
            multi_checker.append(check_list)

        for index, content in enumerate(multi_checker):
            print(f'{index}: {content}')

        majority_vote_list = majority_vote(multi_checker)
        # print('\n\nMajority Vote: ', majority_vote_list)
        result_df_dict['corrected_cot'].append(corrected_cot)
        result_df_dict['Hallu Seq'].append(check_list)

    result_df = pd.DataFrame.from_dict(result_df_dict)
    result_df.to_csv('../result/error_analysis_partial_cot_1attempt.csv')