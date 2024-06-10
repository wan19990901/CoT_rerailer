import os
import pandas as pd
from llm_agents import *
from Parsers import *
from utils import check_consistency
import numpy as np
from tqdm import tqdm
import re
llm_config = {
    # change these three together
    'llm_type': 'openai',  # openai, ollama, anthropic
    'api_key_link': 'api_key.txt',
    'model': "gpt-4o",  # see llm_model.txt
    'temperature': 0,
}
with open(llm_config['api_key_link'], 'r') as f:
    api_key = f.read()

def load_data(data_dir):
    """Load data from the specified directory."""
    df = pd.read_csv(os.path.join(data_dir, 'filter_df.csv'))
    return df

def preprocess_samples(df, sample_size, random_seed=42):
    """Preprocess samples by choosing first n samples from specified categories and concatenating them."""
    # Since we're not sampling randomly, the random_seed parameter is not needed anymore,
    # but I'm keeping it in the function definition to maintain compatibility.
    
    # Define the categories of interest

    sample_data = pd.DataFrame()
    
    # Iterate over unique categories in the filtered DataFrame and take the first n samples
    for category in df['Category'].unique():
        category_samples = df[df['Category'] == category].sample(sample_size, replace=False, random_state=random_seed)
        sample_data = pd.concat([sample_data, category_samples])
    
    return sample_data


def run_experiment(temp_df):
    # Initialize lists to store results
    categories = []
    questions = []
    correct_answers = []
    consistencies = []
    cots = []
    selected_answers = []
    cot_debug = []
    answers_debug = []
    confidences = []
    # final_answers = []
    current_category = temp_df['Category'].iloc[0]
    print(current_category)
    # Iterate over rows of temp_df for the specific category
    for _, sample in tqdm(temp_df.iterrows(), total=len(temp_df), desc="Processing samples"):
        subject = sample['Category']
        question = sample['Question']
        correct_answer = sample['Correct Answer']
        results = []  # cot results
        answers = []  # GPT Generated answers
        

        for _ in range(3):
            cot_agent = LLM_agent(llm_type=llm_config['llm_type'], api_key=api_key, model=llm_config['model'],
                                      temperature=llm_config['temperature'])
            cot_agent.set_prompt('prompt_templates/cot_agent.json')
            cot_agent.set_parser(CoT_Agent)
            arguments_dict_cot = {
                'subject': subject,
                'question': question
            }
            result = cot_agent.involk(arguments_dict_cot)

            answers_debug.append(result['Final_Answer'])
            answers.append(result['Final_Answer'])
            cot_debug.append(result['Chain_of_Thought'])
            results.append(result['Chain_of_Thought'])

        consistency = check_consistency(answers)

        # Update cot based on judged_cot
        if consistency:
            cot = None  # If consistent, put NA
            answer = result['Final_Answer']
            confidences.append(100)
        else:
            judge_agent = LLM_agent(llm_type=llm_config['llm_type'], api_key=api_key, model=llm_config['model'],
                                      temperature=llm_config['temperature'])
            judge_agent.set_prompt('prompt_templates/judge.json')
            judge_agent.set_parser(Judge)
            arguments_dict_judge = {
                'subject': subject,
                'question': question,
                'cot1':results[0],
                'cot2': results[1],
                'cot3': results[2],
            }
            judged_cot_str = judge_agent.involk(arguments_dict_judge)['Selected_COT']

            print('Selected Index:', judged_cot_str)
            if judged_cot_str != 'None':
                cot = results[int(judged_cot_str) - 1]  # Use the selected COT index
                answer = answers[int(judged_cot_str) - 1]
            else:  # None selected, we will select the last one
                cot = result['Chain_of_Thought']
                answer = result['Final_Answer']
        if(subject!=current_category):
            temp_df = pd.DataFrame({
                'Category': categories,
                'Question': questions,
                'Correct_Answer': correct_answers,
                'Output_Answer': selected_answers,
                'Consistency': consistencies,
                'Cot': cots,
            })
            debug_df = pd.DataFrame({
            'Chain of Thought': cot_debug,
            'Generated Answers': answers_debug
            })
            temp_df.to_csv(f'result_{current_category}.csv', index=False)
            debug_df.to_csv(f'debug_{current_category}.csv', index=False)
            current_category = subject
        selected_answers.append(answer)
        categories.append(subject)
        questions.append(question)
        correct_answers.append(correct_answer)
        consistencies.append(consistency)
        cots.append(cot)



    # Construct dataframe for the specific category
    category_df = pd.DataFrame({
        'Category': categories,
        'Question': questions,
        'Correct_Answer': correct_answers,
        'Output_Answer': selected_answers,
        'Consistency': consistencies,
        'Cot': cots,
    })
    debug_df = pd.DataFrame({
    'Chain of Thought': cot_debug,
    'Generated Answers': answers_debug
    })

    return category_df,debug_df

def save_results(final_df, debug_df):
    """Save the final and debug dataframes to CSV files."""
    final_df.to_csv('result.csv', index=False)
    debug_df.to_csv('debug_output.csv', index=False)

if __name__ == '__main__':
    PREPROCESSED_FP = '../data/preprocessed'
    df = load_data(PREPROCESSED_FP)
    temp_df = preprocess_samples(df,sample_size=10)
    final_df, debug_df = run_experiment(temp_df)
    save_results(final_df, debug_df)
