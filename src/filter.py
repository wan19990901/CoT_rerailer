import os
import pandas as pd
from llm_agents import cot_agent, judge_agent
from utils import check_consistency
import numpy as np

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
    for category in filtered_df['Category'].unique():
        category_samples = filtered_df[filtered_df['Category'] == category].sample(sample_size, replace=False, random_state=random_seed)
        sample_data = pd.concat([sample_data, category_samples])
    
    return sample_data

from tqdm import tqdm
import re
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
            result = cot_agent(model_name='gpt-4-0125-preview', subject=subject, question=question)
            answers_debug.append(result['Final Answer'])
            answers.append(result['Final Answer'])
            cot_debug.append(result['Chain of Thought'])
            results.append(result['Chain of Thought'])

        consistency = check_consistency(answers)

        # Update cot based on judged_cot
        if consistency:
            cot = None  # If consistent, put NA
            answer = result['Final Answer']
            confidences.append(100)
        else:
            judged_cot_str = judge_agent(subject, question, results)
            print('Selected Index:', judged_cot_str)
            if judged_cot_str != 'None':
                cot = results[int(judged_cot_str) - 1]  # Use the selected COT index
                answer = answers[int(judged_cot_str) - 1]
            else:  # None selected, we will select the last one
                cot = result['Chain of Thought']
                answer = result['Final Answer']
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
    df_self_check, df_mmlu = load_data(PREPROCESSED_FP)
    temp_df = preprocess_samples(df_self_check, df_mmlu)
    final_df, debug_df = run_experiment(temp_df)
    save_results(final_df, debug_df)
