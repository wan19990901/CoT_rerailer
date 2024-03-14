import os
import pandas as pd
from llm_agents import cot_agent, judge_agent
from utils import check_consistency
import numpy as np

def load_data(data_dir):
    """Load data from the specified directory."""
    df_self_check = pd.read_csv(os.path.join(data_dir, 'Self_Check.csv'))
    df_mmlu = pd.read_csv(os.path.join(data_dir, 'MMLU_test.csv'))
    return df_self_check, df_mmlu

def preprocess_samples(df_self_check, df_mmlu, sample_size=2, random_seed=42):
    """Preprocess samples by choosing random samples and concatenating them."""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    sample_self_check = df_self_check.sample(sample_size)
    sample_mmlu = df_mmlu.sample(sample_size)
    return pd.concat([sample_self_check, sample_mmlu])

def run_experiment(temp_df, cot_model='gpt-4-0125-preview'):
    """Run the experiment."""
    # Initialize lists to store results
    categories = []
    questions = []
    correct_answers = []
    consistencies = []
    cots = []
    selected_answers = []
    cot_debug = []
    answers_debug = []
    final_answers = []

    # Iterate over rows of temp_df
    for _, sample in temp_df.iterrows():
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
        else:
            judged_cot_str = judge_agent(subject, question, results)
            print(judged_cot_str)
            if judged_cot_str != 'None':
                cot = results[int(judged_cot_str) - 1]  # Use the selected COT index
                answer = answers[int(judged_cot_str) - 1]
            else:  # None selected, we will select the last one
                cot = result['Chain of Thought']
                answer = result['Final Answer']

        selected_answers.append(answer)
        categories.append(subject)
        questions.append(question)
        correct_answers.append(correct_answer)
        consistencies.append(consistency)
        cots.append(cot)

        # Set seed for reproducibility
        np.random.seed(fixed_seed)

        # Call forward_agent again with fixed seed
        final_result = forward_agent(model_name='gpt-4-0125-preview', subject=subject, question=question)
        final_answer = final_result['Final Answer']

        # Append final answer to list
        final_answers.append(final_answer)

    # Construct final dataframe
    final_df = pd.DataFrame({
        'Category': categories,
        'Question': questions,
        'Correct_Answer': correct_answers,
        'Output_Answer': selected_answers,
        'Consistency': consistencies,
        'Cot': cots,
        'Final_Answer': final_answers
    })

    
    debug_df = pd.DataFrame({
        'Chain of Thought': cot_debug,
        'Generated Answers': answers_debug
    })
    
    return final_df, debug_df

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
