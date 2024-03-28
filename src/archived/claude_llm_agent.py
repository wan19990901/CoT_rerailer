import anthropic
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import pandas as pd
API_KEY = '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'
def root_checker_agent(subject, current_step, cot,question):
    response_schemas = [

        ResponseSchema(name="Verification",
                       description='''
                                Help me verify the factuality and the faithfulness  of the current step, 
                               and tell me the reason. 
                               REASON is important. The reasoning step should cite the variable and formula you use!!! If at Step 1, since 
                                we have no step 0, verify if I correctly understood the answer
                                YOU MUST END THIS ITEM WITH  *<Verification>*
                                '''),

        ResponseSchema(name="Step Hallucination",
                       description='''
                               say "YES" if the current step logic and computation are NOT factual or faithful 
                               based on the question and my previous steps, otherwise "NO" .!!! If at Step 1, since 
                                we have no step 0, check for the factuality and faithfulness of the current step only. 
                                YOU MUST END THIS ITEM WITH  *<Step Hallucination>*
                                '''),
        ResponseSchema(name="Type of Hallucination",
                       description='''
                               Identify if the step violated factuality or faithfulness or both. Return "None" if my current step
                               was correct.
                               YOU MUST END THIS ITEM WITH  *<Type of Hallucination>*
                                '''),
        ResponseSchema(name="Correction",
                       description='''
                                       If you think Step Hallucination is Yes, help me generate a corrected version of the current
                                       step instead. Notice that do not simply identify the error here, instead
                                       you should directly give me the correct version with calculation (if applicable)
                                       Follow the format:
                                       "Correction: Step n : Corrected version...."    
                                       YOU MUST END THIS ITEM WITH  *<Correction>*                           
                                       '''),
        ResponseSchema(name="Dependency",
                       description='''
                                   Find which previous steps led to the unfactual or unfaithful . The whole idea is to 
                                   discuss that if the current step is unfactual or unfaithful, where did the error chain start
                                   from. What previous steps are the root cause of the error. Follow the template:
                                   [[Unfactual] <- [Unfactual Previous Steps Indices]\n
                                   [Unfaithful] <- [Unfaithful Previous Steps Indices]]
                                   If it is caused by misunderstanding of question, then the dependency should be [Original Question]
                                   If no unfactual or unfaithful, simply return [N/A]
                                   YOU MUST END THIS ITEM WITH  *<Dependency>*
                                   '''),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser
    system_prompt = (
        f'''You are a professional specialized in {subject}. You need to help me verify my steps when I solve the question.
        I am currently at step #{current_step}.

        Before you perform the task, I want you to keep in mind several definitions for my possible mistakes. 
        1. Factuality： This type of error emphasizes the discrepancy between generated content and verifable real-word facts, including
        factual inconsistency or fabrication. In mathematics for instance, it may represents the computational error.

        2. Faithfulness: This type of error refers to the divergence of my step analysis from the original question or 
        previous steps, as well as self-consistency within my steps. In mathematics for instance, it may represents that
        I understood the question wrongly or my proposed step is inconsistent with my previous step. 

        Based on my current step response, question, previous steps, and my error definitions, help me verify if any of 
        the mistakes (factuality or faithfulness) occur on my analysis. Notice that skipping step should not be considered
        as error as long as the calculation is correct! For instance, 2x+2 should be the same as 2+2x. Also
        2x+2+3 should be the same as 2x+5


        At step 1, since we have no step 0, instead the factuality and faithfulness check
         should reflect if I correctly understood the answer.
        YOU MUST FOLLOW THE FORMAT TO GIVE ME THE RESPONSE, THE FORMAT IS :\n{format_instructions}
''')
    human_prompt = f"Here is my complete thought process {cot} and this is the original question {question}"

    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=API_KEY,
    )

    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4000,
        temperature=0.0,
        system=system_prompt,
        messages=[
            {"role": "user", "content": human_prompt}
        ],
    )

    return message.content[0].__dict__['text']

def debate_agent(subject, question, current_step, cot, response):
    response_schemas = [

        ResponseSchema(name="Justification",
                       description='''
                                Give me the your response to the other agent and justify
                                 that whether you think the other agents' correction to my step was correct.
                                 YOU MUST END THIS ITEM WITH  *<Justification>*
                                '''),

        ResponseSchema(name="Agreement",
                       description='''
                               say "YES" if you agree with the other agents corrections to my current step analysis. Otherwise,
                               say "NO"
                               YOU MUST END THIS ITEM WITH  *<Agreement>*
                                '''),

        ResponseSchema(name="Correction",
                       description='''
                                       Help me generate a  version of the current
                                       step that you think is correct
                                       . Notice that do not simply identify the error here, instead
                                       you should directly give me the correct version with calculation (if applicable)
                                        Using the format
                                       'Correction: Step n : [Corrected version]'     
                                       YOU MUST END THIS ITEM WITH  *<Correction>*                          
                                       '''),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)


    system_prompt = (
        f'''You are a professional specialized in {subject}. You need to help me verify my steps when I solve the question.
        I am currently at step #{current_step}.

        Before you perform the task, I want you to keep in mind several definitions for my possible mistakes. 
        1. Factuality： This type of error emphasizes the discrepancy between generated content and verifable real-word facts, including
        factual inconsistency or fabrication. In mathematics for instance, it may represents the computational error.

        2. Faithfulness: This type of error refers to the divergence of my step analysis from the original question or 
        previous steps, as well as self-consistency within my steps. In mathematics for instance, it may represents that
        I understood the question wrongly or my proposed step is inconsistent with my previous step. 

        Other agents had helped me identify the error I made in the current step. You goal is to debate with the other
        agents and justify if their corrections were correct based on my question, thought process. Please use Critical
        Thinking.
        YOU MUST FOLLOW THE FORMAT TO GIVE ME THE RESPONSE, THE FORMAT IS :\n{output_parser}
''')
    human_prompt = (f'''Here is my complete thought process {cot} and this is the original question {question}. The full
                    response from the other agents were given as {response}''')


    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=API_KEY,
    )

    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4000,
        temperature=0.0,
        system=system_prompt,
        messages=[
            {"role": "user", "content": human_prompt}
        ],
    )

    return message.content[0].__dict__['text']


def correct_answer_agent_partial_cot(subject, cot,question):
    response_schemas = [
        ResponseSchema(name="Complete Thought Process",
                       description="Continue my thought process in order to answer the question,"
                                   "You must include my initial thought process as well."
                                   "Return the complete chain of thought by following the format:"
                                   "'Complete Thought Process: Step n: [step process]...'"
                                   ". YOU MUST END THIS ITEM WITH  *<Complete Thought Process>*"),
        ResponseSchema(name="Final Answer",
                       description="Give me your final answer based on my thought process , if have options provided, "
                                   "just give me the option index. Follow"
                                   "The format: "
                                   "'Final Answer: final_answer or correct_option_index'"
                       "YOU MUST END THIS ITEM WITH  *<Final Answer>*")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    system_prompt = (
        f'''
        You are a professional specialized in {subject}. Your task is help me answer the question based on my initial 
        thoughts. I will provide you several steps of my attempt. Your task is to CONTINUE my thought process and then 
        answer my question step by step. Also, maximum 12 steps allowed and you can assume my initial thoughts had been
        checked since could be trusted. Remember, your response should based on my initial thoughts!
        YOU MUST FOLLOW THE FORMAT TO GIVE ME THE RESPONSE, THE FORMAT IS :\n{output_parser}
        ''')
    human_prompt = f"Here is my question :{question}. And my intial thought process is given as {cot}"

    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=API_KEY,
    )

    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4000,
        temperature=0.0,
        system=system_prompt,
        messages=[
            {"role": "user", "content": human_prompt}
        ],
    )

    return message.content[0].__dict__['text']


def output_item(input_string, item_name):
    index_start = input_string.find(item_name + ':') + len(item_name + ':')

    # Find the index of the next newline character after "Step Hallucination:"
    index_end = input_string.find(f'*<{item_name}>*', index_start)

    # Extract the content immediately after "Step Hallucination:" and before the next newline
    hallucination_response = input_string[index_start:index_end].strip()

    return (hallucination_response)
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

    df = pd.read_csv('3-15_data.csv')


    row = df.iloc[0]


    subject = row['Category']
    question = row['Question']
    correct_answer = row['Correct_Answer']
    cot = row['Cot']
    raw_cot_answer = row['Output_Answer']
    current_step = 0
    root_checker_agent(subject, current_step, cot, question)