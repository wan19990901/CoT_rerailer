from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import json
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains import LLMChain
from dotenv.main import load_dotenv


class ChatModelWorker:
    def __init__(self, output_parser, temperature=0, model='gpt-4'):
        with open('api_key.txt', 'r') as f:
            apikey = f.read()
        self.chat_model = ChatOpenAI(openai_api_key=apikey, model_name=model, temperature=temperature)
        self.output_parser = output_parser

    def prompt_temps(self, sys_temp, human_temp, format_instructions):
        sys_msg_prompt = SystemMessagePromptTemplate.from_template(sys_temp)
        human_msg_prompt = HumanMessagePromptTemplate.from_template(human_temp)
        chat_prompt = ChatPromptTemplate(partial_variables={"format_instructions": format_instructions},
                                         messages=[sys_msg_prompt, human_msg_prompt])
        return chat_prompt

    def chain_generator(self, template, human_template):
        output_parser = self.output_parser
        format_instructions = output_parser.get_format_instructions()
        chain = LLMChain(
            llm=self.chat_model,
            prompt=self.prompt_temps(template, human_template, format_instructions)
        )
        return chain


def output_repraser(input_string):
    json_str = input_string.strip('```json\n').rstrip('\n```').strip()

    # Step 2: Parse the JSON string into a dictionary
    data_dict = json.loads(json_str)
    return data_dict

#gpt-4-0125-preview,gpt-3.5-turbo-0125
def cot_agent(subject, question, temp=0, model_name='gpt-3.5-turbo-0125'):
    system_prompt = (
        "You are a professional specialized in {subject}. You need to help me answer the given question."
        "Notice that you need to solve the question step by step and as detailed as possible. Do not jump to the answer directly."
        "If it is a math question, please provide me the  detailed calculation in your steps, not just say the method!!!"
        "Your intermediate steps and thoughts are critical!. Also, maximum 10 steps allowed"
        "\n{format_instructions}")
    human_prompt = "{question}"

    response_schemas = [
        ResponseSchema(name="Chain of Thought",
                       description="Provide step by step analysis. For instance, you should follow the pattern 'step 1:... \nstep 2:...'"),
        ResponseSchema(name="Number of Steps Proposed",
                       description="return a simple integer output which indicates the number of steps you proposed in the Chain of Thought"),
        ResponseSchema(name="Final Answer",
                       description="Give me your final answer, if have options provided, just give me the option index. Follow"
                                   "The format [final_answer or correct_option_index]")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    success = False
    while not success:
        try:
            worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
            chain = worker.chain_generator(system_prompt, human_prompt)
            out_put = chain.run(subject=subject,
                                question=question)

            success = True
        except:
            worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
            chain = worker.chain_generator(system_prompt, human_prompt)
            out_put = chain.run(subject=subject,
                                question=question)
    return out_put


def ngram_checker_agent(subject, question, current_step, cot, temp=0, model_name='gpt-4-0125-preview'):
    system_prompt = (
        '''You are a professional specialized in {subject}. You need to help me verify my steps when I solve the question.
        I am currently at step #{current_step}.

        Is the current step logically or computationally correct in condition of all previous steps?. Notice that you should follow my throught process
        and determine the correctness using my approach.Also, if it involves equation computation, any format of equation should be considered as CORRECT
        as long as it holds. If my equation is slightly different than yours but if the result in correct computational result. It should be considered as CORRECT. 

        If it is correct, then verify that if my current step make the previous step hold. In other words, 
        check the logic consistency of step n in conditional of <step k to step n-1> where n is my current step and k is the first step number in the provided cot steps.
        provided thourght process. In the other words, is my current step supported by previous n-1 steps? It is important that for each analysis, ignore 
        steps other than the current step and the previous steps! In addition, for your reference, the question is given as {question}. 

        At step 1, since we have no step 0, instead the correctness and consistency should reflect if I correctly understood the answer.
        \n{format_instructions}
''')
    human_prompt = "Here is my complete thought process {cot}"

    response_schemas = [
        ResponseSchema(name="Steps",
                       description='''
                       States step indices on the current provided in the cot as [indices]
                       '''),
        ResponseSchema(name="Verification",
                       description='''
                        Help me verify the correctness and the logic consistency  of the current step, 
                       and tell me the reason. If they are due to the current step, say [Caused by step n], if due to previous
                       step, clearly indicate which step cause the inconsistency or incorrectness by saying [Caused by step a, b ...]
                       REASON is important. !!! If at Step 1, since 
                        we have no step 0, verify if I correctly understood the answer
                        '''),
        ResponseSchema(name="Corrected Step",
                       description='''
                        According to your analysis of correctness and consistency, help me revise the current step so that
                        it becomes correct and consistent.
                    '''),
        ResponseSchema(name="Step Correctness",
                       description='''
                       say [YES] if the logic is correct and the question is well-understood, otherwise [NO] .!!! If at Step 1, since 
                        we have no step 0, instead the correctness should reflect if I correctly understood the answer.
                        '''),
        ResponseSchema(name="Logic Consistency",
                       description='''
                       say [YES] if consistent, otherwise [NO].!!! If at Step 1, since 
                        we have no step 0, instead say [N/A]
                        ''')
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    success = False
    while not success:
        try:
            worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
            chain = worker.chain_generator(system_prompt, human_prompt)
            out_put = chain.run(subject=subject,
                                current_step=current_step, cot=cot, final_answer=final_answer, question=question)

            success = True
        except:
            worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
            chain = worker.chain_generator(system_prompt, human_prompt)
            out_put = chain.run(subject=subject,
                                current_step=current_step, cot=cot, final_answer=final_answer, question=question)
    return out_put


def variable_agent(subject, question, cot,current_step,previous_variables ,temp=0, model_name='gpt-4-0125-preview'):
    system_prompt = (
        '''
        You are a professional specialized in {subject}. You need to help me extract key variables when I solve the question.
        Your task is to extract the key variables from step analysis. Update the existing variable when necessary. My current
        step index is at {current_step}.
        \n{format_instructions}''')
    human_prompt = (
        '''Here is my complete thought process {cot} and here is the extract variable list from your buddy agent
           {previous_variables}. If the buddy agent did not give you anything, it implies that you are the first agent
           to analyze the step and define the initial variable based on the question.              
                    ''')

    response_schemas = [
        ResponseSchema(name="Current step",
                       description="State what is your current step: [current_step]"),
        ResponseSchema(name="Previous variable",
                       description='''
                       State what are the previous step that your buddy gave to you. If you are the first agent,
                       simple return [N/A] otherwise, return [previous variables]
                       '''),
        ResponseSchema(name="Current Variable",
                       description='''
                       based on your analysis of the current step and the previous variables (if applied), update the 
                       existing variable list, if necessary, you can define new variable. Your answer should look like:
                       [[variable1] = [value2] ---> [definition of the variable1],\n
                       [variable2] = [value2] ---> [definition of the variable1],\n...
                       ]
                       '''),
        ResponseSchema(name="Update Summarization",
                       description='''
                           You should clearly state what variable did you update and create by following the template:
                           [updated variables: [variable 1]: [old value] -> [new value]\n...
                           new variables: [new variable 1], [new variable 2]\n...
                           ]
                           '''),
        ResponseSchema(name="Used Formula",
                       description='''
                               You should state the formula use in the current step in terms of 
                                variables, following the template:
                               [Formula]: [formula 1], [formula 2] ...
                               If there is no formula, just return [Formula]: N/A
                               ''')
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    success = False
    while not success:
        try:
            worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
            chain = worker.chain_generator(system_prompt, human_prompt)
            out_put = chain.run(subject=subject, cot = cot, current_step = current_step,previous_variables=previous_variables,
                                question=question)

            success = True
        except:
            worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
            chain = worker.chain_generator(system_prompt, human_prompt)
            out_put = chain.run(subject=subject, cot = cot, current_step = current_step,previous_variables=previous_variables,
                                question=question)
    return out_put

def ngram_checker_agent2(subject, question, current_step, cot,extracted_var, temp=0, model_name='gpt-4-0125-preview'):
    system_prompt = (
        '''You are a professional specialized in {subject}. You need to help me verify my steps when I solve the question.
        I am currently at step #{current_step}.

        Is the current step  computationally correct in condition of provided variables and formulas?. 
        To justify the correctness, please refer to the extracted variable and formula I provided to you.  
        YOU MUST NOTICE THAT MATHEMATICAL EQUIVALENT STATEMENTS ARE THE SAME EVEN THOUGH THEY HAVE DIFFERENT EXPRESSIONS!
        THEY ARE BOTH CORRECT!

        If it is correct, then verify that if my current step make the previous step hold. In other words, 
        check the logic consistency of step n in conditional of <step k to step n-1> where n is my current step and k is the first step number in the provided cot steps.
        provided thought process. In the other words, is my current step supported by previous n-1 steps? It is important that for each analysis, ignore 
        steps other than the current step and the previous steps! In addition, for your reference, the question is given as {question}. 

        At step 1, since we have no step 0, instead the correctness and consistency should reflect if I correctly understood the answer.
        \n{format_instructions}
''')
    human_prompt = "Here is my complete thought process {cot}, you can find all the variable and formula you need on {extracted_var}"

    response_schemas = [

        ResponseSchema(name="Verification",
                       description='''
                        Help me verify the correctness and the logic consistency  of the current step, 
                       and tell me the reason. 
                       REASON is important. The reasoning step should cite the variable and formula you use!!! If at Step 1, since 
                        we have no step 0, verify if I correctly understood the answer
                        '''),
        ResponseSchema(name="Corrected Step",
                       description='''
                        According to your analysis of correctness and consistency, help me revise the current step so that
                        it becomes correct and consistent.
                    '''),
        ResponseSchema(name="Step Correctness",
                       description='''
                       say [YES] if the logic and computation are correct and the question is well-understood, otherwise [NO] .!!! If at Step 1, since 
                        we have no step 0, instead the correctness should reflect if I correctly understood the answer.
                        Is the current step  computationally correct in condition of provided variables and formulas?. 
                        To justify the correctness, please refer to the extracted variable and formula I provided to you.  
                        '''),
        ResponseSchema(name="Logic Consistency",
                       description='''
                       say [YES] if consistent, otherwise [NO].!!! If at Step 1, since 
                        we have no step 0, instead say [N/A]
                        '''),
        ResponseSchema(name="Dependency",
                       description='''
                           Find which previous steps led to the incorrectness or inconsistency . The whole idea is to 
                           discuss that if the current step is incorrect or inconsistent, where did the error chain start
                           from. What previous steps are the root cause of the error. Follow the template:
                           [[Incorrectness] <- [Incorrect Previous Steps]\n
                           [Inconsistency] <- [Inconsistent Previous Steps]]
                           If no incorrectness or inconsistency, simply return [N/A]
                           '''),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    success = False
    while not success:
        try:
            worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
            chain = worker.chain_generator(system_prompt, human_prompt)
            out_put = chain.run(subject=subject, extracted_var=extracted_var,
                                current_step=current_step, cot=cot,  question=question)

            success = True
        except:
            worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
            chain = worker.chain_generator(system_prompt, human_prompt)
            out_put = chain.run(subject=subject,extracted_var=extracted_var,
                                current_step=current_step, cot=cot, question=question)
    return out_put


def root_checker_agent(subject, question, current_step, cot, temp=0, model_name='gpt-4-0125-preview'):
    system_prompt = (
        '''You are a professional specialized in {subject}. You need to help me verify my steps when I solve the question.
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
        \n{format_instructions}
''')
    human_prompt = "Here is my complete thought process {cot} and this is the original question {question}"

    response_schemas = [

        ResponseSchema(name="Verification",
                       description='''
                        Help me verify the factuality and the faithfulness  of the current step, 
                       and tell me the reason. 
                       REASON is important. The reasoning step should cite the variable and formula you use!!! If at Step 1, since 
                        we have no step 0, verify if I correctly understood the answer
                        '''),

        ResponseSchema(name="Step Hallucination",
                       description='''
                       say [YES] if the current step logic and computation are NOT factual or faithful 
                       based on the question and my previous steps, otherwise [NO] .!!! If at Step 1, since 
                        we have no step 0, check for the factuality and faithfulness of the current step only. 
                        '''),
        ResponseSchema(name="Type of Hallucination",
                       description='''
                       Identify if the step violated factuality or faithfulness or both. Return [None] if my current step
                       was correct.
                        '''),
        ResponseSchema(name="Correction",
                       description='''
                               If you think Step Hallucination is Yes, help me generate a corrected version of the current
                               step instead. Notice that do not simply identify the error here, instead
                               you should directly give me the correct version with calculation (if applicable)
                               Follow the format:
                               Step n : [Corrected version]                               
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
                           '''),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    success = False
    while not success:
        try:
            worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
            chain = worker.chain_generator(system_prompt, human_prompt)
            out_put = chain.run(subject=subject,
                                current_step=current_step, cot=cot,  question=question)

            success = True
        except:
            worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
            chain = worker.chain_generator(system_prompt, human_prompt)
            out_put = chain.run(subject=subject,
                                current_step=current_step, cot=cot, question=question)
    return out_put


def debate_agent(subject, question, current_step, cot, response ,temp=0, model_name='gpt-4-0125-preview'):
    system_prompt = (
        '''You are a professional specialized in {subject}. You need to help me verify my steps when I solve the question.
        I am currently at step #{current_step}.

        Before you perform the task, I want you to keep in mind several definitions for my possible mistakes. 
        1. Factuality： This type of error emphasizes the discrepancy between generated content and verifable real-word facts, including
        factual inconsistency or fabrication. In mathematics for instance, it may represents the computational error.

        2. Faithfulness: This type of error refers to the divergence of my step analysis from the original question or 
        previous steps, as well as self-consistency within my steps. In mathematics for instance, it may represents that
        I understood the question wrongly or my proposed step is inconsistent with my previous step. 

        Other agents had helped me identify the error I made in the current step. You goal is to debate with the other
        agents and justify if their corrections were correct based on my question, thought process. Please use Critical
        Thinking. Please be harsh and picky since your verification is important!
        \n{format_instructions}
''')
    human_prompt = ("Here is my complete thought process {cot} and this is the original question {question}. The full"
                    "response from the other agents were given as {response}")

    response_schemas = [

        ResponseSchema(name="Justification",
                       description='''
                        Give me the reason that whether you think the other agents' correction to my step was correct.
                        '''),

        ResponseSchema(name="Agreement",
                       description='''
                       say [YES] if you agree with the other agents corrections to my current step analysis. Otherwise,
                       say [NO]
                        '''),

        ResponseSchema(name="Correction",
                       description='''
                               Help me generate a  version of the current
                               step that you think is correct
                               . Notice that do not simply identify the error here, instead
                               you should directly give me the correct version with calculation (if applicable)
                               
                               Step n : [Corrected version]                               
                               '''),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    success = False
    while not success:
        try:
            worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
            chain = worker.chain_generator(system_prompt, human_prompt)
            out_put = chain.run(subject=subject,
                                current_step=current_step, cot=cot, question=question, response = response)

            success = True
        except:
            worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
            chain = worker.chain_generator(system_prompt, human_prompt)
            out_put = chain.run(subject=subject,
                                current_step=current_step, cot=cot, question=question, response = response1)
    return out_put

def correct_answer_agent(subject, cot,question, temp=0, model_name='gpt-4-0125-preview'):
    system_prompt = (
        '''
        You are a professional specialized in {subject}. Your task is help me summarize my thought process towards a 
        question, then help me return the final answer to the question based on my thought process.
        \n{format_instructions}
        ''')
    human_prompt = "Here is my question :{question}. And my thought process is given as {cot}"

    response_schemas = [
        ResponseSchema(name="Summarization",
                       description="Summarize my thought process"),
        ResponseSchema(name="Final Answer",
                       description="Give me your final answer based on my thought process , if have options provided, "
                                   "just give me the option index. Follow"
                                   "The format [final_answer or correct_option_index]")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    success = False
    while not success:
        try:
            worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
            chain = worker.chain_generator(system_prompt, human_prompt)
            out_put = chain.run(subject=subject, cot = cot,
                                question=question)

            success = True
        except:
            worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
            chain = worker.chain_generator(system_prompt, human_prompt)
            out_put = chain.run(subject=subject, cot = cot,
                                question=question)
    return out_put