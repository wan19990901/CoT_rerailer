from langchain_openai import ChatOpenAI
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
                                current_step=current_step, cot=cot, question=question)

            success = True
        except:
            worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
            chain = worker.chain_generator(system_prompt, human_prompt)
            out_put = chain.run(subject=subject,
                                current_step=current_step, cot=cot, question=question)
    return out_put


def debate_agent(subject, question, current_step, cot, response, temp=0, model_name='gpt-4-0125-preview'):
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
        Thinking.
        \n{format_instructions}
''')
    human_prompt = ("Here is my complete thought process {cot} and this is the original question {question}. The full"
                    "response from the other agents were given as {response}")

    response_schemas = [

        ResponseSchema(name="Justification",
                       description='''
                        Give me the your response to the other agent and justify
                         that whether you think the other agents' correction to my step was correct.
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
                                current_step=current_step, cot=cot, question=question, response=response)

            success = True
        except:
            worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
            chain = worker.chain_generator(system_prompt, human_prompt)
            out_put = chain.run(subject=subject,
                                current_step=current_step, cot=cot, question=question, response=response1)
    return out_put


def correct_answer_agent_partial_cot(subject, cot,question, temp=0, model_name='gpt-4-0125-preview'):
    system_prompt = (
        '''
        You are a professional specialized in {subject}. Your task is help me answer the question based on my initial 
        thoughts. I will provide you several steps of my attempt. Your task is to CONTINUE my thought process and then 
        answer my question step by step. Also, maximum 12 steps allowed and you can assume my initial thoughts had been
        checked since could be trusted. Remember, your response should based on my initial thoughts!
        \n{format_instructions}
        ''')
    human_prompt = "Here is my question :{question}. And my intial thought process is given as {cot}"

    response_schemas = [
        ResponseSchema(name="Complete Thought Process",
                       description="Continue my thought process in order to answer the question,"
                                   "You must include my initial thought process as well."
                                   "Return the complete chain of thought by following the format:"
                                   "Step n: [step process]."),
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


def debate_whole_agent(subject, question, cot, temp=0, model_name='gpt-4-0125-preview'):
    system_prompt = (
        '''You are a professional specialized in {subject}. You need to help me verify the other agents' thought process 
        when they solve the question.

        Before you perform the task, I want you to keep in mind several definitions for my possible mistakes. 
        1. Factuality： This type of error emphasizes the discrepancy between generated content and verifable real-word facts, including
        factual inconsistency or fabrication. In mathematics for instance, it may represents the computational error.

        2. Faithfulness: This type of error refers to the divergence of my step analysis from the original question or 
        previous steps, as well as self-consistency within my steps. In mathematics for instance, it may represents that
        I understood the question wrongly or my proposed step is inconsistent with my previous step. 

        Therefore, I need you to critically debate with the other agents. Your goal is to check their thought process,
        identify which step made mistakes and what type of hallucination were those. Then , generate your own version
        based on your justification and their thought process. Finally, generate a log to state your updates.
        \n{format_instructions}
''')
    human_prompt = ("Here is their complete thought process {cot} and this is the original question {question}. ")

    response_schemas = [

        ResponseSchema(name="Justification",
                       description='''
                        Give me the your response to the other agent and justify
                         that whether you think the other agents' thought process to solve the question was correct.
                        '''),

        ResponseSchema(name="Step Verification",
                       description='''
                       State which step had problem and what type by following the format:
                       [step n]: [Mistake type (factuality or faithfulness)]
                        '''),

        ResponseSchema(name="Corrected COT",
                       description='''
                               Help me generate a new version of thought process to solve the question. You should follow
                               the format: 
                               [Step n]: [Step process]                             
                               '''),
        ResponseSchema(name="Updates",
                       description='''
                                   Give me a log history regarding your updates by stating what you changed and what are
                                   the new steps you proposed by following the format:
                                   'Updated Steps': [step indices],
                                   'New Steps': [step indices]                            
                                   '''),
        ResponseSchema(name="Final Answer",
                       description="Give me your final answer based your revised thought process , if have options provided, "
                                   "just give me the option index. Follow"
                                   "The format [final_answer or correct_option_index]")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    success = False
    while not success:
        try:
            worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
            chain = worker.chain_generator(system_prompt, human_prompt)
            out_put = chain.run(subject=subject,
                                 cot=cot, question=question)

            success = True
        except:
            worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
            chain = worker.chain_generator(system_prompt, human_prompt)
            out_put = chain.run(subject=subject,
                                cot=cot, question=question)
    return out_put
