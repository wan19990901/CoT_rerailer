from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from typing import List, Optional
import re
from langchain_community.chat_models import ChatOllama
from Parsers import *


class LLM_agent:
    def __init__(self, api_key=None, llm_type='openai', model="gpt-3.5-turbo-0125", temperature=0):
        self.api_key = api_key
        if llm_type == 'openai':
            self.llm = ChatOpenAI(openai_api_key=self.api_key, model_name=model, temperature=temperature)

        elif llm_type == 'anthropic':
            self.llm = ChatAnthropic(model=model,
                                     anthropic_api_key=self.api_key)
        elif llm_type == 'ollama':
            self.llm = ChatOllama(model=model)
        self.llm_type = llm_type
        self.chat_prompt = None

    def involk(self, var_dict):
        if self.llm_type == 'openai':
            output_parser = self.parser
        elif self.llm_type == 'anthropic':
            output_parser = extract_json

        if self.llm_type == 'ollama':
            chain = self.chat_prompt | self.llm
        else:
            chain = self.chat_prompt | self.llm | output_parser
        success = False
        attempts = 0
        response = None
        while (not success) and attempts < 3:
            try:
                response = chain.invoke(var_dict)
                if (self.llm_type != 'ollama'):

                    if len(response) == self.num_of_llm_output:
                        success = True
                    else:

                        print("Response length does not match the expected length.")
                else:
                    success = True
            except Exception as e:
                print("An exception occurred:", str(e))
                attempts += 1
                success = False

            if attempts == 3:
                print('Maximum attempts reached.')
                chain = self.chat_prompt | self.llm
                response = chain.invoke(var_dict)

        return response

    def get_llm(self):
        return self.llm

    def set_prompt(self, prompt_json_link):
        with open(prompt_json_link) as f:
            message = []
            for key, val in json.load(f).items():
                if (self.llm_type != 'ollama'):
                    if key == 'system':
                        val += 'You need to output your responses by following the format. NOT OUTPUT THIS FORMAT INSTRUCTION\n{format_instructions}'
                message.append((key, val))

        chat_prompt = ChatPromptTemplate.from_messages(message)
        self.chat_prompt = chat_prompt

    def get_prompt(self):
        return self.chat_prompt

    def set_parser(self, parser_obj):
        self.parser = JsonOutputParser(pydantic_object=parser_obj)
        self.num_of_llm_output = len(parser_obj.__fields__)
        self.chat_prompt = self.chat_prompt.partial(format_instructions=self.parser.get_format_instructions())

    def get_parser(self):
        return self.parser


def extract_json(message: AIMessage) -> List[dict]:
    """Extracts JSON content from a string where JSON is embedded between ```json and ``` tags.

    Parameters:
        text (str): The text containing the JSON content.

    Returns:
        list: A list of extracted JSON strings.
    """
    text = message.content
    # Define the regular expression pattern to match JSON blocks
    pattern = r'"([^"]+)":\s*"([^"]+)"'

    # Find all non-overlapping matches of the pattern in the string
    matches = re.findall(pattern, text)

    # Return the list of matched JSON strings, stripping any leading or trailing whitespace
    try:
        json_string = json.dumps(dict(matches), indent=4)
        return dict(matches)
    except Exception:
        raise ValueError(f"Failed to parse: {message}")


# def cot_agent(subject, question, temp=0, model_name='gpt-3.5-turbo-0125'):
#     system_prompt = (
#         "You are a professional specialized in {subject}. You need to help me answer the given question."
#         "Notice that you need to solve the question step by step and as detailed as possible. Do not jump to the answer directly."
#         "If it is a math question, please provide me the  detailed calculation in your steps, not just say the method!!!"
#         "Your intermediate steps and thoughts are critical!. Also, maximum 10 steps allowed"
#         "\n{format_instructions}")
#     human_prompt = "{question}"
#
#     response_schemas = [
#         ResponseSchema(name="Chain of Thought",
#                        description="Provide step by step analysis. For instance, you should follow the pattern 'step 1:... \nstep 2:...'"),
#         ResponseSchema(name="Number of Steps Proposed",
#                        description="return a simple integer output which indicates the number of steps you proposed in the Chain of Thought"),
#         ResponseSchema(name="Final Answer",
#                        description="Give me your final answer, if have options provided, just give me the option index. Follow"
#                                    "The format [final_answer or correct_option_index]")
#     ]
#     output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
#     success = False
#     while not success:
#         try:
#             worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
#             chain = worker.chain_generator(system_prompt, human_prompt)
#             out_put = chain.run(subject=subject,
#                                 question=question)
#
#             success = True
#         except:
#             worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
#             chain = worker.chain_generator(system_prompt, human_prompt)
#             out_put = chain.run(subject=subject,
#                                 question=question)
#     return out_put
#
#
# def root_checker_agent(subject, question, current_step, cot, temp=0, model_name='gpt-4-0125-preview'):
#     system_prompt = (
#         '''You are a professional specialized in {subject}. You need to help me verify my steps when I solve the question.
#         I am currently at step #{current_step}.
#
#         Before you perform the task, I want you to keep in mind several definitions for my possible mistakes.
#         1. Factuality： This type of error emphasizes the discrepancy between generated content and verifable real-word facts, including
#         factual inconsistency or fabrication. In mathematics for instance, it may represents the computational error.
#
#         2. Faithfulness: This type of error refers to the divergence of my step analysis from the original question or
#         previous steps, as well as self-consistency within my steps. In mathematics for instance, it may represents that
#         I understood the question wrongly or my proposed step is inconsistent with my previous step.
#
#         Based on my current step response, question, previous steps, and my error definitions, help me verify if any of
#         the mistakes (factuality or faithfulness) occur on my analysis. Notice that skipping step should not be considered
#         as error as long as the calculation is correct! For instance, 2x+2 should be the same as 2+2x. Also
#         2x+2+3 should be the same as 2x+5
#
#
#         At step 1, since we have no step 0, instead the factuality and faithfulness check
#          should reflect if I correctly understood the answer.
#
#          Do not detect any minor hallucinations! In other words, only targeting the mistakes that contain calculation error
#          or apparent logical flaw or contradict real-world facts! If the provided step acknowledge mistake, you need to
#          capture it and correct it.
#
#          If you see any step ended up with '*<verified>*' it means it have been checked without any mistake, so just consider
#          it as correct and do not have to give the verification. Simply say step hallu is [NO]!!!
#         \n{format_instructions}
# ''')
#     human_prompt = "Here is my complete thought process {cot} and this is the original question {question}"
#
#     response_schemas = [
#
#         ResponseSchema(name="Verification",
#                        description='''
#                         Help me verify the factuality and the faithfulness  of the current step,
#                        and tell me the reason.
#                        REASON is important. The reasoning step should cite the variable and formula you use!!! If at Step 1, since
#                         we have no step 0, verify if I correctly understood the answer
#                         If you see any step ended up with *<verified>* it means it have been checked without any mistake, so just return 'None'!!!
#                         '''),
#
#         ResponseSchema(name="Step Hallucination",
#                        description='''
#                        say 'YES' if the current step logic and computation are NOT factual or faithful
#                        based on the question and my previous steps, otherwise 'NO' .!!! If at Step 1, since
#                         we have no step 0, check for the factuality and faithfulness of the current step only.
#                         If you see any step ended up with *<verified>* it means it have been checked without any mistake, so just consider
#          it as correct and say 'NO'!!!
#                         '''),
#         ResponseSchema(name="Correction",
#                        description='''
#                                If you think Step Hallucination is Yes, help me generate a corrected version of the current
#                                step instead. Notice that do not simply identify the error here, instead
#                                you should directly give me the correct version with calculation (if applicable)
#                                Follow the format:
#                                'Step n : [Corrected version]'
#                                '''),
#         ResponseSchema(name="Dependency",
#                        description='''
#                            Find which previous steps led to the unfactual or unfaithful . The whole idea is to
#                            discuss that if the current step is unfactual or unfaithful, where did the error chain start
#                            from. What previous steps are the root cause of the error. Follow the template:
#                            [[Unfactual] <- [Unfactual Previous Steps Indices]\n
#                            [Unfaithful] <- [Unfaithful Previous Steps Indices]]
#                            If it is caused by misunderstanding of question, then the dependency should be [Original Question]
#                            If no unfactual or unfaithful, simply return [N/A]
#                            '''),
#     ]
#     output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
#     success = False
#     while not success:
#         try:
#             worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
#             chain = worker.chain_generator(system_prompt, human_prompt)
#             out_put = chain.run(subject=subject,
#                                 current_step=current_step, cot=cot, question=question)
#
#             success = True
#         except:
#             worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
#             chain = worker.chain_generator(system_prompt, human_prompt)
#             out_put = chain.run(subject=subject,
#                                 current_step=current_step, cot=cot, question=question)
#     return out_put
#
#
# def debate_agent(subject, question, current_step, cot, response, temp=0, model_name='gpt-4-0125-preview'):
#     system_prompt = (
#         '''You are a professional specialized in {subject}. You need to help me verify my steps when I solve the question.
#         I am currently at step #{current_step}.
#
#         Before you perform the task, I want you to keep in mind several definitions for my possible mistakes.
#         1. Factuality： This type of error emphasizes the discrepancy between generated content and verifable real-word facts, including
#         factual inconsistency or fabrication. In mathematics for instance, it may represents the computational error.
#
#         2. Faithfulness: This type of error refers to the divergence of my step analysis from the original question or
#         previous steps, as well as self-consistency within my steps. In mathematics for instance, it may represents that
#         I understood the question wrongly or my proposed step is inconsistent with my previous step.
#
#         Other agents had helped me identify the error I made in the current step. You goal is to debate with the other
#         agents and justify if their corrections were correct based on my question, thought process. Please use Critical
#         Thinking and only capture the significant mistake that will lead to wrong answer. Errors like different interpretation
#         should be ignored.
#         \n{format_instructions}
# ''')
#     human_prompt = ("Here is my complete thought process {cot} and this is the original question {question}. The full"
#                     "response from the other agents were given as {response}")
#
#     response_schemas = [
#
#         ResponseSchema(name="Justification",
#                        description='''
#                         Give me the your response to the other agent and justify
#                          that whether you think the other agents' correction to my step was correct.
#                         '''),
#
#         ResponseSchema(name="Agreement",
#                        description='''
#                        say 'YES' if you agree with the other agents corrections to my current step analysis. Otherwise,
#                        say 'NO'
#                         '''),
#
#         ResponseSchema(name="Correction",
#                        description='''
#                                Help me generate a  version of the current
#                                step that you think is correct
#                                . Notice that do not simply identify the error here, instead
#                                you should directly give me the correct version with calculation (if applicable)
#
#                                'Step n : [Corrected version] *<verified>*'
#                                '''),
#     ]
#     output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
#     success = False
#     while not success:
#         try:
#             worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
#             chain = worker.chain_generator(system_prompt, human_prompt)
#             out_put = chain.run(subject=subject,
#                                 current_step=current_step, cot=cot, question=question, response=response)
#
#             success = True
#         except:
#             worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
#             chain = worker.chain_generator(system_prompt, human_prompt)
#             out_put = chain.run(subject=subject,
#                                 current_step=current_step, cot=cot, question=question, response=response)
#     return out_put
#
#
# def correct_answer_agent_partial_cot(subject, cot,question, temp=0, model_name='gpt-4-0125-preview'):
#     system_prompt = (
#         '''
#         You are a professional specialized in {subject}. Your task is help me answer the question based on my initial
#         thoughts. I will provide you several steps of my attempt. Your task is to CONTINUE my thought process and then
#         answer my question step by step. Also, maximum 12 steps allowed and you can assume my initial thoughts had been
#         checked since could be trusted. Remember, your response should based on my initial thoughts!
#         \n{format_instructions}
#         ''')
#     human_prompt = "Here is my question :{question}. And my initial thought process is given as {cot}"
#
#     response_schemas = [
#         ResponseSchema(name="Complete Thought Process",
#                        description="Continue my thought process in order to answer the question,"
#                                    "You must include my initial thought process as well and leave them as the EXACT TERMS "
#                                    "provided in my initial thoughts!"
#                                    "Return the complete chain of thought by following the format:"
#                                    "Step n: [step process]."),
#         ResponseSchema(name="Final Answer",
#                        description="Give me your final answer based on my thought process , if have options provided, "
#                                    "just give me the option index. Follow"
#                                    "The format [final_answer or correct_option_index]")
#     ]
#     output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
#     success = False
#     while not success:
#         try:
#             worker = ChatModelWorker(output_parser=output_parser, temperature=temp, model=model_name)
#             chain = worker.chain_generator(system_prompt, human_prompt)
#             out_put = chain.run(subject=subject, cot = cot,
#                                 question=question)
#
#             success = True
#         except:
#             print('gpt_error')
#     return out_put
#
# def judge_agent(subject, question, cots, model_name='gpt-4-0125-preview'):
#     # Define the system and human prompts
#     system_prompt = (
#     '''You are a professional specialized in {subject}.
#     A Chain of Thought (COT) is a step-by-step reasoning process used to solve a problem or answer a question.
#     You have been presented with three different COTs below for the question "{question}".
#     Please carefully analyze these COTs and provide your assessment on which one is the most logically sound based on the given information and your expertise in the subject. \n\n{format_instructions}'''
#     )
#
#     human_prompt = (
#     "Here are the three Chains of Thought (COTs) for your analysis: \n\n"
#     "COT 1: {cot1}\n\n"
#     "COT 2: {cot2}\n\n"
#     "COT 3: {cot3}\n\n"
#     )
#
#     # Define response schemas
#     response_schemas = [
#     ResponseSchema(
#     name="Selected COT",
#     description='''Indicates the most logically correct Chain of Thought (COT) selected by the expert. Please provide the index of the most correct COT (1, 2, or 3). If you think more than two CoTs are equally correct, please pick a short chain. If none of the chains of thought make sense, simply output 'None'.''',
#     ),
#     ]
#
#     # Initialize a structured output parser
#     output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
#
#     # Initialize a dictionary to store the judged cots
#     judged_cots = None
#
#     # Iterate over the cots
#     cot1,cot2,cot3 = cots[0],cots[1],cots[2]
#     success = False
#     while not success:
#         try:
#             # Initialize a ChatModelWorker
#             worker = ChatModelWorker(output_parser=output_parser, model=model_name)
#             chain = worker.chain_generator(system_prompt, human_prompt)
#             # Run the chain
#             output = chain.run(subject=subject, cot1=cot1, cot2=cot2, cot3=cot3, question=question)
#             # Store the judged cot
#             judged_cot = output_repraser(output)['Selected COT']
#             success = True
#         except Exception as e:
#             print("Error:", e)
#             continue
#
#     return judged_cot
#
#
#
