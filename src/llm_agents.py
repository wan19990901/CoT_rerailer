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


def cot_agent(subject, question, temp=0, model_name='gpt-4-0125-preview'):
    system_prompt = (
        "You are a professional specialized in {subject}. You need to help me answer the given question."
        "Notice that you need to solve the question step by step. Do not jump to the answer directly."
        "Your intermediate steps and thoughts are critical!. Also, maximum 10 steps allowed"
        "\n{format_instructions}")
    human_prompt = "{question}"

    response_schemas = [
        ResponseSchema(name="Chain of Thought",
                       description="Provide step by step analysis. For instance, you should follow the pattern 'step 1:... \nstep 2:...'"),
        ResponseSchema(name="Number of Steps Proposed",
                       description="return a simple integer output which indicates the number of steps you proposed in the Chain of Thought"),
        ResponseSchema(name="Final Answer",
                       description="Give me your final answer, if have options provided, just give me the option")
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


def ngram_checker_agent(subject, question, current_step, cot, final_answer, temp=0, model_name='gpt-4-0125-preview'):
    system_prompt = (
        '''You are a professional specialized in {subject}. You need to help me verify my steps when I solve the question.
        Your goal is help me first verify that given the final answer as {final_answer} <Notice that the answer can be wrong hence you need to use it 
        with caucious> and I am currently at step #{current_step}.

        Is the current step logically or computationally correct ?. Notice that correctness means the logic holds or based on fact. 

        If it is correct, then verify that if my current step make the previous step hold. In other words, 
        check the logic consistency of step n in conditional of <step k to step n-1> where n is my current step and k is the first step number in the
        provided thourght process. In the other words, is my current step supported by previous n-1 steps? It is important that for each analysis, ignore 
        steps other than the current step and the previous steps! In addition, for your reference, the question is given as {question}. 

        At step 1, since we have no step 0, instead the correctness and consistency should reflect if I correctly understood the answer.
        \n{format_instructions}
''')
    human_prompt = "Here is my complete thought process {cot}"

    response_schemas = [
        ResponseSchema(name="Step",
                       description='''
                       Just say what step we are current at and which steps we are comparing to {current_step}
                       '''),
        ResponseSchema(name="Verification",
                       description='''
                        Help me verify the correctness and the logic consistency  of the current step, 
                       and tell me the reason. REASON is important. !!! If at Step 1, since 
                        we have no step 0, verify if I correctly understood the answer
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
