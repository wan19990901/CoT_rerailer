from langchain_core.pydantic_v1 import BaseModel, Field

# Define your desired data structure.
class CoT_Agent(BaseModel):
    Chain_of_Thought: str = Field(...,description='''
        Provide step by step analysis. For instance, you should follow the pattern 'step 1:... \nstep 2:...
        ''')
    Number_of_Steps_Proposed: str = Field(...,description='''return a simple integer output which indicates the 
    number of steps you proposed in the Chain of Thought''')
    Final_Answer: str = Field(...,description='''Give me your final answer, if have options provided, just give me 
    the option index. Follow The format [final_answer or correct_option_index]''')

class Root_Checker_Agent(BaseModel):
    Verification: str = Field(...,description='''
        Provide step by step analysis. For instance, you should follow the pattern 'step 1:... \nstep 2:...
        ''')
    Step_Hallucination: str = Field(...,description='''return a simple integer output which indicates the number of 
    steps you proposed in the Chain of Thought''')
    Correction: str = Field(...,description='''Give me your final answer, if have options provided, just give me the 
    option index. Follow The format [final_answer or correct_option_index]''')
    Dependency: str = Field(...,
                            description='''Give me your final answer, if have options provided, just give me the 
                            option index. Follow The format [final_answer or correct_option_index]''')

class Debate_Agent(BaseModel):
    Justification: str = Field(...,description=''' Give me the your response to the other agent and justify
                         that whether you think the other agents' correction to my step was correct.''')
    Agreement: str = Field(...,description='''say 'YES' if you agree with the other agents corrections to my current 
    step analysis. Otherwise, say 'NO''')
    Correction: str = Field(...,description='''
        Help me generate a  version of the current
                               step that you think is correct
                               . Notice that do not simply identify the error here, instead
                               you should directly give me the correct version with calculation (if applicable)

                               'Step n : [Corrected version] *<verified>*''')
class Correct_Answer_Agent_Partial_CoT(BaseModel):
    Complete_Thought_Process: str = Field(...,description='''Continue my thought process in order to answer the 
    question, provided in my initial thoughts! Return the complete chain of thought by following the format: Step n: 
    [step process]''')
    Final_Answer: str = Field(...,description='''Give me your final answer based on my thought process , if have 
    options provided, just give me the option index. Follow The format [final_answer or correct_option_index]''')

class Judge(BaseModel):
    Selected_COT: str = Field(...,description='''Indicates the most logically correct Chain of Thought (COT) selected 
    by the expert. Please provide the index of the most correct COT (1, 2, or 3). If you think more than two CoTs are 
    equally correct, please pick a short chain. If none of the chains of thought make sense, simply output 'None'. ''')
