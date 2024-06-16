# Overall Pipeline

## Uncertainty Estimation (check hallucination via self-consistency):
the origin of LLM hallucinations, is inherently tied to the model’s uncertainty. Therefore, by estimating the uncertainty of the factuat content generated by the model, it becomes feasible to detect hallucinations (https://arxiv.org/pdf/2311.05232.pdf)

Single LLM Multiple Generation:
1: Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language
models; Direct Asking （https://github.com/potsawee/selfcheckgpt)


Hallucination Detection:

### factual detection
https://arxiv.org/abs/2305.13281
https://arxiv.org/pdf/2305.15852.pdf

sampling based, via self-consistency or self-contradictiton.

2 : Do language models know when they’re hallucinating references? Indirect Asking
3: SelfCheck: Using LLMs to Zero-Shot Check Their Own Step-by-Step Reasoning (reasoning) (https://github.com/NingMiao/SelfCheck)
4:  Chatgpt as a factual inconsistency evaluator for text summarization
Multi-Agent Approach:
1: LM vs LM: detecting factual errors via cross examination. 

No external knowledge or finetuning:

Miao et al. (2023) concentrates on error detection in complex reasoning by employing SelfCheck, a step-by-step
checker that evaluates each reasoning step
within LLMs. The system aggregates confidence scores through a streamlined process of
target extraction, information collection, step
regeneration, and result comparison, thereby
enhancing question-answering accuracy (https://arxiv.org/pdf/2308.00436.pdf)

Large Language Models are Better Reasoners with Self-Verification 
Specifically, in Forward Reasoning, LLM reasoners generate candidate answers using CoT, and the
question and candidate answers form different conclusions to be verified. And in Backward Verification, We mask the original condition and predict
its result using another CoT. We rank candidate
conclusions based on a verification score, which
is calculated by assessing the consistency between
the predicted and original condition values. （https://arxiv.org/pdf/2212.09561.pdf）

RAG:

Knowledge-Driven CoT: Exploring Faithful Reasoning in LLMs for
Knowledge-intensive Question Answering
we formulate the CoT rationale process of LLMs into a structured multi-round QA format. In each round, LLMs interact with a QA system that retrieves external knowledge and
produce faithful reasoning traces based on retrieved precise
answers. The structured CoT reasoning of LLMs is facilitated by our developed KBQA CoT collection, which serves
as in-context learning demonstrations and can also be utilized as feedback augmentation to train a robust retriever

Multiagent:
Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate 


https://arxiv.org/abs/2305.19118







### Note:
GroundTruth Label is manual annotated or use some other approximation. To simplify the work, Should we aviod open-end questions?

## 
In a nutshell, we first adapt our pipeline to some complex questions (Math Reasoning/Generative Question Answering/Multiple Choices/text summarization);    


We proposed a hallucination detector to give scores and rank the hallucination.
We proposed a method to explain the most hallcinated output and compared it with the least hallucinated output.


(A implies B implies C != C implies B implies A)


## Paper Notes







