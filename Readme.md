# CoT Rerailer
Reproduction repo for the paper "CoT Rerailer: Enhancing the Reliability of Large Language
Models in Complex Reasoning Tasks through Error Detection
and Correction"


## Conda env configuration
The required packages to run the code are included in [src/environment.yaml](src/environment.yaml)

To configure the env, run

`conda env create -f src/environment.yml`

## OPENAI GPT-4 API KEY
In this reproduction repo, we did not include our API key since they are confidential.

To Obtain a key, please go to openai official website and obtain an api key.
Once the apikey is obtained, please open the file [src/api_key.txt](src/api_key.txt) and replace your API key to the 
placeholder

## Reproduction
To run the full pipeline, run:

`python src/`

To test Deviation Identifier, run:

`python src/`

To test Rerailer with multiple steps, run:

`python src/rerailer_multisteps.py`

To test Rerailer with one steps, run:

`python src/rerailer_onestep.py`



## Try your own example
If you want to try your own question, please follow the steps:

1. Open [data/question_template.json](data/question_template.json).
2. Enter your Question, Options, Answer, and Category in the provided field and save.
3. Run `python src/pipeline.py --TEST True`



## Data
* Preprocessed QA dataset are stored
in [data/preprocessed](data/preprocessed)
* The code scripts can be found in [src/](src/)
* The results are stored in [result/](result/)
  * [result/annotated_text](result/annotated_text) contains new causability report if run the reproduction code  sult/Expert_Eval) contains pre-trained model check-points and expert evaluation results


## References
* MathQA dataset: 
* GSM8K dataset:
* MMLU dataset:
* BigBench dataset: