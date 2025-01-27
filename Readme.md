# CoT Rerailer
Reproduction repo for the paper "CoT Rerailer: Enhancing the Reliability of Large Language Models in Complex Reasoning Tasks through Error Detection and Correction"

## Conda env configuration
The required packages to run the code are included in [src/environment.yaml](src/environment.yaml)

To configure the environment, run:

`conda env create -f src/environment.yml`

## OpenAI GPT-4 API Key
In this reproduction repo, we did not include our API key since it is confidential.

To obtain a key, please go to the OpenAI official website and obtain an API key. Once you have obtained the API key, open the file [src/api_key.txt](src/api_key.txt) and replace the placeholder with your API key.

## Data
All data is organized under the `data` folder, with subfolders for each category of data:
- Everydataset is located under the name (e.g.`TruthfulQA` are located in the `TruthfulQA` subfolder)
- `MMLU` dataset is obtained from the Hugging Face dataset: https://huggingface.co/datasets/lukaemon/mmlu
- `BigBench` dataset is obtained from the Hugging Face dataset: https://huggingface.co/datasets/maveriq/bigbenchhard

There is also a Jupyter notebook file to preprocess the data. Once the data is processed, it is stored in the `preprocessed` subfolder to be combined and sampled.

We also keep a subfolder called final_test data, which includes the results file after running the first derailing part of the pipeline, so that people are free to test our second rerailer part freely with them.

## Code
All main code is located in the `src` folder:
- `Evaluation.ipynb` contains all the evaluation results we reported in the paper.
- `SOTA_comparison.ipynb` contains all the state-of-the-art comparision results we reported in the paper.
- `prompt_templates` folder defines the prompts for the various agent used in the paper.
- `Parsers.py` contains all the parser for different agents.
- `llm_agent.py` defines llm model configs.
- `derailer.py` contains the code for the first derailing part of our pipeline.
- `rerailer.py` contains the code for the rerailer part of the pipeline.
- `ablation_study_1.py` and `ablation_study_2.py` are used for running the two ablation studies.

To run the code:
1. Ensure you have set up the conda environment and added your OpenAI GPT-4 API key.
2. Preprocess the datasets based on the template.
3. Run `derailer.py` to perform the derailing step of the pipeline.
4. Run `rerailer.py` to perform the rerailer step of the pipeline.
5. (Optional) Run `ablation_study_derailer_only.py` and `ablation_study_rerailer_only.py` to conduct the ablation studies.

## Results
All results are stored in the `result` folder. There are two subfolders:
- `results`: Contains the results for our main experiments.
- `sota_comparison`: Contains the results for state-of-the-art comparison
- `ablation_study`: Contains the results for the ablation studies.

## Checking Results In the Paper
- Within the `src` subfolder, there is a `Evaluation` notebook that contains the results of our reported results. The `Evaluation.ipynb` notebook in this folder summarizes and visualizes all the key results. 

- For State-of-the-Art comparison, please see `src/SOTA_comparison.ipynb`

## Use of Deductive Verification and SelfCheck
- Deductive Verification GitHub link: https://github.com/ritun16/chain-of-verification/tree/main?tab=readme-ov-file
- Self Check GitHub link: https://github.com/lz1oceani/verify_cot/tree/main

