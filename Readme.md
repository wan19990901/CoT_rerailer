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
- `GSM8K` and `MATHQA` datasets are located in the `Selfcheck` subfolder, as they were obtained from the SelfCheck repo (https://github.com/NingMiao/SelfCheck/tree/master/data).
- `MMLU` dataset is obtained from the Hugging Face dataset: https://huggingface.co/datasets/lukaemon/mmlu
- `BigBench` dataset is obtained from the Hugging Face dataset: https://huggingface.co/datasets/maveriq/bigbenchhard

There is also a Jupyter notebook file to preprocess the data. Once the data is processed, it is stored in the `preprocessed` subfolder to be combined and sampled.

We also keep a subfolder called final_test data, which includes the results file after running the first derailing part of the pipeline, so that people are free to test our second rerailer part freely with them.

## Code
All main code is located in the `src` folder:
- `llm_agent.py` defines the prompts for the various agents used in the paper.
- `filter.py` contains the code for the first derailing part of our pipeline.
- `rerailer.py` contains the code for the rerailer part of the pipeline.
- `ablation_study_debate.py` and `ablation_study_filter.py` are used for running the two ablation studies.

To run the code:
1. Ensure you have set up the conda environment and added your OpenAI GPT-4 API key.
2. Run the preprocessing Jupyter notebook in the `data` folder to preprocess the datasets.
3. Run `filter.py` to perform the derailing step of the pipeline.
4. Run `rerailer.py` to perform the rerailer step of the pipeline.
5. (Optional) Run `ablation_study_debate.py` and `ablation_study_filter.py` to conduct the ablation studies.

## Results
All results are stored in the `result` folder. There are two subfolders:
- `main_results`: Contains the results for our main experiments.
- `ablation_study`: Contains the results for the ablation studies.

Within the `main_results` subfolder, there is a `performance` subsubfolder that contains the results of our best model (rerailing with multi-step). The `performance.ipynb` notebook in this folder summarizes and visualizes all the key results.

## References
- MathQA dataset: https://github.com/NingMiao/SelfCheck/tree/master/data/MathQA
- GSM8K dataset: https://github.com/NingMiao/SelfCheck/tree/master/data/GSM8K
- MMLU dataset: https://huggingface.co/datasets/lukaemon/mmlu
- BigBench dataset: https://huggingface.co/datasets/maveriq/bigbenchhard
