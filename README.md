# HelloBench: Benchmarking Long-Form Text Generation Capabilities of Large Language Models

HelloBench is an open-source benchmark designed to rigorously evaluate the long-text generation capabilities of large language models (LLMs). This repository includes the complete test dataset, evaluation code from the associated paper:

**[HelloBench: Benchmarking Long-Form Text Generation Capabilities of Large Language Models]**

The test datasets are curated from platforms like Quora and Reddit, providing diverse, real-world challenges to evaluate LLM performance. For detailed usage instructions and insights, please refer to the sections below.

## Repository Contents

```
│  LICENSE
│  llm_judge.py
│  README.md
│  regression.py
│  requirements.txt
│  run.py
│
├─Checklists
│      chat_checklist.json
│      heuristic_text_generation_checklist.json
│      open_ended_qa_checklist.json
│      summarization_checklist.json
│      text_completion_checklist.json
│
├─Annotation_Interface
│      main.py
│      stats.jsonl
│
└─HelloBench
    │  chat.jsonl
    │  heuristic_text_generation.jsonl
    │  open_ended_qa.jsonl
    │  summarization.jsonl
    │  text_completion.jsonl
    │
    ├─length_constrained_experiments_data
	     heuristic_text_generation_16k.jsonl
       heuristic_text_generation_2k.jsonl
       heuristic_text_generation_4k.jsonl
       heuristic_text_generation_8k.jsonl
 
```

## Setup Instructions

**Language Requirements**: Python 3.6 or later

To set up the environment, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Usage Guidelines

- **Test Data**: The core test data can be found in the `HelloBench` directory.  
- **Evaluation Checklists**: For predefined evaluation criteria, refer to the `Checklists` directory.  
- **Human Evaluation**: To facilitate human evaluation, the required code is located in the `Annotation_Interface` directory.  
- **LLM Response Generation**: Use `run_=.py` to call an LLM to generate responses for the tasks.  
- **LLM Response Judging**: To invoke LLMs for evaluating responses, run `llm_judge.py`.  
- **Regression**: For Linear Regression, execute `regression.py`.

For additional details and advanced usage, please refer to the code comments and paper.
