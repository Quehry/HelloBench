# HelloBench: Evaluating Long Text Generation Capabilities of Large Language Models

HelloBench is an open-source benchmark designed to evaluate the long text generation capabilities of large language models (LLMs). This repository includes the complete test data and evaluation code from the associated paper:

**[HelloBench: Evaluating Long Text Generation Capabilities of Large Language Models](https://arxiv.org/abs/2409.16191)**

The test data are curated from platforms like Quora and Reddit, providing diverse, real-world challenges to evaluate LLM performance.

## Merge to Opencompass

HelloBench is now merged to [Opencompass](https://github.com/open-compass/opencompass), you can launch the test script automatically in Opencompass, more details can be found in [url](https://github.com/open-compass/opencompass/tree/main/opencompass/configs/datasets/subjective/hellobench)


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

**Language Requirements**: Python 3.10 or later

To set up the environment, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Usage Guidelines

- **Test Data**: The core test data can be found in the `HelloBench` directory.  
- **Evaluation Checklists**: For predefined evaluation criteria, refer to the `Checklists` directory.  
- **Human Evaluation**: To facilitate human evaluation, the required code is located in the `Annotation_Interface` directory.  
- **LLM Response Generation**: Use `run.py` to call an LLM to generate responses for the tasks.  
- **LLM Response Judging**: To invoke LLMs for evaluating responses, run `llm_judge.py`.  
- **Regression**: For Linear Regression, execute `regression.py`.

For additional details and advanced usage, please refer to the code comments and paper.

## Generation Configuration
Here are the generation configurations for different models.

| Model Name | max_new_tokens | temperature | version |
|-------|-------|-------|--------|
| GPT-4o-2024-0806 | 16384 | 0.8 | gpt-4o-2024-08-06 |
| Mistral-Large-API | 16384 | 0.8 | mistral-large-latest |
| o1-Mini | 32768[^1] | 0.8 | o1-mini |
| Claude-3.5-Sonnet | 8192[^2] | 0.8 | claude-3-5-sonnet-20240620 |
| Gemini-1.5-Pro | 8192[^3]| 0.8 | gemini-1.5-pro |
| Deepseek-API | 4096[^4] | 0.8 | deepseek-chat |
| Yi-Large | 16384 | 0.8 | yi-large |
| Qwen-Max | 2000[^5] | 0.8 | qwen-max-0428 |
| GLM-4-API | 4096[^6] | 0.8 | glm-4-0520 |
| Gemma-2-27B | 4096[^7] | 0.8 | google/gemma-2-27b-it |
| LLaMA-3.1-70B | 16384 | 0.8 | meta-llama/Meta-Llama-3.1-70B-Instruct |
| Qwen-2-72B | 16384 | 0.8 | Qwen/Qwen2-72B-Instruct |
| InternLM-2.5-20B | 16384 | 0.8 | internlm/internlm2_5-20b-chat |
| Yi-1.5-34B | 2048[^8] | 0.8 | 01-ai/Yi-1.5-34B-Chat |
| LLaMA-3.1-8B | 16384 | 0.8 | meta-llama/Meta-Llama-3.1-8B-Instruct |
| GLM-4-9B | 16384 | 0.8 | THUDM/glm-4-9b-chat |
| Qwen-2-7B | 16384 | 0.8 | Qwen/Qwen2-7B-Instruct |
| InternLM-2.5-7B | 16384 | 0.8 | internlm/internlm2_5-7b-chat |
| Mistral-7B-0.2 | 16384 | 0.8 | mistralai/Mistral-7B-Instruct-v0.2 |
| Phi-3.5-Moe | 16384 | 0.8 | microsoft/Phi-3.5-MoE-instruct |
| MAP-Neo | 2048[^9] | 0.8 | m-a-p/neo_7b_instruct_v0.1 |
| LongWriter-GLM4-9B | 16384 | 0.8 | THUDM/LongWriter-glm4-9b |
| Suri-I-ORPO | 16384 | 0.8 | chtmp223/suri-i-orpo |
| Yi-1.5-34B-16K | 8192[^10] | 0.8 | 01-ai/Yi-1.5-34B-Chat-16K |
| InternLM-2.5-7B-1M | 16384 | 0.8 | internlm/internlm2_5-7b-chat-1m |
| GLM-4-9B-1M | 16384 | 0.8 | THUDM/glm-4-9b-chat-1m |

[^1]: For the o1-mini model, the parameter here should be max_completion_tokens instead of max_new_tokens, because it includes reasoning tokens. Therefore, I set it to 32768. You can refer to https://platform.openai.com/docs/guides/reasoning#controlling-costs for more details. 
[^2]: For the reason that claude-3.5-sonnet has max output 8192 tokens. You can refer to https://docs.anthropic.com/en/docs/about-claude/models#model-comparison-table for more details.
[^3]: For the reason that gemini-1.5-pro has max output 8192 tokens. You can refer to https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-pro for more details.
[^4]: For the reason that deepseek-chat has max output 4096 tokens. You can refer to https://api-docs.deepseek.com/zh-cn/quick_start/pricing for more details.
[^5]: For the reason that qwen-max-0428 has max output 2000 tokens. You can refer to https://help.aliyun.com/zh/model-studio/getting-started/models?spm=a2c4g.11186623.0.0.74b04823IseC0N#9f8890ce29g5u for more details.
[^6]: For the reason that glm-4-0520 has max output 4096 tokens. You can refer to https://bigmodel.cn/dev/howuse/model for more details.
[^7]: Gemma-2-27B has max_position_embeddings with 8192 tokens, and you need to balance the input and output, thus result the selection of max_new_tokens=4096.
[^8]: Yi-1.5-34B-Chat has max_position_embeddings with 4096 tokens, and you need to balance the input and output, thus result the selection of max_new_tokens=2048.
[^9]: neo_7b_instruct_v0.1 has max_position_embeddings with 8192 tokens, and you need to balance the input and output, thus result the selection of max_new_tokens=2048.
[^10]: Yi-1.5-34B-Chat-16K has max_position_embeddings with 16384 tokens, and you need to balance the input and output, thus result the selection of max_new_tokens=8192.