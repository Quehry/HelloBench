import json
import os
import time
import nltk
import openai


GPT_KEY = "Put your OpenAI API key here"
SYS_PROMPT = """You are a helpful evaluator. Your task is to evaluate the checklists of the responses given by the Large Language Models (LLMs) based on user instructions. These checklists consist of yes or no questions."""
USER_PROMPT = """Your core task is to evaluate the checklists based on the user's instruction and LLM's response, with each checklist item being a yes or no question indicating a specific aspect that the LLM's response should meet. You need to judge the checklist item based on the instruction and response. The evaluation results are scored from 0 to 1, with 5 scores in total, which are:

0: The response fails to meet the checklist requirements, demonstrating substantial need for improvement across multiple areas.
0.25: The response partially meets some checklist requirements, but significant elements remain unaddressed.
0.5: The response meets several checklist requirements, yet the overall evaluation appears ambiguous or unclear.
0.75: The response aligns with most checklist requirements, though there are still minor areas that could be refined or enhanced.
1: The response fully satisfies all checklist requirements, with no identifiable issues or areas for improvement. It means this response is already perfect; you can't find any significant flaws in it.

Here is the instruction:
{{\"instruction\": {instruction}}}

Here is the response given by LLM:
{{\"response\": {response}}}

Since the response may be rather long, I am specifically reminding you here that the response has ended.

Here are checklists of this instruction:
{{\"checklists\": {checklists}}}

To further remind you, I will repeat my requirements:

Your core task is to evaluate the checklists based on the user's instruction and LLM's response, with each checklist item being a yes or no question indicating a specific aspect that the LLM's response should meet. You need to judge the checklist item based on the instruction and response. The evaluation results are scored from 0 to 1, with 5 scores in total, which are:

0: The response fails to meet the checklist requirements, demonstrating substantial need for improvement across multiple areas.
0.25: The response partially meets some checklist requirements, but significant elements remain unaddressed.
0.5: The response meets several checklist requirements, yet the overall evaluation appears ambiguous or unclear.
0.75: The response aligns with most checklist requirements, though there are still minor areas that could be refined or enhanced.
1: The response fully satisfies all checklist requirements, with no identifiable issues or areas for improvement. It means this response is already perfect; you can't find any significant flaws in it.

Always provide the reason for your evaluation results. You should be strict but fair in your evaluation. A score of 1 means that the response perfectly meets all the checklist requirements and you think there are really no room for improvements. When giving a score of 1, you need to carefully consider whether this checklist has been perfectly satisfied.

Evaluate all the checklists and return the evaluation results of the checklists. Output a Python List consisting of the Python Dictionary formatted as follows:
[{{\"checklist_id\": \"the id of the checklist\", \"reason\": \"The reason for your evaluation results\", \"evaluation_score\": \"Your evaluation score for this checklist\"}},{{\"checklist_id\": \"the id of the checklist\", \"reason\": \"The reason for your evaluation results\", \"evaluation_score\": \"Your evaluation score for this checklist\"}}]

There are total {num_checklist} checklists that you need to evaluate. The length of the output list is equal to the number of checklists and you should give an evaluation score for each checklist. You should be strict to the evaluation to further compare the responses from different models. Your response must be a valid Python List and should contain nothing else, as it will be directly executed in Python."""

LLM_EVAL_SYS_PROMPT = """You are a helpful evaluator. Your task is to evaluate the quality of the responses given by the Large Language Models (LLMs) based on user instructions."""
LLM_EVAL_USER_PROMPT = """Your core task is to evaluate the quality of the response given by LLMs based on the user's instruction. The evaluation results are scored from 0 to 10, which are:

0-1: The response is irrelevant or completely incorrect, failing to address the user's request.
2-3: The response contains mostly incorrect information with a few minor relevant points, lacking coherent connection to the user's instructions.
4-5: The response is partially correct but has significant gaps or misunderstandings, addressing some aspects of the instructions but not fully meeting them.
6-7: The response is mostly correct and addresses the user's instructions adequately, but there are still some minor issues or areas lacking in clarity or detail.
8-9: The response is almost entirely correct and closely aligns with the user's instructions, with only a few minor issues that do not affect the overall quality.
10: The response is completely correct, fully satisfying the user's instructions without any issues.

Here is the instruction:
{{\"instruction\": {instruction}}}

Here is the response given by LLM:
{{\"response\": {response}}}

Since the response may be rather long, I am specifically reminding you here that the response has ended.

To further remind you, I will repeat my requirements:

Your core task is to evaluate the quality of the response given by LLMs based on the user's instruction. The evaluation results are scored from 0 to 10, which are:

0-1: The response is irrelevant or completely incorrect, failing to address the user's request.
2-3: The response contains mostly incorrect information with a few minor relevant points, lacking coherent connection to the user's instructions.
4-5: The response is partially correct but has significant gaps or misunderstandings, addressing some aspects of the instructions but not fully meeting them.
6-7: The response is mostly correct and addresses the user's instructions adequately, but there are still some minor issues or areas lacking in clarity or detail.
8-9: The response is almost entirely correct and closely aligns with the user's instructions, with only a few minor issues that do not affect the overall quality.
10: The response is completely correct, fully satisfying the user's instructions without any issues.

Always provide the reason for your evaluation results. You should be strict but fair in your evaluation.

Evaluate the quality of response and return the evaluation results of the response. Output a Python Dictionary formatted as follows:
{{\"reason\": \"The reason for your evaluation results\", \"evaluation_score\": \"Your evaluation results\"}}

You should be very very very strict to the evaluation to further compare the responses from different models. Your response must be a valid Python Dictionary and should contain nothing else, as it will be directly executed in Python."""
LLM_EVAL_WITH_CHECKLIST_USER_PROMPT = """You are an expert evaluator. Your task is to evaluate the quality of the responses generated by AI models. We will provide you with the user query and an AI-generated response. You should first read the user query and the AI-generated response carefully for analyzing the task, and then evaluate the quality of the responses based on the rules provided below.

Here is the instruction:
{{\"instruction\": {instruction}}}

Here is the response given by LLM:
{{\"response\": {response}}}

Since the response may be rather long, I am specifically reminding you here that the response has ended.

Here are the checklists of this instruction:
{{\"checklists\": {checklists}}}

You should evaluate based on your analysis of the user instruction and AI-generated response. You should first write down your analysis and the checklist that you used for the evaluation, and then provide your evaluation according to the checklist. The scores are in the range of 0~10, where 0 means the response is very poor and 10 means the response is perfect.

Here are more detailed criteria for the scores:
0-1: The response is irrelevant or completely incorrect, failing to address the user's request.
2-3: The response contains mostly incorrect information with a few minor relevant points, lacking coherent connection to the user's instructions.
4-5: The response is partially correct but has significant gaps or misunderstandings, addressing some aspects of the instructions but not fully meeting them.
6-7: The response is mostly correct and addresses the user's instructions adequately, but there are still some minor issues or areas lacking in clarity or detail.
8-9: The response is almost entirely correct and closely aligns with the user's instructions, with only a few minor issues that do not affect the overall quality.
10: The response is completely correct, fully satisfying the user's instructions without any issues.

Always provide the reason for your evaluation results. You should be strict but fair in your evaluation.

Evaluate the quality of response and return the evaluation results of the response. Output a Python Dictionary formatted as follows:
{{\"reason\": \"The reason for your evaluation results\", \"evaluation_score\": \"Your evaluation results\"}}

You should be very very very strict to the evaluation to further compare the responses from different models. Your response must be a valid Python Dictionary and should contain nothing else, as it will be directly executed in Python."""


def gpt4o_ckwise_evaluation(task_type_list, model_name="gpt4o", category=None):
    """
    Evaluate model with the help of gpt4o llm-judge
    :param task_type_list: list of task types to evaluate
    :param model_name: name of the model to be evaluated
    :param category: evaluated category, if not None
    """
    client = openai.Client(api_key=GPT_KEY)

    for task_type in task_type_list:
        # Path to checklists
        ck_path = f"HelloBench/{task_type}.jsonl"

        # Path to model responses
        response_path = f"results/{model_name}/{task_type}_results.jsonl"

        # Output path for results
        output_path = os.path.join("ckwise_results", model_name, f"{task_type}_results.jsonl")

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load checklist data and model response data
        with open(ck_path, "r", encoding="utf-8") as f:
            ck_data_list = [json.loads(line) for line in f.readlines()]
        with open(response_path, "r", encoding="utf-8") as f:
            response_data_list = [json.loads(line) for line in f.readlines()]

        # Create a dictionary mapping IDs to checklist data
        ck_dict = {ck_data["id"]: ck_data for ck_data in ck_data_list}

        # Filter response data based on category, if provided
        tmp_response_data_list = [response for response in response_data_list
                                  if response["id"] in ck_dict and (not category or response["category"] == category)]

        for response_data in tmp_response_data_list:
            id = response_data["id"]
            ck_data = ck_dict[id]  # Get corresponding checklist data
            instruction = response_data["instruction"]
            response = response_data["response"]

            # Tokenize and limit input length
            response_word = nltk.word_tokenize(response)
            response = " ".join(response_word[:16000]) if len(response_word) > 15000 else " ".join(response_word)

            # Prepare checklists for evaluation
            checklists = ck_data["checklists"]
            checklist_list = [{"checklist_id": i, "checklist_content": ck} for i, ck in enumerate(checklists)]

            # Prepare messages for model to evaluation
            messages = [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content":
                    USER_PROMPT.format(instruction=instruction, response=response,
                                       checklists=json.dumps(checklist_list, ensure_ascii=False),
                                       num_checklist=len(checklists))}
            ]

            llm_judge_response_list = []
            for _ in range(3):
                try:
                    completion = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=4096,
                        temperature=0.8,
                        seed=42,
                    )

                    # Clean up the response and parse JSON
                    llm_judge_response = completion.choices[0].message.content
                    llm_judge_response = (llm_judge_response.replace("```json", "").replace("```python", "")
                                          .replace("```", "").replace("\n", "").replace("\\", ""))
                    llm_judge_response_list = json.loads(llm_judge_response)

                    # Ensure the number of checklist items matches the model response
                    assert len(llm_judge_response_list) == len(checklists)
                    break
                except Exception as e:
                    print(e)
                    time.sleep(3)
                    continue

            # Process and store evaluation results
            for llm_judge_response_dict in llm_judge_response_list:
                llm_judge_response_dict["checklist_id"] = int(llm_judge_response_dict["checklist_id"])
                llm_judge_response_dict["evaluation_score"] = float(llm_judge_response_dict["evaluation_score"])
            response_data["checklist_wise_evaluation"] = llm_judge_response_list

            # Save the evaluation results to the output file
            with open(output_path, "a", encoding="utf-8") as f:
                json.dump(response_data, f, ensure_ascii=False)
                f.write("\n")


def gpt4o_ckwise_evaluation_length_constrained_exps(model_name="gpt4o", category=None):
    """
    Evaluate the length-constrained experments by the help of gpt4o llm-judge
    :param model_name: name of the model to be evaluated
    :param category: evaluated category, if not None
    """
    task_type_list = ["heuristic_text_generation"]
    client = openai.Client(api_key=GPT_KEY)

    for task_type in task_type_list:
        for length in ["2k", "4k", "8k", "16k"]:
            # Construct paths for checklist and response data
            ck_path = f"HelloBench/length_constrained_experiments_data/{task_type}_{length}.jsonl"
            response_path = f"results/{model_name}/{task_type}_{length}_results.jsonl"
            output_path = os.path.join("ckwise_results", model_name, f"{task_type}_{length}_results.jsonl")

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Load checklist and response data
            with open(ck_path, "r", encoding="utf-8") as f:
                ck_data_list = [json.loads(line) for line in f.readlines()]
            with open(response_path, "r", encoding="utf-8") as f:
                response_data_list = [json.loads(line) for line in f.readlines()]

            # Map checklist data by ID
            ck_dict = {ck_data["id"]: ck_data for ck_data in ck_data_list}

            # Filter response data by category (if provided) and check if ID exists in checklist
            tmp_response_data_list = [response for response in response_data_list
                                      if
                                      response["id"] in ck_dict and (not category or response["category"] == category)
                                      ]

            # Process each response in the filtered list
            for response_data in tmp_response_data_list:
                id = response_data["id"]
                ck_data = ck_dict[id]  # Get corresponding checklist data
                instruction = response_data["instruction"]
                response = response_data["response"]

                # Tokenize and limit input length to 16k words
                response_word = nltk.word_tokenize(response)
                response = " ".join(response_word[:16000]) if len(response_word) > 15000 else " ".join(response_word)

                # Prepare checklists for evaluation
                checklists = ck_data["checklists"]
                checklist_list = [{"checklist_id": i, "checklist_content": ck} for i, ck in enumerate(checklists)]

                # Prepare messages for GPT-4o evaluation
                messages = [
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": USER_PROMPT.format(instruction=instruction, response=response,
                                                    checklists=json.dumps(checklist_list, ensure_ascii=False),
                                                    num_checklist=len(checklists))}
                ]
                llm_judge_response_list = []
                # Retry model completion up to 3 times if an error occurs
                for _ in range(3):
                    try:
                        completion = client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            max_tokens=4096,
                            temperature=0.8,
                            seed=42,
                        )

                        # Clean and parse model response
                        llm_judge_response = completion.choices[0].message.content
                        llm_judge_response = llm_judge_response.replace("```json", "").replace("```python", "").replace(
                            "```", "").replace("\n", "").replace("\\", "")
                        llm_judge_response_list = json.loads(llm_judge_response)

                        # Ensure the number of checklist items matches the model output
                        assert len(llm_judge_response_list) == len(checklists)
                        break
                    except Exception as e:
                        print(e)
                        time.sleep(3)  # Wait before retrying
                        continue

                # Process model response and add evaluation score
                for llm_judge_response_dict in llm_judge_response_list:
                    llm_judge_response_dict["checklist_id"] = int(llm_judge_response_dict["checklist_id"])
                    llm_judge_response_dict["evaluation_score"] = float(llm_judge_response_dict["evaluation_score"])
                response_data["checklist_wise_evaluation"] = llm_judge_response_list

                # Append the results to the output file
                with open(output_path, "a", encoding="utf-8") as f:
                    json.dump(response_data, f, ensure_ascii=False)
                    f.write("\n")


def llm_eval(model_name_list):
    """
    Evaluate LLMs by using LLM-Eval
    :param model_name_list: list of model names to evaluate
    """
    task_type = "summarization"
    client = openai.Client(api_key=GPT_KEY)

    for model_name in model_name_list:
        # Construct paths for checklist and response data
        response_path = f"results/{model_name}/{task_type}_results.jsonl"
        ck_path = f"HelloBench/{task_type}.jsonl"
        output_path = os.path.join("llm_eval_results", model_name, f"{task_type}_results.jsonl")

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load checklist and response data
        with open(ck_path, "r", encoding="utf-8") as f:
            ck_data_list = [json.loads(line) for line in f.readlines()]
        with open(response_path, "r", encoding="utf-8") as f:
            response_data_list = [json.loads(line) for line in f.readlines()]

        # Map checklist data by ID
        ck_dict = {ck_data["id"]: ck_data for ck_data in ck_data_list}

        # Filter response data by category (if provided) or ID presence in checklist
        tmp_response_data_list = [
            response for response in response_data_list
            if response["id"] in ck_dict and (not category or response["category"] == category)
        ]

        # Process each response in the filtered list
        for response_data in tmp_response_data_list:
            id = response_data["id"]
            instruction = response_data["instruction"]
            response = response_data["response"]

            # Tokenize and limit input length to 16k words
            response_word = nltk.word_tokenize(response)
            response = " ".join(response_word[:16000]) if len(response_word) > 15000 else " ".join(response_word)

            # Prepare messages for LLM evaluation
            messages = [
                {"role": "system", "content": LLM_EVAL_SYS_PROMPT},
                {"role": "user", "content": LLM_EVAL_USER_PROMPT.format(instruction=instruction, response=response)}
            ]

            llm_judge_response_dict = {"reason": "", "evaluation_score": 0}
            # Retry model completion up to 3 times if an error occurs
            for _ in range(3):
                try:
                    completion = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=4096,
                        temperature=0.8,
                        seed=42,
                    )

                    # Clean and parse model response
                    llm_judge_response = completion.choices[0].message.content
                    llm_judge_response = llm_judge_response.replace("```json", "").replace("```python", "").replace(
                        "```", "").replace("\n", "").replace("\\", "")
                    llm_judge_response_dict = json.loads(llm_judge_response)

                    # Ensure the response contains valid keys and values
                    assert llm_judge_response_dict.keys() == {"reason", "evaluation_score"}
                    llm_judge_response_dict["evaluation_score"] = float(llm_judge_response_dict["evaluation_score"])
                    assert 0 <= llm_judge_response_dict["evaluation_score"] <= 10
                    break
                except Exception as e:
                    print(e)
                    time.sleep(1)  # Wait before retrying
                    continue

            # Add evaluation results to response data
            response_data["llm_eval_evaluation"] = llm_judge_response_dict

            # Append the results to the output file
            with open(output_path, "a", encoding="utf-8") as f:
                json.dump(response_data, f, ensure_ascii=False)
                f.write("\n")


def llm_eval_with_checklist_results(model_name_list):
    """
    Evaluate LLMs with checklists using LLM-Eval-with-checklists
    :param model_name_list: list of model names to evaluate
    """
    task_type = "summarization"
    client = openai.Client(api_key=GPT_KEY)

    # Loop through each model in the list
    for model_name in model_name_list:
        # Define paths for checklist, response, and output data
        response_path = f"results/{model_name}/{task_type}_results.jsonl"
        ck_path = f"HelloBench/{task_type}.jsonl"
        output_path = os.path.join("llm_eval_with_checklists_results", model_name, f"{task_type}_results.jsonl")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load checklist and response data
        with open(ck_path, "r", encoding="utf-8") as f:
            ck_data_list = [json.loads(line) for line in f.readlines()]
        with open(response_path, "r", encoding="utf-8") as f:
            response_data_list = [json.loads(line) for line in f.readlines()]

        # Map checklist data by ID
        ck_dict = {ck_data["id"]: ck_data for ck_data in ck_data_list}

        # Filter response data by category (if provided) or checklist ID presence
        tmp_response_data_list = [
            response for response in response_data_list
            if response["id"] in ck_dict and (not category or response["category"] == category)
        ]

        # Process each filtered response
        for response_data in tmp_response_data_list:
            id = response_data["id"]
            ck_data_dict = ck_dict[id]  # Get corresponding checklist data
            instruction = response_data["instruction"]
            response = response_data["response"]

            # Tokenize response and limit input length to 16k words
            response_word = nltk.word_tokenize(response)
            response = " ".join(response_word[:16000]) if len(response_word) > 15000 else " ".join(response_word)

            # Prepare messages for LLM evaluation with checklists
            checklists = ck_data_dict["checklists"]
            messages = [
                {
                    "role": "user", "content": LLM_EVAL_WITH_CHECKLIST_USER_PROMPT.format(
                    instruction=instruction,
                    response=response,
                    checklists=json.dumps(checklists, ensure_ascii=False))
                }
            ]

            llm_judge_response_dict = {"reason": "", "evaluation_score": 0}
            # Retry up to 3 times if there's an error
            for _ in range(3):
                try:
                    completion = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=4096,
                        temperature=0.8,
                        seed=42,
                    )

                    # Clean and parse the model response
                    llm_judge_response = completion.choices[0].message.content
                    llm_judge_response = llm_judge_response.replace("```json", "").replace("```python", "").replace("```", "").replace("\n", "").replace("\\", "")
                    llm_judge_response_dict = json.loads(llm_judge_response)

                    # Ensure response contains valid keys and values
                    assert llm_judge_response_dict.keys() == {"reason", "evaluation_score"}
                    llm_judge_response_dict["evaluation_score"] = float(llm_judge_response_dict["evaluation_score"])
                    assert 0 <= llm_judge_response_dict["evaluation_score"] <= 10
                    break
                except Exception as e:
                    print(e)
                    time.sleep(3)  # Wait before retrying
                    continue

            # Store the evaluation results
            response_data["llm_eval_with_checklists_evaluation"] = llm_judge_response_dict

            # Append results to output file
            with open(output_path, "a", encoding="utf-8") as f:
                json.dump(response_data, f, ensure_ascii=False)
                f.write("\n")


if __name__ == "__main__":
    # select your task from this list, and keep them
    task_type_list = ["open_ended_qa", "summarization", "chat", "text_completion", "heuristic_text_generation"]
    # put your models here, it may be:
    # - gpt4o
    # - qwen2_7b
    # - qwen2_72b
    # - llama31_8b
    # - llama31_70b
    # - mistral_large_instruct
    # - deepseek_chat_api
    # - gemma2_27b_it
    # - glm_4_9b_chat
    # - internlm2_5_20b_chat
    # - glm_4_9b_chat_1m
    # - yi_1_5_34b_chat
    # - yi_1_5_34b_chat_16k
    # - map_neo
    # - phi_35_moe_instruct
    # - gpt4o_mini
    # - gpt4o_0806
    # - o1_mini
    # - gpt4_turbo
    # - claude35
    # - qwen_max
    # - yi_large
    # - glm_4_api
    # - doubao_pro_32k
    # - suri
    # - mistral_7b_instruct_v0_2
    # - longwriter
    # - internlm2_5_7b_chat
    # - internlm2_5_7b_chat_1m
    # - gemini_1_5_pro
    model_name = "o1_mini"
    category = None
    # Uncomment the following lines to evaluate the models
    # gpt4o_ckwise_evaluation(task_type_list, model_name, category)
    # gpt4o_ckwise_evaluation_length_constrained_exps(model_name, category)
    model_name_list = ["mistral_large_api"]
    # Uncomment the following lines to evaluate the models
    # llm_eval(model_name_list=model_name_list)
    # llm_eval_with_checklist_results(model_name_list=model_name_list)