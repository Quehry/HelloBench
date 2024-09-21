import json
import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
import openai
import time
from peft import PeftModel, PeftConfig


GPT_KEY = "Put your OpenAI API key here"


def run_llama31_8b(task_type_list):
    """
    Run Meta-Llama-3.1-8B-Instruct on HelloBench
    :param task_type_list: list of task types
    """
    pipeline = transformers.pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    for task_type in task_type_list:
        load_path = "HelloBench/" + task_type + ".jsonl"
        output_path = os.path.join("results", "llama31_8b", task_type + "_results.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(load_path, "r", encoding="utf-8") as f:
            data_list = [json.loads(line) for line in f]

        for data_dict in data_list:
            instruction = data_dict["chat_prompt"]
            messages = [{"role": "user", "content": instruction}]
            response = pipeline(messages, max_new_tokens=16384, temperature=0.8)[0]["generated_text"][-1]["content"]

            output_dict = {"id": data_dict["id"], "instruction": instruction, "response": response}
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


def run_llama31_70b(task_type_list):
    """
    Run Meta-Llama-3.1-70B-Instruct on HelloBench
    :param task_type_list: list of task types
    """
    pipeline = transformers.pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    for task_type in task_type_list:
        load_path = "HelloBench/" + task_type + ".jsonl"
        output_path = os.path.join("results", "llama31_70b", task_type + "_results.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(load_path, "r", encoding="utf-8") as f:
            data_list = [json.loads(line) for line in f]

        for data_dict in data_list:
            chat_prompt = data_dict["chat_prompt"]
            messages = [{"role": "user", "content": chat_prompt}]
            response = pipeline(messages, max_new_tokens=16384, temperature=0.8)[0]["generated_text"][-1]["content"]

            output_dict = {"id": data_dict["id"], "instruction": chat_prompt, "response": response}
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


def run_qwen2_7b(task_type_list):
    """
    Run Qwen2-7B-Instruct on HelloBench
    :param task_type_list: list of task types
    """
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

    for task_type in task_type_list:
        load_path = "HelloBench/" + task_type + ".jsonl"
        output_path = os.path.join("results", "qwen2_7b", task_type + "_results.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(load_path, "r", encoding="utf-8") as f:
            data_list = [json.loads(line) for line in f.readlines()]

        for data_dict in data_list:
            chat_prompt = data_dict["chat_prompt"]
            messages = [
                {"role": "user", "content": chat_prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(device)
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=16384,
                temperature=0.8
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            output_dict = {"id": data_dict["id"], "instruction": chat_prompt, "response": response}
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


def run_gpt4o_mini(task_type_list):
    """
    Run GPT-4o-mini on HelloBench
    :param task_type_list: list of task types
    """
    client = openai.Client(api_key=GPT_KEY)

    for task_type in task_type_list:
        load_path = "HelloBench" + task_type + ".jsonl"
        output_path = os.path.join("results", "gpt4o_mini", task_type + "_results.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(load_path, "r", encoding="utf-8") as f:
            data_list = [json.loads(line) for line in f.readlines()]

        for data_dict in data_list:
            chat_prompt = data_dict["chat_prompt"]
            messages = [
                {"role": "user", "content": chat_prompt}
            ]
            response = ""
            for _ in range(3):
                try:
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        max_tokens=16384,
                        temperature=0.8
                    )
                    response = completion.choices[0].message.content
                    break
                except Exception as e:
                    print(e)
                    time.sleep(3)
                    continue

            output_dict = {"id": data_dict["id"], "instruction": chat_prompt, "response": response}
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


def run_gpt4o(task_type_list):
    """
    Run GPT-4o on HelloBench
    :param task_type_list: list of task types
    """
    client = openai.Client(api_key=GPT_KEY)

    for task_type in task_type_list:
        load_path = "HelloBench/" + task_type + ".jsonl"
        output_path = os.path.join("results", "gpt4o", task_type + "_results.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(load_path, "r", encoding="utf-8") as f:
            data_list = [json.loads(line) for line in f.readlines()]

        for data_dict in data_list:
            chat_prompt = data_dict["chat_prompt"]
            messages = [
                {"role": "user", "content": chat_prompt}
            ]
            response = ""
            for _ in range(3):
                try:
                    completion = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        max_tokens=4096,
                        temperature=0.8
                    )
                    response = completion.choices[0].message.content
                    break
                except Exception as e:
                    print(e)
                    time.sleep(3)
                    continue

            output_dict = {"id": data_dict["id"], "instruction": chat_prompt, "response": response}
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


def run_gpt4_turbo(task_type_list):
    """
    Run GPT-4-turbo on HelloBench
    :param task_type_list: list of task types
    """
    client = openai.Client(api_key=GPT_KEY)

    for task_type in task_type_list:
        load_path = "HelloBench/" + task_type + ".jsonl"
        output_path = os.path.join("results", "gpt4_turbo", task_type + "_results.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(load_path, "r", encoding="utf-8") as f:
            data_list = [json.loads(line) for line in f.readlines()]

        for data_dict in data_list:
            chat_prompt = data_dict["chat_prompt"]
            messages = [
                {"role": "user", "content": chat_prompt}
            ]
            response = ""
            for _ in range(3):
                try:
                    completion = client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=messages,
                        max_tokens=4096,
                        temperature=0.8
                    )
                    response = completion.choices[0].message.content
                    break
                except Exception as e:
                    print(e)
                    time.sleep(3)
                    continue

            output_dict = {"id": data_dict["id"], "instruction": chat_prompt, "response": response}
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


def run_gemma2_27b(task_type_list):
    """
    Run GEMMA-2-27B on HelloBench
    :param task_type_list: list of task types
    """
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-27b-it",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    for task_type in task_type_list:
        load_path = "HelloBench/" + task_type + ".jsonl"
        output_path = os.path.join("results", "gemma2_27b_it", task_type + "_results.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(load_path, "r", encoding="utf-8") as f:
            data_list = [json.loads(line) for line in f.readlines()]

        for data_dict in data_list:
            chat_prompt = data_dict["base_prompt"]
            input_ids = tokenizer(chat_prompt, return_tensors="pt").to("cuda")
            input_id_len = input_ids.input_ids.shape[-1]
            if input_id_len > 6000:
                continue

            outputs = model.generate(**input_ids, max_new_tokens=4096, temperature=0.8, do_sample=True)
            response = tokenizer.decode(outputs[0][input_id_len:], skip_special_tokens=True)

            output_dict = {"id": data_dict["id"], "instruction": chat_prompt, "response": response}
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


def run_map_neo(task_type_list):
    """
    Run M-A-P/neo-7b-instruct on HelloBench
    :param task_type_list: list of task types
    """
    model_path = "m-a-p/neo_7b_instruct_v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype='auto'
    ).eval()

    for task_type in task_type_list:
        load_path = "HelloBench/" + task_type + ".jsonl"
        output_path = os.path.join("results", "map_neo", task_type + "_results.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(load_path, "r", encoding="utf-8") as f:
            data_list = [json.loads(line) for line in f.readlines()]

        for data_dict in data_list:
            chat_prompt = data_dict["chat_prompt"]
            messages = [{"role": "user", "content": chat_prompt}]
            input_ids = tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True,
                                                      return_tensors="pt").to("cuda")
            input_id_len = input_ids.shape[-1]
            if input_id_len > 6500:
                continue
            output_ids = model.generate(input_ids, max_new_tokens=2048, temperature=0.8, do_sample=True)
            response = tokenizer.decode(output_ids[0][input_id_len:], skip_special_tokens=True)

            output_dict = {"id": data_dict["id"], "instruction": chat_prompt, "response": response}
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


def run_phi_35(task_type_list):
    """
    Run Phi-3.5-MoE-instruct on HelloBench
    :param task_type_list: list of task types
    """
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-MoE-instruct",
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-MoE-instruct")

    for task_type in task_type_list:
        load_path = "HelloBench/" + task_type + ".jsonl"
        output_path = os.path.join("results", "phi_35_moe_instruct", task_type + "_results.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(load_path, "r", encoding="utf-8") as f:
            data_list = [json.loads(line) for line in f.readlines()]

        for data_dict in data_list:
            chat_prompt = data_dict["chat_prompt"]
            messages = [
                {"role": "user", "content": chat_prompt}
            ]
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            generation_args = {
                "max_new_tokens": 16384,
                "temperature": 0.8,
                "do_sample": True,
            }
            output = pipe(messages, **generation_args)
            response = output[0]["generated_text"][-1]["content"]

            output_dict = {"id": data_dict["id"], "instruction": chat_prompt, "response": response}
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


def run_suri(task_type_list):
    """
    Run Suri on HelloBench
    :param task_type_list: list of task types
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    model_name = "cache_models/suri"
    base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    config = PeftConfig.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
    model = PeftModel.from_pretrained(base_model, model_name, device_map="auto", config=config)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    for task_type in task_type_list:
        load_path = "HelloBench/" + task_type + ".jsonl"
        output_path = os.path.join("results", "suri", task_type + "_results.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(load_path, "r", encoding="utf-8") as f:
            data_list = [json.loads(line) for line in f.readlines()]

        for data_dict in data_list:
            chat_prompt = data_dict["chat_prompt"]
            prompt = [
                {
                    "role": "user",
                    "content": chat_prompt,
                }
            ]
            input_context = tokenizer.apply_chat_template(
                prompt, add_generation_prompt=True, tokenize=False
            )
            input_ids = tokenizer.encode(
                input_context, return_tensors="pt", add_special_tokens=False
            ).to(model.device)
            output = model.generate(
                input_ids, max_length=16384, do_sample=True, use_cache=True, temperature=0.8
            ).cpu()
            response = tokenizer.decode(output[0][input_ids.shape[-1]:])

            output_dict = {"id": data_dict["id"], "instruction": chat_prompt, "response": response}
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


def run_longwriter(task_type_list):
    """
    Run LongWriter on HelloBench
    :param task_type_list: list of task types
    """
    tokenizer = AutoTokenizer.from_pretrained("THUDM/LongWriter-glm4-9b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("THUDM/LongWriter-glm4-9b", torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True, device_map="auto")
    model = model.eval()

    for task_type in task_type_list:
        load_path = "HelloBench/" + task_type + ".jsonl"
        output_path = os.path.join("results", "longwriter", task_type + "_results.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(load_path, "r", encoding="utf-8") as f:
            data_list = [json.loads(line) for line in f.readlines()]

        for data_dict in data_list:
            chat_prompt = data_dict["chat_prompt"]
            query = chat_prompt
            response, _ = model.chat(tokenizer, query, history=[], max_new_tokens=16384, temperature=0.8)

            output_dict = {"id": data_dict["id"], "instruction": chat_prompt, "response": response}
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


def run_glm4_9b_chat_1m(task_type_list):
    """
    Run GLM-4-9B-Chat-1M on HelloBench
    :param task_type_list: list of task types
    """
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat-1m", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "THUDM/glm-4-9b-chat-1m",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()

    for task_type in task_type_list:
        load_path = "HelloBench/" + task_type + ".jsonl"
        output_path = os.path.join("results", "glm4_9b_chat_1m", task_type + "_results.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(load_path, "r", encoding="utf-8") as f:
            data_list = [json.loads(line) for line in f.readlines()]

        for data_dict in data_list:
            chat_prompt = data_dict["chat_prompt"]
            query = chat_prompt
            inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                                   add_generation_prompt=True,
                                                   tokenize=True,
                                                   return_tensors="pt",
                                                   return_dict=True
                                                   )
            inputs = inputs.to(device)
            gen_kwargs = {"max_length": 16384, "do_sample": True, "temperature": 0.8}
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            output_dict = {"id": data_dict["id"], "instruction": chat_prompt, "response": response}
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


def run_llama31_70b_length_constrained_exps():
    """
    Run Meta-Llama-3.1-70B-Instruct on HelloBench-Heuristic_Text_Generation with different length-constaints
    """
    pipeline = transformers.pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    task_type_list = ["heuristic_text_generation"]

    for task_type in task_type_list:
        for length in ["2k", "4k", "8k", "16k"]:
            load_path = "HelloBench/length_constrained_experiments_data/" + task_type + "_" + length + ".jsonl"
            output_path = os.path.join("results", "llama31_70b",
                                       task_type + "_" + length + "_results.jsonl")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(load_path, "r", encoding="utf-8") as f:
                data_list = [json.loads(line) for line in f]
            for data_dict in data_list:
                instruction = data_dict["chat_prompt"]
                messages = [{"role": "user", "content": instruction}]
                if length == "16k":
                    max_tokens = 24000
                else:
                    max_tokens = 16384

                response = pipeline(messages, max_new_tokens=max_tokens, temperature=0.8)[0]["generated_text"][-1][
                    "content"]

                output_dict = {"id": data_dict["id"], "instruction": instruction, "response": response}
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


def run_qwen2_72b_length_constrained_exps():
    """
    Run Qwen2-72B-Instruct on HelloBench-Heuristic_Text_Generation with different length-constaints
    """
    device = "cuda" 
    model_id = "Qwen/Qwen2-72B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    task_type_list = ["heuristic_text_generation"]
    for task_type in task_type_list:
        for length in ["2k", "4k", "8k", "16k"]:
            load_path = "HelloBench/length_constrained_experiments_data/" + task_type + "_" + length + ".jsonl"
            output_path = os.path.join("results", "qwen2_72b", task_type + "_" + length + "_results.jsonl")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(load_path, "r", encoding="utf-8") as f:
                data_list = [json.loads(line) for line in f]

            for data_dict in data_list:
                instruction = data_dict["chat_prompt"]
                messages = [{"role": "user", "content": instruction}]
                if length == "16k":
                    max_tokens = 24000
                else:
                    max_tokens = 16384
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(device)
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=0.8
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                output_dict = {"id": data_dict["id"], "instruction": instruction, "response": response}
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


def run_suri_length_constrained_exps():
    """
    Run Suri on HelloBench-Heuristic_Text_Generation with different length-constaints
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    model_name = "cache_models/suri"
    base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    config = PeftConfig.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
    model = PeftModel.from_pretrained(base_model, model_name, device_map="auto", config=config)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    task_type_list = ["heuristic_text_generation"]
    for task_type in task_type_list:
        for length in ["2k", "4k", "8k", "16k"]:
            load_path = "HelloBench/length_constrained_experiments_data/" + task_type + "_" + length + ".jsonl"
            output_path = os.path.join("results", "suri", task_type + "_" + length + "_results.jsonl")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(load_path, "r", encoding="utf-8") as f:
                data_list = [json.loads(line) for line in f.readlines()]

            for data_dict in data_list:
                chat_prompt = data_dict["chat_prompt"]
                prompt = [
                    {
                        "role": "user",
                        "content": chat_prompt,
                    }
                ]
                input_context = tokenizer.apply_chat_template(
                    prompt, add_generation_prompt=True, tokenize=False
                )
                input_ids = tokenizer.encode(
                    input_context, return_tensors="pt", add_special_tokens=False
                ).to(model.device)
                if length == "16k":
                    max_length = 24000
                else:
                    max_length = 16384
                output = model.generate(
                    input_ids, max_length=max_length, do_sample=True, use_cache=True, temperature=0.8
                ).cpu()
                response = tokenizer.decode(output[0][input_ids.shape[-1]:])

                output_dict = {"id": data_dict["id"], "instruction": chat_prompt, "response": response}
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


def run_internlm_2_5_20b_chat_length_constrained_exps():
    """
    Run InternLM-2.5-20B-Chat on HelloBench-Heuristic_Text_Generation with different length-constaints
    """
    tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2_5-20b-chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("internlm/internlm2_5-20b-chat", torch_dtype=torch.float16,
                                                 trust_remote_code=True, device_map="auto")
    model = model.eval()

    task_type_list = ["heuristic_text_generation"]
    for task_type in task_type_list:
        for length in ["2k", "4k", "8k", "16k"]:
            load_path = "HelloBench/length_constrained_experiments_data/" + task_type + "_" + length + ".jsonl"
            output_path = os.path.join("results", "internlm2_5_20b_chat",
                                       task_type + "_" + length + "_results.jsonl")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(load_path, "r", encoding="utf-8") as f:
                data_list = [json.loads(line) for line in f.readlines()]

            for data_dict in data_list:
                chat_prompt = data_dict["chat_prompt"]
                query = chat_prompt
                if length == "16k":
                    max_tokens = 24000
                else:
                    max_tokens = 16384
                response, _ = model.chat(tokenizer, query, history=[], max_new_tokens=max_tokens, temperature=0.8)

                output_dict = {"id": data_dict["id"], "instruction": chat_prompt, "response": response}
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


def run_longwriter_length_constrained_exps():
    """
    Run LongWriter on HelloBench-Heuristic_Text_Generation with different length-constaints
    """
    tokenizer = AutoTokenizer.from_pretrained("THUDM/LongWriter-glm4-9b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("THUDM/LongWriter-glm4-9b", torch_dtype=torch.bfloat16,
                                                trust_remote_code=True, device_map="auto")
    model = model.eval()

    task_type_list = ["heuristic_text_generation"]
    for task_type in task_type_list:
        length_list = ["2k", "4k", "8k", "16k"]
        for length in length_list:
            load_path = "HelloBench/length_constrained_experiments_data/" + task_type + "_" + length + ".jsonl"
            output_path = os.path.join("results", "longwriter", task_type + "_" + length + "_results.jsonl")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(load_path, "r", encoding="utf-8") as f:
                data_list = [json.loads(line) for line in f.readlines()]

            for data_dict in data_list:
                chat_prompt = data_dict["chat_prompt"]
                query = chat_prompt
                if length == "16k":
                    max_tokens = 24000
                else:
                    max_tokens = 16384
                response, _ = model.chat(tokenizer, query, history=[], max_new_tokens=max_tokens, temperature=0.8)

                output_dict = {"id": data_dict["id"], "instruction": chat_prompt, "response": response}
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    task_type_list = ["heuristic_text_generation", "summarization", "text_completion", "chat", "open_ended_qa"]
    # Uncomment this line if you want to use the corresponding model
    # run_llama31_8b(task_type_list=task_type_list, model_type="chat")
    # run_qwen2_7b(task_type_list=task_type_list, model_type="chat")
    # run_gpt4o_mini(task_type_list=task_type_list)
    # run_llama31_70b(task_type_list=task_type_list, model_type="chat")
    # run_gpt4o(task_type_list=task_type_list)
    # run_gpt4_turbo(task_type_list=task_type_list)
    # run_gemma2_27b(task_type_list=task_type_list)
    # run_map_neo(task_type_list=task_type_list)
    # run_phi_35(task_type_list=task_type_list)
    # run_suri(task_type_list=task_type_list)
    # run_longwriter(task_type_list=task_type_list)
    # run_glm4_9b_chat_1m(task_type_list=task_type_list)
    # run_llama31_70b_length_constrained_exps()
    # run_qwen2_72b_length_constrained_exps()
    # run_suri_length_constrained_exps()
    # run_internlm_2_5_20b_chat_length_constrained_exps()
    # run_longwriter_length_constrained_exps()