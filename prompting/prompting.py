import pandas as pd
import numpy as np

from vllm import LLM,  SamplingParams
from transformers import AutoTokenizer
import os

import json
import torch
import logging

import argparse


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


HF_HOME = "/project/jonmay_231/spangher/huggingface_cache"
config_data = json.load(open('config.json'))
os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
os.environ['HF_HOME'] = HF_HOME

def load_model(model):
    torch.cuda.memory_summary(device=None, abbreviated=False)
    model = LLM(
        model,
        dtype=torch.float16,
        tensor_parallel_size=torch.cuda.device_count(),
        download_dir=HF_HOME, # sometimes the distributed model doesn't pay attention to the 
        enforce_eager=True
    )
    return model


def infer(model, messages):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
    formatted_prompt =  tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    output = model.generate(formatted_prompt, sampling_params)

    return output[0].outputs[0].text


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    parser.add_argument('--data_dir', type=str, default='./data')
    # parser.add_argument('--start_idx', type=int, default=None)
    # parser.add_argument('--end_idx', type=int, default=None)
    args = parser.parse_args()




    sources_path = '../conditional_information_retrieval/sources_data_70b__200000_200100.txt'
    with open(sources_path, 'r') as f:
        sources = f.read()
        print('sources:', sources)


    system_prefix = '''
    For each given text, obscure the specific details by leaving out all important information except for a short, generalized biographical description.

    Format:
    1. **Original**: Identity information + Biographical information + Given information
    2. **Obscured**: Identity information + Biographical information

    Here're some examples:
    1. **Socrata Foundation:** The Socrata Foundation provides information about its philanthropic philosophy and mandate to support unique organizations that lack resources or financial means to fulfill their data-driven mission. It also explains how it will proactively support open data efforts that deliver social impact and long-term value.
    **Socrata Foundation:** The Socrata Foundation supports organizations lacking resources or financial means.

    2. **Robert Runge:** Robert Runge, a member of the Socrata Board of Directors, provides additional context on the role of the Socrata Foundation in bridging the gap between publicly funded open data projects and underfunded or unfunded opportunities.
    **Robert Runge:** Robert Runge, a board member of the Socrata Foundation.

    It's important to return the obscured text only.
    Here's the text:
    {source}
    '''

    prompt = system_prefix.format(sources=sources)
    print('prompt:\n', prompt)
    message = [
            {
                "role": "system",
                "content": "You are an experienced journalist.",
            },

            {
                "role": "user",
                "content": prompt
            },
        ]

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    formatted_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    model = load_model(args.model)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    output = model.generate(message, sampling_params)

    fname = 'sources_data_70b__200000_200100_obscured.txt'
    with open(fname, 'w') as f:
        f.write(output)