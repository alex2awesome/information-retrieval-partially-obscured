
from vllm import LLM,  SamplingParams
from transformers import AutoTokenizer
import os
import pdb

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
# with open('config.json', 'r') as f:
#     config_data = json.load(f)
# config_data = json.load(open('config.json'))
os.environ['HF_TOKEN'] = "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"
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
    pdb.set_trace()




    sources_path = 'sources_data_70b__200000_200100.txt'
    # with open(sources_path, 'r') as f:
    #     sources = f.read()
    #     print('sources:', sources)

    sources = '''The source tripulantes refers to flight attendants. They provide information about their concerns and practices during the pandemic, such as leaving food trays on an intermediate seat between them and the passenger, and wanting concrete measures to avoid contagion.'''

    system_prefix = '''
    For each given text, obscure the specific details by leaving out all important information except for a short, generalized biographical contextless description about who/what the source is.

    Format:
    INPUT: Identity information + Biographical information + Given information
    OUTPUT: Identity information + Biographical information

    Example:
    INPUT: The Biden Administration provided information on its efforts to improve access to mental health resources in schools, including a nearly $300 million allotment to expand access to mental health care.
    OUTPUT: The Biden Administration is the executive branch of the U.S. federal government under the leadership of President Joe Biden.

    If there is no information in the original entry on what the source is, only include the source name and nothing else. However, if you are fully certain that you can infer what the source is without hallucinating, include that.
    Execute this task on all entries below. Only include one output line for each entry, include nothing except what comes after "OUTPUT" (excluding the word"OUTPUT"):

    It's important to return the obscured text only.
    Here's the text:
    {source}
    '''



    prompt = system_prefix.format(source=sources)
    # print('prompt:\n', prompt)
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
    outputs = model.generate([message], sampling_params)
    fname = 'sources_data_70b__200000_200100_obscured.txt'
    with open(fname, 'w') as f:
        f.write("wtf?????")

    for output in outputs:
        content = output.outputs[0].text
        fname = 'sources_data_70b__200000_200100_obscured.txt'
        with open(fname, 'w') as f:
            f.write("wtf?????")
            f.write(content)

    
    print("DONE!!!!!!!!!!!!")