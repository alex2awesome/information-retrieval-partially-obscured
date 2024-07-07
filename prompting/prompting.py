
from vllm import LLM,  SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
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


def main(args):

    source_file = args.source_file
    sources_path = '../data/' + source_file + '.json'
    with open(sources_path, 'r') as f:
        contents = f.read()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)

    messages = []
    urls = []
    sources = []

    for content in contents:
        url = content['article_url']
        source = content['source']
        prompt = system_prefix.format(source=source)

        
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
        formatted_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        messages.append(formatted_prompt)
        urls.append(url)
        sources.append(source)
        
    outputs = model.generate(messages, sampling_params)  

    output_content = []
    for url, source, output in zip(urls, sources, outputs):
        obsc_source = output.outputs[0].text
        output_content.append({
            "article_url": url,
            "source": source,
            "obscured_source": obsc_source
        })
    
    print("DONE!!!!!!!!!!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file', type=str)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    args = parser.parse_args()
    pdb.set_trace()
    main(args)