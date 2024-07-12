
from vllm import LLM,  SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import pdb

import json
import logging
import argparse


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


HF_HOME = "/project/jonmay_231/spangher/huggingface_cache"
os.environ['HF_TOKEN'] = "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"
os.environ['HF_HOME'] = HF_HOME

system_prefix = '''
    Given the following information provided by a source, create a question that would elicit this information as an answer. 
    The question should be very general and it should be difficult to pick the right source. 
    The question shouldn't reveal the source's name, the source itself, the answer itself, or any proper nouns. 
    It shouldn't give any hints about what it could be at all. 

    Format: 
    INPUT: Information provided to the story 
    OUTPUT: Broad, general question that might elicit this information 
    
    Example: 
    INPUT: The Biden Administration provided information on its efforts to improve access to mental health resources in schools, including a nearly $300 million allotment to expand access to mental health care. 
    OUTPUT: What specific actions have been taken to address mental health issues in schools? 
    
    It's important to return only the question, without any additional text or explanation, and exclude the word 'OUTPUT'. Execute this prompt:
    For each given text, obscure the specific details by leaving out all important information except for a short, generalized biographical contextless description about who/what the source is.

    Here's the source:
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

def obscure(contents, tokenizer, model, sampling_params):
    jsonfile = []

    for content in contents:
        url = content['article_url']
        sources = content['sources']      # a dictionary
        messages = []
        for name, description in sources.items():

            prompt = system_prefix.format(source=description)
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
        
        outputs = model.generate(messages, sampling_params)  
        sources_obsc = {}

        for name, output in zip(sources.keys(), outputs):
            sources_obsc[name] = output.outputs[0].text


        jsonfile.append({
            'article_url': url,
            'sources': sources,
            'obscured_sources': sources_obsc
        })

    return json.dumps(jsonfile, indent=2, ensure_ascii=False)


def main(args):

    source_file = args.source_file
    sources_path = '../data/' + source_file + '.json'
    with open(sources_path, 'r') as f:
        contents = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)

    obscured_content = obscure(contents, tokenizer, model, sampling_params)
    output_path = '../data/' + source_file + '_obscured.json'

    with open(output_path, 'w') as f:
        f.write(obscured_content)

    
    print("DONE!!!!!!!!!!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file', type=str)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    args = parser.parse_args()
    # pdb.set_trace()
    main(args)