import re
import json
import logging
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import os

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

HF_HOME = "/project/jonmay_231/spangher/huggingface_cache"
os.environ['HF_TOKEN'] = "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"
os.environ['HF_HOME'] = HF_HOME

system_prefix_question = '''

Given the information provided by a source, your task is to reason about the kinds of information it can provide, by checking the information that's included in this source. List the information as number of augmentations. 

Here's an example to guide you: 
Input: Mitch McConnell is a U.S. Senator 
Output: U.S. Senators are responsible for passing legislation, U.S. Senators frequently go on T.V. and give interviews, Mitch McConnell is a senior senator who has been in the Senate since the 1990s 

It's important to output only the augmentations separated by commas and nothing else. Follow this example and do the same analysis for the following source:

{obscured_source}
'''

def load_model(model):
    torch.cuda.memory_summary(device=None, abbreviated=False)
    model = LLM(
        model,
        dtype=torch.float16,
        tensor_parallel_size=torch.cuda.device_count(),
        download_dir=HF_HOME,
        enforce_eager=True
    )
    return model

def process_content(contents, tokenizer, model, sampling_params):
    jsonfile = []

    for content in contents:
        url = content['article_url']
        sources = content['sources']
        obscured_sources = content['obscured_sources']
        messages_question = []

        for name, description in obscured_sources.items():
            prompt_question = system_prefix_question.format(obscured_source=description)

            message_question = [
                {"role": "system", "content": "You are an AI assistant helping a journalist find appropriate sources for a story."},
                {"role": "user", "content": prompt_question}
            ]

            messages_question.append(tokenizer.apply_chat_template(message_question, tokenize=False, add_generation_prompt=True))

        outputs_question = model.generate(messages_question, sampling_params)

        sources_augmentation = {}

        for name, output_question in zip(sources.keys(), outputs_question):
            sources_augmentation[name] = output_question.outputs[0].text

        jsonfile.append({
            'article_url': url,
            'sources': sources,
            'obscured_sources': obscured_sources,
            'augmented_sources': sources_augmentation
        })

    return json.dumps(jsonfile, indent=2, ensure_ascii=False)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)

    directory = '../data'
    output_directory = '../data_augmented'
    count = 0
    for filename in os.listdir(directory):
        if 'obscured' not in filename:
            continue
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as f:
            contents = json.load(f)

        processed_content = process_content(contents, tokenizer, model, sampling_params)
        output_name = filename.replace("obscured", "source_aug")
        output_path = os.path.join(output_directory, output_name)

        with open(output_path, 'w') as f:
            f.write(processed_content)
        count += 1
        print(f"{count} files augmented!")

    print("Augmentation completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    args = parser.parse_args()
    main(args)
