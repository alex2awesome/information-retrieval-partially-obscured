import re
import json
import logging
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import unicodedata
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
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

system_prefix_question = '''

You are an AI assistant helping a journalist refine their query to find appropriate sources for a story. Given an initial query, your task is to generate a series of augmentations that progressively narrow down the types of sources that could answer the query.

Here's an example to guide you:

Input: How is the Senate dealing with recent events, are they going to take a standard process or something different?
Output: We want to find some Republicans in positions of power that might have information on legislation, Republicans in the Senate have more power to block legislation, some Republicans in positions of power are: Kevin McCarthy, Donald Trump, Mitch McConnell

It's important to output only the augmentations separated by commas and nothing else. Follow this example and do the same analysis for the following source:

{question}
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
        questions = content['questions']
        messages_question = []

        for name, description in questions.items():
            prompt_question = system_prefix_question.format(question=description)

            message_question = [
                {"role": "system", "content": "You are an AI assistant helping a journalist find appropriate sources for a story."},
                {"role": "user", "content": prompt_question}
            ]

            messages_question.append(tokenizer.apply_chat_template(message_question, tokenize=False, add_generation_prompt=True))

        outputs_question = model.generate(messages_question, sampling_params)

        questions_augmentation = {}

        for name, output_question in zip(sources.keys(), outputs_question):
            questions_augmentation[name] = unicodedata.normalize('NFKC', output_question.outputs[0].text)

        jsonfile.append({
            'article_url': url,
            'sources': sources,
            'questions': questions,
            'questions_augmentation': questions_augmentation
        })

    return json.dumps(jsonfile, indent=2, ensure_ascii=False)

def exist_aug(output_path):
    if os.path.isfile(output_path):
        return True
    return False


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)

    directory = '../data_preprocessed'
    output_directory = '../question_augmented'
    count = 0
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        output_name = filename.replace("preprocessed", "question_aug")
        output_path = os.path.join(output_directory, output_name)
        if exist_aug(output_path):
            print(f'{filename} is already augmented.')
            continue
        if os.path.isdir(file_path):
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                contents = json.load(f)
            except json.decoder.JSONDecodeError:
                print(f'{filename} is empty or not well formatted')
                continue
            # contents = json.load(f)

        processed_content = process_content(contents, tokenizer, model, sampling_params)
        

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        count += 1
        print(f"{count} files augmented!")

    print("Augmentation completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    args = parser.parse_args()
    main(args)
