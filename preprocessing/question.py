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
You are an AI assistant for journalists. Output one sentence only based on the information provided by the overview below. State the preliminary question the overview answers, Please output this one question only. Incorporate the initial story lead and the reason why the journalist started investigating this topic.
{source}
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

        for name, description in sources.items():
            prompt_question = system_prefix_question.format(source=description)

            message_question = [
                {"role": "system", "content": "You are an experienced journalist."},
                {"role": "user", "content": prompt_question}
            ]

            messages_question.append(tokenizer.apply_chat_template(message_question, tokenize=False, add_generation_prompt=True))

        outputs_question = model.generate(messages_question, sampling_params)

        sources_question = {}

        for name, output_question in zip(sources.keys(), outputs_question):
            output_question = unicodedata.normalize('NFKC', output_question.outputs[0].text)
            sources_question[name] = output_question

        jsonfile.append({
            'article_url': url,
            'sources': sources,
            'obscured_sources': obscured_sources,
            'questions': sources_question
        })

    return json.dumps(jsonfile, indent=2, ensure_ascii=False)

def exist_question(output_path):
    if os.path.isfile(output_path):
        return True
    return False


def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)

    directory = '../data'
    output_directory = '../data_preprocessed_new'
    count = 0
    for filename in os.listdir(directory):
        if 'obscured' not in filename:
            continue
        file_path = os.path.join(directory, filename)
        output_name = filename.replace("obscured", "processed")
        output_path = os.path.join(output_directory, output_name)
        if exist_question(output_path):
            print(f'{filename} is already preprocessed.')
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            contents = json.load(f)

        processed_content = process_content(contents, tokenizer, model, sampling_params)
        

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        count += 1
        print(f"{count} files preprocessed!")

    print("Processing completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    args = parser.parse_args()
    main(args)
