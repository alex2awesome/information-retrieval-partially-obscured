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
Given the following information provided by a source, create a question that would elicit this information as an answer. The question should be specific enough to target the given information, but general enough that it doesn't reveal the answer itself.

Format:
INPUT: Information provided to the story
OUTPUT: Question that would elicit this information

Example:
INPUT: The Biden Administration provided information on its efforts to improve access to mental health resources in schools, including a nearly 00 million allotment to expand access to mental health care.
OUTPUT: What specific actions has the Biden Administration taken to address mental health issues in schools?

It's important to return only the question, without any additional text or explanation.
Here's the information:
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
            sources_question[name] = output_question.outputs[0].text

        jsonfile.append({
            'article_url': url,
            'sources': sources,
            'obscured_sources': obscured_sources,
            'questions': sources_question
        })

    return json.dumps(jsonfile, indent=2, ensure_ascii=False)

def main(args):
    source_file = args.source_file
    if 'obsured' not in source_file:
        return
    
    sources_path = os.path.join('../data', source_file)
    with open(sources_path, 'r') as f:
        contents = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)

    processed_content = process_content(contents, tokenizer, model, sampling_params)
    output_path = sources_path.replace("obscured", "processed")

    with open(output_path, 'w') as f:
        f.write(processed_content)

    print("Processing completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file', type=str)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    args = parser.parse_args()
    main(args)
    