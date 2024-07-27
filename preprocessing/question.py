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
You are an AI assistant helping a journalist find appropriate sources for a story. Given the information provided by a source, your task is to create a very brief question that might elicit this source. The question should be specific enough to target the given information, but general enough that it doesn't reveal the source's identity.
Follow these steps:
1. Analyze the source information.
2. Identify the key topics or expertise demonstrated by the source.
3. Create a question requiring the source's specific knowledge or position to answer.
4. Ensure the question doesn't directly reveal the source's identity.

Here are some examples to guide you:
Example 1:
Input: The BBC Research & Development department provides technical information and insights about the halfRF system, a new technology developed by the BBC. They share details about the spectrum analyzer, transmitter antennas, transmit diversity, and the benefits of using MIMO (Multiple-Input Multiple-Output) signals.
Output: How does the halfRF system created by the BBC work?
Example 2:
Input: CVS is the primary source providing information about their decision to remove certain drugs from their insurance coverage. They announce that over a dozen drugs will be dropped, and that equally effective products with lower costs will remain on the list.
Output: Which insurance companies are removing drugs from their insurance coverage? Which drugs are insurance companies removing? Why are insurance companies removing drugs from their coverage?
Example 3:
Input: Mike Duggan, Detroit Mayor, provides context on why his city turned to data-driven government, highlighting its potential for transparency, accountability, and fact-based decision-making.
Output: Why do cities become data-driven? What benefits does data-driven governance have?
Example 4:
Input: Susan Scrupski, Founder of Big Mountain Data (BMD), shares the organization's long-term ambition to establish a national open-source repository on repeat offender data and discusses its current efforts in tracking heating violations in New York City.
Output: What kinds of projects are considered "data-driven city projects"? What types of issues can you address with data-driven governance?

Now, process the following information and generate a similar question that would elicit this source without revealing their identity. Reply only with the brief, resulting question:

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
            sources_question[name] = output_question.encode('utf-8')

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
    output_directory = '../data_preprocessed'
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
        with open(file_path, 'r') as f:
            contents = json.load(f)

        processed_content = process_content(contents, tokenizer, model, sampling_params)
        

        with open(output_path, 'w') as f:
            f.write(processed_content)
        count += 1
        print(f"{count} files preprocessed!")

    print("Processing completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    args = parser.parse_args()
    main(args)
