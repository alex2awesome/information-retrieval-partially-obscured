
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

system_prefix_obscure = '''
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
        download_dir=HF_HOME, # sometimes the distributed model doesn't pay attention to the 
        enforce_eager=True
    )
    return model

def obscure(contents, tokenizer, model, sampling_params):
    jsonfile = []

    for content in contents:
        url = content['article_url']
        sources = content['sources'] 
        messages_obscure = []
        # messages_question = []
        for name, description in sources.items():

            prompt_obscure = system_prefix_obscure.format(source=description)
            # prompt_question = system_prefix_question.format(source=description)
            message_obscure = [
                    {
                        "role": "system",
                        "content": "You are an experienced journalist.",
                    },

                    {
                        "role": "user",
                        "content": prompt_obscure
                    },
                ]
            
            # message_question = [
            #         {
            #             "role": "system",
            #             "content": "You are an experienced journalist.",
            #         },

            #         {
            #             "role": "user",
            #             "content": prompt_question
            #         },
            #     ]
            
                
            formatted_prompt_obscure = tokenizer.apply_chat_template(message_obscure, tokenize=False, add_generation_prompt=True)
            # formatted_prompt_question = tokenizer.apply_chat_template(message_question, tokenize=False, add_generation_prompt=True)
            messages_obscure.append(formatted_prompt_obscure)
            # messages_question.append(formatted_prompt_question)
        
        outputs_obscure = model.generate(messages_obscure, sampling_params)  
        # outputs_question = model.generate(messages_question, sampling_params)  
        sources_obscure = {}
        # questions = {}

        for name, output in zip(sources.keys(), outputs_obscure):
            sources_obscure[name] = output.outputs[0].text
        
        # for name, output in zip(sources.keys(), outputs_question):
        #     questions[name] = output.outputs[0].text


        jsonfile.append({
            'article_url': url,
            'sources': sources,
            'obscured_sources': sources_obscure,
            # 'question': questions
        })

    return json.dumps(jsonfile, indent=2, ensure_ascii=False)

def exist_obscure(filename):
    base, extension = os.path.splitext(filename)
    filename_obscured = f"{base}_obscured{extension}"
    if os.path.isfile(filename_obscured):
        return True
    return False



def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    directory = '../data'
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if 'obscured' in file_name:
            continue
        if exist_obscure(file_path):
            print(f'{file_name} is already obscured.')
            break
            continue
        
        with open(file_path, 'r') as f:
            contents = json.load(f)

        obscured_content = obscure(contents, tokenizer, model, sampling_params)
        output_name = file_name.split('.')[0] + '_obscured.json'
        output_path = '../data/' + output_name

        with open(output_path, 'w') as f:
            f.write(obscured_content)

    
    print("DONE!!!!!!!!!!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    args = parser.parse_args()
    main(args)
