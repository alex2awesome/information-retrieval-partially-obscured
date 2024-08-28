from datasets import load_from_disk
import pandas as pd
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import unicodedata

import os
import json
import torch
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

from vllm import LLM,  SamplingParams
from transformers import AutoTokenizer
import os
HF_HOME = "/project/jonmay_231/spangher/huggingface_cache"
proj_dir = '/project/jonmay_231/spangher/Projects/conditional-information-retrieval'
config_data = json.load(open(f'{proj_dir}/config.json'))
os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
os.environ['HF_HOME'] = HF_HOME
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

BATCH_SIZE = 50
INFORMATION_PROMPT = '''
Here is a news article, with each sentence annotated according to the source of it‛s information:
    ```{json_str}```

    Please restate the biographical information used to describe the source in the article. 
    Do NOT include any information that the source gives to the story. ONLY include biographical information about the source.
    Include unnamed sources (e.g. "witnesses"). Include any biographical information that might pertain to the source, even if we didn't label it.
    Generate only ONE summary per source. Group sources that are clearly the same but named slightly differently. For example: "Andrew Dresden" and "Dresden" should be grouped together as one source. "Lao Diplomats" and "Laotian Diplomats" should be grouped together as one source.
    Split source annotations that refer to multiple sources into separate summaries. For example: if the annotation is "John and Jane", generate two separate summaries, one for "John" and one for "Jane". 
    
    For each source, provide the following information:
        (1) Name: who the source is.
        (2) Original Name: What their original name(s) are in our annotations.
        (3) Information: Restate the biographical information provided by the source. Be as SPECIFIC and be as VERBOSE as possible. 
            Contextualize all biographical information about the source with AS MUCH BACKGROUND INFORMATION mentioned in the article so I can fully understand who the source is.
            Be verbose.
            For example, don't just say "the crash", say "the plane crash carrying key senior Laotian government officials".
            Don't include any information that the source directly provided to the story.
    Output the summary in a list of python dictionaries with one key per number. Don't say anything else.
'''


def format_prompt(prompt: str, json_str: str) -> str:
    message = [
        {
            "role": "system",
            "content": "You are an experienced journalist.",
        },

        {
            "role": "user",
            "content": prompt.format(json_str=json_str)
        },
    ]
    formatted_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    return formatted_prompt

def load_model(model_name: str):
    torch.cuda.memory_summary(device=None, abbreviated=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LLM(
        model_name,
        dtype=torch.float16,
        tensor_parallel_size=torch.cuda.device_count(),
        download_dir=HF_HOME, # sometimes the distributed model doesn't pay attention to the 
        enforce_eager=True
    )
    return tokenizer, model



def load_full_dataset_from_disk(args):
    # load in the data
    source_df = pd.read_json(
        f'{args.data_dir}/{args.source_data_file}', nrows=args.end_idx, lines=True
    ).iloc[args.start_idx:]
    article_d = load_from_disk(f'{args.data_dir}/all-coref-resolved')

    # process the data into right format: article with annotated sentences
    a_urls_lookup = set(source_df['article_url'])
    filtered_article_d = article_d.filter(lambda x: x['article_url'] in a_urls_lookup, num_proc=10)
    disallowed_quote_types = set(['Other', 'No Quote'])
    # disallowed_sources = set(['journalist', 'passive-voice'])
    # disallowed_sources = set(['passive-voice'])
    sentences_with_quotes = (
        filtered_article_d
        .to_pandas()
        .merge(source_df, on='article_url')
        [['article_url', 'attributions', 'quote_type', 'sent_lists', ]]
        .explode(['attributions', 'quote_type', 'sent_lists'])
    )

    sentences_with_quotes = (
        sentences_with_quotes.assign(
            attributions=lambda df: df.apply(lambda x:
                x['attributions'] if (
                    (len(x['attributions']) < 50)
                    or (x['quote_type'] not in disallowed_quote_types)
                    # or (x['attributions'] not in disallowed_sources)
            ) else np.nan, axis=1)
        )
    )
    return sentences_with_quotes


def write_to_file(fname, urls, outputs):
    with open(fname, 'wb') as file:
        for url, output in zip(urls, outputs):
            response = output.outputs[0].text
            response = unicodedata.normalize('NFKC', response)
            if response and url:
                output = {}
                output['url'] = str(url)
                output['response'] = str(response)
                file.write(json.dumps(output).encode('utf-8'))
                file.write(b'\n')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    parser.add_argument('--data_dir', type=str, default=f'{proj_dir}/data')
    parser.add_argument('--source_data_file', type=str, default='full-source-scored-data.jsonl')
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--input_data_file', type=str, default=None)
    parser.add_argument('--id_col', type=str, default='article_url')
    parser.add_argument('--source_col', type=str, default='attributions')
    parser.add_argument('--sent_col', type=str, default='sent_lists')
    parser.add_argument('--output_file', type=str, default='sources_data_70b.txt')
    args = parser.parse_args()

    if args.input_data_file is None:
        sentences_with_quotes = load_full_dataset_from_disk(args)
    else:
        sentences_with_quotes = pd.read_csv(args.input_data_file, index_col=0)

    tokenizer, model = load_model(args.model)
    # store each article_url, annoted_sentences pair
    # hold the batches
    url_batches, message_batches = [], []
    # each batch
    urls, info_messages = [], []
    for url in sentences_with_quotes[args.id_col].unique():
        one_article = (
            sentences_with_quotes
                .loc[lambda df: df[args.id_col] == url]
                .reset_index(drop=True)
        )

        json_str = (
            one_article[[args.sent_col, args.source_col]]
            .rename(columns={args.sent_col: 'sentence', args.source_col: 'source'})
            .to_json(lines=True, orient='records')
        )

        info_messages.append(format_prompt(INFORMATION_PROMPT, json_str))
        urls.append(url)

        if len(info_messages) >= BATCH_SIZE:
            message_batches.append(info_messages)
            url_batches.append(urls)
            info_messages = []
            urls = []

    # load the model
    sampling_params = SamplingParams(temperature=0.1, max_tokens=1024)
    if args.start_idx is None:
        args.start_idx = 0
    if args.end_idx is None:
        args.end_idx = len(urls)

    # generate the summaries
    start_idx = args.start_idx
    end_idx = start_idx + BATCH_SIZE
    for info_messages, urls in zip(tqdm(message_batches), url_batches):
        fname, fext = os.path.splitext(args.output_file)
        info_fname = f'{fname}__{start_idx}_{end_idx}{fext}'
        # generate the informational summaries
        info_outputs = model.generate(info_messages, sampling_params)
        write_to_file(info_fname, urls, info_outputs)
        # update the indices
        start_idx = end_idx
        end_idx = start_idx + BATCH_SIZE


"""
import attrdict
args = attrdict.AttrDict()
args.id_col = 'doc_id'
args.source_col = 'head'
args.sent_col = 'sent'
args.output_file = 'annotated_sources_summarized.txt'
args.input_data_file = 'full-training-df.csv'
args.model = 'meta-llama/Meta-Llama-3-70B-Instruct'
args.start_idx = None
args.end_idx = None


    python data_vllm_70b.py \
      --start_idx 0 \
      --end_idx 5 \
      --id_col  doc_id \
      --source_col  head \
      --sent_col  sent \
      --output_file  annotated_sources_summarized.txt \
      --input_data_file  full-training-df.csv


"""

"""
original prompt:

    Here is a news article, with each sentence annotated according to the source of it's information:
    ```
    {json_str}
    ```

    Please summarize each source, based on our source annotations. 
    Tell me in one paragraph per source: (1) who the source is (2) what informational content they provide to the article. 
    Only rely on the annotations I have provided, don't identify additional sources. 
    Generate only ONE summary per source. Group sources that are clearly the same but named slightly differently.
    That is, summarize the SAME source if it occurs in multiple source annotations. 

"""