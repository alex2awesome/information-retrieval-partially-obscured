import json
import os
import logging
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log system information
logging.info(f"Python version: {sys.version}")
logging.info(f"Current working directory: {os.getcwd()}")
logging.info(f"PyTorch version: {torch.__version__}")
logging.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

HF_HOME = "/project/jonmay_231/spangher/huggingface_cache"
os.environ['HF_HOME'] = HF_HOME

def load_model(model_name: str):
    logging.info(f"Attempting to load model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        logging.info("Model loaded successfully")
        return classifier
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        raise

def process_entry(entry, classifier):
    url, content = entry.split('{', 1)
    content = '{' + content.strip()
    
    try:
        source_data = json.loads(content)
    except json.JSONDecodeError:
        logging.error(f"Failed to parse JSON for URL: {url}")
        logging.error(f"Problematic content: {content}")
        return None

    obscured_sources = {}
    for source, info in source_data.items():
        prompt = f"Task: For each given text, obscure the specific details by leaving out all important information except for a short, generalized biographical contextless description about who/what the source is. Format: INPUT: Source name + what the source is + what information the source provides. OUTPUT: Source name + what the source is. Example: INPUT: The Biden Administration provided information on its efforts to improve access to mental health resources in schools, including a nearly $300 million allotment to expand access to mental health care. OUTPUT: The Biden Administration is the executive branch of the U.S. federal government under the leadership of President Joe Biden. If there is no information in the original entry on what the source is, only include the source name and nothing else. However, if you are 100% certain that you can infer what the source is without hallucinating, include that. Execute this task on the entry below. Only include one output line, include nothing except what comes after 'OUTPUT' (excluding the word 'OUTPUT'): INPUT: {source}: {info}"
        
        # Use the classifier to generate a response
        result = classifier(prompt, max_length=100, num_return_sequences=1)[0]
        obscured_info = result['generated_text'].strip()
        
        # Ensure the output starts with the source name
        if not obscured_info.startswith(source):
            obscured_info = f"{source}: {obscured_info}"
        
        obscured_sources[source] = obscured_info

    return url, obscured_sources

def main():
    logging.info("Starting main function")
    try:
        classifier = load_model("distilbert-base-uncased-finetuned-sst-2-english")

        input_file = 'sources_data_70b__200000_200100.txt'
        output_file = 'obscured_sources_output.txt'

        logging.info(f"Input file: {input_file}")
        logging.info(f"Output file: {output_file}")

        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            current_entry = ""
            for line in infile:
                if line.strip() == '}':
                    current_entry += line
                    result = process_entry(current_entry, classifier)
                    if result:
                        url, obscured_sources = result
                        outfile.write(f"{url}\n")
                        outfile.write("{\n")
                        for source, obscured in obscured_sources.items():
                            outfile.write(f'"{source}": "{obscured}",\n')
                        outfile.write("}\n\n")
                    current_entry = ""
                else:
                    current_entry += line

        logging.info("Processing complete")
    except Exception as e:
        logging.error(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
    main()