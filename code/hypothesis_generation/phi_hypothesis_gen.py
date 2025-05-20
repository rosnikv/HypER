import os
# add gpu support
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
from dotenv import load_dotenv
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
from huggingface_hub import login
login(token=huggingface_token)
import sys
sys.path.append('path')

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import torch
import json
#from together import Together
from researchagent import ra_prompt_edited, ra_prompt_edited_for_validity, reviewagent, ra_prompt_edited_for_validity_fewshot, fewshot_reviewagent
from novelty_rag_001 import novelty_query_prompt, novelty_judgement_prompt, async_semantic_call, novelty_system_prompt, call_semantic
import asyncio
from collections import defaultdict
import re

base_model_path = "microsoft/Phi-3-mini-128k-instruct"
fine_tuned_model_path = "./llm-reason-hg/axolotl/outputs/phi3-hypER-mixed-lora-out-full2/merged/"


# Load model and tokenizer
# torch_dtype="auto",
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    fine_tuned_model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)

pipe = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 2048,
    "return_full_text": True,
    "temperature": 0.3,
    "do_sample": True,
}


def load_dataset(data_path):
    with open(data_path, 'r') as f:
        d = json.load(f)
    metadata_fields = ["file_name", "chain_label", "chain_source", "file_path", "generated_from_split", "target_hypothesis"]
    chains = []
    meta_chains = []
    label_papers = []
    for data in d:
        if data['content'][-1]['year'] == 2024:
            metadata = {key: data.get(key) for key in metadata_fields}
            meta_chains.append(metadata)
            chains.append(data['content'][:-1])
            label_papers.append(data['content'][-1])
    return meta_chains, chains, label_papers


def load_balanced_dataset(data_path):
    with open(data_path, 'r') as f:
        d = json.load(f)

    metadata_fields = ["file_name", "chain_label", "chain_source", "file_path", "generated_from_split", "target_hypothesis"]
    chains = []
    meta_chains = []
    label_papers = []
    
    # Organize data by type
    type_buckets = defaultdict(list)

    for data in d:
        if data['content'][-1]['year'] == 2024:
            metadata = {key: data.get(key) for key in metadata_fields}
            type_buckets[metadata["chain_label"]].append((metadata, data['content'][:-1], data['content'][-1]))

    # Define how many samples you want per type
    samples_per_type = 5
    for label, items in type_buckets.items():
        print(label)
        selected_items = items[:samples_per_type]  # Pick first 5 (or shuffle and sample)
        for meta, chain, label_paper in selected_items:
            meta_chains.append(meta)
            chains.append(chain)
            label_papers.append(label_paper)
    
    return meta_chains, chains, label_papers


def data_prompt(chain):
    prompt_builder = ""
    for idx, val in enumerate(chain):
        prompt_builder += f"{idx}. Title: {val['title']}; \nAbstract: {val['abstract']}; \nYear: {val['year']}\n\n"
    return prompt_builder

def system_prompt():
    prompt = """You are a medical researcher ideating the next big research to publish. Your task is to conduct
    literature review and come with a novel hypothesis informed by your understanding of the existing work."""
    return prompt


def user_prompt_v1(prompt_builder, variable, topic):
    prompt = f"""Your research goal is to make a novel contribution related to {variable}. You must come with a new hypothesis to define your work.
    You have already identified relevant literature and arranged them in a logical and temporal order as in <paper_list>, such that paper # k+1. is inspired by 
    or strongly dependent on the findings of paper k. Now conduct a literature review identifying these connections and 
    develop a new hypothesis informed by your review. \n\n"""
    prompt += f"<paper_list>: \n{prompt_builder}"
    prompt +=  f"""Your output should be a valid JSON with the following fields. This JSON will be automatically parsed, so ensure the format is precise.   
    Output a JSON object in the following format:
    ```json
    {{
    "Literature review": "Your review of the given papers in the <paper_list> in
    sequential order identifying how each paper built on top of the previous work
    with inline citation of the paper numbers as provided in <paper_list>.",
    "Research idea": "Your novel research idea informed by the review. Clearly
    identify how your idea is inspired by the previous work and different from the 
    previous work.",
    "Hypothesis": "A concise hypothesis based on the research idea with clear variables and 
    their potential relations."
    }}
    ```
    """
    return prompt

def user_prompt_v2(hypothesis_graph):
    prompt = """You are given a graph of hypothesis."""


def run_model(pipe, system_prompt, user_prompt):
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    output = pipe(messages, **generation_args)
    return output

'''
def call_together(system_prompt, user_prompt):
    deployment_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    client = Together()
    response = client.chat.completions.create(
        model = deployment_name,
        messages = [{'role':'system', 'content': system_prompt}, 
        {'role':'user', 'content':user_prompt}],
        temperature = 0,
        seed = 42
    )
    return response.choices[0].message.content
'''
def call_gpt(user_prompt, system_prompt = "You are a scientist reviewing a paper. Your response must be in JSON format"):
    import openai
    import os
    from openai import OpenAI
    openai.api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    print("GPT4 usage: {}".format(response.usage))
    return response.choices[0].message.content

def get_variable_topic(paper):
    system_prompt = "You are a scientist reviewing a paper."
    user_prompt = """
    You are given the abstract of a medical paper. Identify the key hypothesis, the associated variables,
    and their relationship from the abstract. Also breifly describe the paper topic in terms of the problem
    being addressed in the abstract.
    Your output should be structured as follows:
    RESPONSE:
    ```json
    <JSON>
    ```
    In <JSON>, respond in JSON format with ONLY the following field:
    - "Hypothesis": The key hypothesis in this abstract.
    - "Variable": The most important variable being studied in the abstract.
    - "Topic": What problem is this study trying to address or solve.
    This JSON will be automatically parsed, so ensure the format is precise.
    """
    #response = call_together(system_prompt, user_prompt)
    #response = call_local_llama(user_prompt, system_prompt)
    response = call_gpt(user_prompt, system_prompt)
    return response["Hypothesis"], response["Variable"], response["Topic"]


def main(source_paper, citing_papers):
    #system_message, user_message = ra_prompt_edited_for_validity(source_paper, citing_papers)
    system_message, user_message = ra_prompt_edited_for_validity_fewshot(source_paper, citing_papers)
    #print(user_message)
    response = run_model(pipe, system_message, user_message)
    return response[0]['generated_text'][-1]['content']

def parse_research_idea1(response):
    import re
    if response.strip().startswith("```json"):
        print("YESSSSSS")
        json_string = re.search("```json\s*(.*?)\s*```", response, re.DOTALL).group(1)
    else:
        json_string = response
    print(json_string)
    return json.loads(json_string)


def fix_json_syntax(output):
    """
    Fixes common JSON formatting issues, such as missing commas and extra brackets.
    """
    # Ensure missing commas before "Research idea" and "Hypothesis"
    output = re.sub(r'("Rationale":\s*".+?")\s*("Research idea")', r'\1,\n\2', output, flags=re.DOTALL)
    output = re.sub(r'("Research idea":\s*".+?")\s*("Hypothesis")', r'\1,\n\2', output, flags=re.DOTALL)
    
    # Remove any extra closing brackets at the end (common error)
    output = re.sub(r'\]\}$', '}', output)

    return output

def parse_research_idea(output):
    """Extracts and parses JSON content from the output, ensuring valid extraction."""
    
    try:
        # Fix JSON formatting issues (missing commas, misplaced brackets)
        output = fix_json_syntax(output)

        # Extract JSON content (Handling optional closing backticks)
        json_pattern = r"```json\s*(\{[\s\S]*?\})\s*```?"
        match = re.search(json_pattern, output, re.DOTALL)

        if match:
            json_content = match.group(1).strip()
        else:
            # Fallback: Try extracting any JSON-like structure
            json_pattern = r"(\{.*?\})"
            match = re.search(json_pattern, output, re.DOTALL)
            if match:
                json_content = match.group(1).strip()
            else:
                raise ValueError("No JSON found in the output")

        # Attempt to parse JSON
        parsed_data = json.loads(json_content)

        return {
            "Analysis": json.dumps(parsed_data.get("Analysis", {}), indent=2),
            "Rationale": parsed_data.get("Rationale", "Not Found"),
            "Research idea": parsed_data.get("Research idea", "Not Found"),
            "Hypothesis": parsed_data.get("Hypothesis", "Not Found"),
        }

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"⚠️ Attempting manual extraction...")
        return extract_manual_fields(output)
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return {"Error": "Parsing failed"}


import re
def extract_analysis_section(output):
    """
    Extracts everything from "Analysis" up to, but not including, "Rationale".
    """
    match = re.search(r'"Analysis":\s*\{(.*?)\}\s*,\s*"Rationale":', output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "Not Found"

def parse_research_idea2(output):
    """Extracts and cleans JSON from GPT response, ensuring proper parsing. Falls back to manual extraction if needed."""
    
    try:
        output = fix_json_syntax(output)
        # Check if output is empty or None
        if not output or not output.strip():
            raise ValueError("GPT output is empty")

        # Use regex to find the JSON block within triple backticks
        #json_pattern = r"```json\s*(\{.*?\})\s*```"
        json_pattern = r"```json\s*(\{.*?\})"
        match = re.search(json_pattern, output, re.DOTALL)

        if match:
            json_content = match.group(1).strip()
        else:
            # Fall back to extracting any JSON-like structure
            json_pattern = r"(\{.*?\})"
            match = re.search(json_pattern, output, re.DOTALL)
            if match:
                json_content = match.group(1).strip()
            else:
                raise ValueError("No JSON found in the output")

        # Remove trailing extra braces if needed
        while json_content.count("{") > json_content.count("}"):
            json_content += "}"

        while json_content.count("}") > json_content.count("{"):
            json_content = json_content[:-1]  # Remove last character

        # Attempt to parse the cleaned JSON
        parsed_data = json.loads(json_content)
        analysis_str = extract_analysis_section(output)
        
        # Ensure extraction of key fields while preserving "Analysis"
        return {
            "Analysis": analysis_str,
            "Rationale": parsed_data.get("Rationale", "Not Found"),
            "Research idea": parsed_data.get("Research idea", "Not Found"),
            "Hypothesis": parsed_data.get("Hypothesis", "Not Found")
        }

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"⚠️ Attempting manual extraction...")
        return extract_manual_fields(output)

    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return {}


def extract_manual_fields(output):
    """Manually extracts Analysis, Rationale, Research Idea, and Hypothesis from GPT output."""
    print("Attempting manual extraction...")
    print(output)
    analysis_match = re.search(r'"Analysis":\s*\{(.*?)\}\s*,\s*"Rationale":', output, re.DOTALL)
    rationale_match = re.search(r'"Rationale":\s*"([\s\S]*?)"\s*,\s*"Research idea":', output, re.DOTALL)
    research_idea_match = re.search(r'"Research idea":\s*"([\s\S]*?)"\s*,\s*"Hypothesis":', output, re.DOTALL)
    hypothesis_match = re.search(r'"Hypothesis":\s*"([\s\S]*?)"\s*}', output, re.DOTALL)

    analysis = analysis_match.group(1) if analysis_match else "Not Found"
    rationale = rationale_match.group(1) if rationale_match else "Not Found"
    research_idea = research_idea_match.group(1) if research_idea_match else "Not Found"
    hypothesis = hypothesis_match.group(1) if hypothesis_match else "Not Found"

    return {
        "Analysis": analysis,
        "Rationale": rationale,
        "Research idea": research_idea,
        "Hypothesis": hypothesis
    }

'''
def call_together(system_prompt, user_prompt):
    deployment_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    client = Together()
    response = client.chat.completions.create(
        model = deployment_name,
        messages = [{'role':'system', 'content': system_prompt}, 
        {'role':'user', 'content':user_prompt}],
        temperature = 0,
        seed = 42
    )
    return response.choices[0].message.content
'''
    
def call_local_llama(user_prompt, system_prompt = "You are a scientist reviewing a paper."):
    torch.cuda.empty_cache()  # Free up GPU memory
    llama_pipeline = load_llama_model()

    try:
        # Create the messages input
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Generate a response
        outputs = llama_pipeline(messages, max_new_tokens=2048, do_sample=True, temperature=0.7, top_p=0.9,)
        # Extract the assistant's response
        response = outputs[0]["generated_text"]

        return response
    except Exception as e:
        print(f"Error with LLaMA pipeline: {e}")
        return None

def aiscientist_novelty(hypothesis):
    query_prompt = novelty_query_prompt(hypothesis)
    #query = json.loads(call_together(query_prompt))["Query"]
    #query = json.loads(call_local_llama(query_prompt))["Query"]
    query = call_gpt(novelty_system_prompt(), query_prompt)
    print(f"Generated query: {query}")
    if "```json" in query:
        query = re.search("```json\s*(.*?)\s*```", query, re.DOTALL).group(1)
    query = json.loads(query)["Query"]
    print(query)
    docs = asyncio.run(call_semantic(query))
    #print(docs)
    docs = [x['text'] for x in docs]
    judgement_prompt = novelty_judgement_prompt(hypothesis, docs)
    #print(judgement_prompt)
    #novelty_res = call_together(novelty_system_prompt(), judgement_prompt)
    #novelty_res = call_local_llama(judgement_prompt, novelty_system_prompt())
    novelty_res = call_gpt(judgement_prompt, novelty_system_prompt())
    print(novelty_res)
    return {
        "query": query,
        "retrieved_docs": docs,
        "novelty_result": novelty_res
    }

# Function to load LLaMA model
def load_llama_model():
    try:
        model_id = "meta-llama/Llama-3.3-70B-Instruct"
        llama_pipeline = pipeline("text-generation", model=model_id, torch_dtype=torch.bfloat16,
            device_map="auto", model_kwargs={"use_cache": True})
        return llama_pipeline
    except Exception as e:
        print(f"Error loading LLaMA model: {e}")
        return None

    
if __name__ == "__main__":
    data_path = "./balanced_splits_w_hyp/test_hyp.json"
    meta_chains, chains, labels = load_dataset(data_path)
    #meta_chains, chains, labels = load_balanced_dataset(data_path)
    print(f"Total chains: {len(chains)}")
    
    # Load scoring protocol
    with open('./scoring_protocol.json', 'r') as f:
        scoring_protocol = json.load(f)

    metrics = scoring_protocol.keys()
    print("Evaluation Metrics:", metrics)
    
    import time

    MAX_RETRIES = 3  # Number of retries for generating research ideas
    RETRY_DELAY = 2  # Seconds to wait before retrying

    # Define checkpoint and output file paths
    checkpoint_file = "./hyper_checkpoint.json"
    output_file = "./phi3-hyper-test.json"

    # Load existing results if resuming
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            output_data = json.load(f)
    else:
        output_data = []

    # Track processed chains to avoid duplicates
    processed_chain_ids = {entry["metadata"]["file_path"] for entry in output_data}

    # Resume from last saved index
    last_saved_index = 0
    skipped_chains = set()  # Store skipped entries

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            checkpoint_data = json.load(f)
            last_saved_index = checkpoint_data.get("last_processed_index", 0)
            skipped_chains = set(checkpoint_data.get("skipped_chains", []))

    print(f"Resuming from chain {last_saved_index + 1}/{len(chains)}")

    for i, (chain, metadata) in enumerate(zip(chains, meta_chains)):
                    
        if i < last_saved_index or metadata["file_path"] in processed_chain_ids:
            continue  # Skip already processed chains

        print(f"Processing chain {i+1}/{len(chains)}...")

        for attempt in range(1, MAX_RETRIES + 1):
            avg_score = defaultdict(int)
            try:
                # Extract source paper and citing papers
                source_paper = chain[0]
                citing_papers = data_prompt(chain[1:])

                # **Retry this block if necessary**
                response = main(source_paper, citing_papers)
                research_idea = parse_research_idea(response)
                print(research_idea)
                
                # Ensure required fields exist
                if not research_idea or "Research idea" not in research_idea or research_idea["Research idea"] == "Not Found" or research_idea["Hypothesis"] == "Not Found" or research_idea["Hypothesis"] == "":
                    raise ValueError(f"Attempt {attempt}: Missing 'Research idea' for chain {i+1}: {response}. Retrying...")

                # Store results
                chain_results = {
                    "metadata": metadata,
                    "original_chain": chain,
                    "generated_research_idea": research_idea,
                    "evaluation_results": {},
                    "scores": "",
                    "novelty_analysis": {}
                }

                for _ in range(3):
                    # Evaluate research idea across metrics
                    for metric in metrics:
                        system_message, user_message = fewshot_reviewagent(scoring_protocol, metric, citing_papers, source_paper, research_idea) # or reviewagent
                        evaluation_result = call_gpt(user_message, system_message)
                        chain_results["evaluation_results"][metric] = json.loads(evaluation_result)
                        rating = chain_results["evaluation_results"][metric]["Rating (1-5) for Hypothesis"]
                        avg_score[metric] += int(rating)      
                        print(avg_score)
                        
                avg_score_dict = dict(avg_score)
                for metric in metrics:
                    avg_score_dict[metric] = float(avg_score_dict[metric] / 3.)

                print(avg_score_dict)
                chain_results['scores'] = avg_score_dict
                
                # Run novelty analysis
                hypothesis = research_idea["Hypothesis"]
                rag_novelty = aiscientist_novelty(hypothesis)                
                chain_results["novelty_analysis"] = rag_novelty
                    
                # Append new results without overwriting old ones
                output_data.append(chain_results)

                # Save progress **only after a successful chain processing**
                with open(output_file, "w") as f:
                    json.dump(output_data, f, indent=4)

                # Update checkpoint
                with open(checkpoint_file, "w") as f:
                    json.dump({
                        "last_processed_index": i,  
                        "skipped_chains": list(skipped_chains) 
                }, f)

                print(f"Chain {i+1} successfully processed and saved.")
                break  # **Exit retry loop on success**

            except Exception as e:
                print(f"Error on attempt {attempt}/{MAX_RETRIES} for chain {i+1}: {e}")
                if attempt < MAX_RETRIES:
                    print(f"Retrying after {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)  # Short delay before retrying
                else:
                    print(f"Skipping chain {i+1} after {MAX_RETRIES} failed attempts.")
                    skipped_chains.add(i)

    print(f"Processing completed. Results saved in {output_file}")
  