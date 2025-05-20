# task details
# hyper-1-hop
### ins3: source, target paper -> relevancy score (lot more data points)

# hyper-multi-hop2_c1
### ins1: chain -> valid, invalid, if invalid identify invalid links,

# hyper-multi-hop2_c2
### ins2: chain, hypothesis -> valid, invalid, if invalid identify invalid links

import sys
sys.path.append('add project path here')
import os
import json
import time

# Paths to the original dataset splits
DATASET_PATH = "training_data/balanced_splits_w_hyp"
OUTPUT_PATH = "./training_data/multi_task_data3/"

# Ensure the output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

def format_reasoning_chain(content):
    """
    Formats the reasoning chain as a numbered list with paper title and abstract.
    """
    formatted_chain = []
    for i, paper in enumerate(content, start=1):
        title = paper.get("title", "No Title Provided")
        abstract = paper.get("abstract", "No Abstract Provided")
        year = paper.get("year", "No Year Provided")
        formatted_chain.append(f"{i}. PAPER TITLE: {title}\n   ABSTRACT: {abstract}\nYear: {year}")
    return "\n".join(formatted_chain)

def remove_2024_papers(data):
    """
    Removes papers published in 2024 from the reasoning chains in the dataset.
    """
    updated_data = []
    for chain in data:
        chain["content"] = [paper for paper in chain["content"] if paper.get("year") != 2024]
        if chain["content"]:
            updated_data.append(chain)
    return updated_data

CACHE_FILE = "paper_cache.json"

# Load cache from file
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

# Save updated cache to file
def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)


def get_variable_topic(paper):
    system_prompt = "You are a scientist reviewing a paper."
    user_prompt = f"""
    You are given the abstract of a medical paper. Identify the key hypothesis, the associated variables,
    and their relationship from the abstract. Also breifly describe the paper topic in terms of the problem
    being addressed in the abstract. 
    The paper details are as follows:
    {paper}
    Your output should be structured as follows:
    RESPONSE:
    ```json
    <JSON>
    ```
    In <JSON>, respond in JSON format with ONLY the following field:
    - "Hypothesis": The key hypothesis in this study.
    - "Variable": The most important variable being studied in the study.
    - "Topic": What problem is this study trying to address or solve.
    This JSON will be automatically parsed, so ensure the format is precise.
    """
    cache = load_cache()

    # Check if paper is already analyzed
    if paper in cache:
        print("Using cached result from JSON file.")
        return cache[paper]

    time.sleep(2)  # Simulating API delay
    response = call_gpt(user_prompt, system_prompt)  # Replace with actual GPT call

    try:
        response_data = json.loads(response)
    except json.JSONDecodeError:
        print("Error: JSON decoding failed. Returning empty values.")
        return {"Hypothesis": "", "Variable": "", "Topic": ""}

    result = {
        "Hypothesis": response_data.get("Hypothesis", ""),
        "Variable": response_data.get("Variable", ""),
        "Topic": response_data.get("Topic", "")
    }

    # Save result to cache
    cache[paper] = result
    save_cache(cache)

    return result

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

def ra_prompt_edited_for_validity(source_paper:dict, citing_paper_list:str):
    role = "an AI assistant"
    system_message = f"""You are {role} whose primary goal is to identify promising, new, and key scientific
problems based on existing scientific literature, in order to aid researchers in discovering novel
and significant research opportunities that can advance the field."""
    user_message = f"""You are going to generate a research problem that should be original, clear, feasible, relevant, and
significant to its field. This will be based on the title and abstract of the source paper, those of
{len(citing_paper_list)} related papers in the existing literature.
Understanding of the target paper, and the related papers is essential:
- The source paper is the primary research study you aim to enhance or build upon through future
research, serving as the central source and focus for identifying and developing the specific
research problem.
- The related papers are arranged in temporal order of citation, such that paper 2 cites paper 1 and 
paper 3 cites paper 2 and so on. The relevant papers provide additional context and insights that are essential for 
understanding and expanding upon the target paper. However, all the papers in the list may not be relevant to the primary 
research you are focusing on. 
Your approach should be systematic:
- Start by thoroughly reading the title and abstract of the source paper to understand its core focus.
- Next, proceed to read the titles and abstracts of the related papers in the order in which they appear in the list.
Identify the papers that form a logical reasoning chain starting from the source paper.
- Use only these papers to gain a broader perspective about the progression of the primary research topic over time. 
I am going to provide the source paper and related papers as an enumerated list of Title, Abstract and Year of publication 
triple, as follows:
Source paper title: {source_paper['title']}
Source paper abstract: {source_paper['abstract']}
Source paper year of publication: {source_paper['year']}
Related papers: {citing_paper_list}
With the provided source paper, and the related papers, your objective now is to formulate a
research problem that not only builds upon these existing studies but also strives to be original,
clear, feasible, relevant, and significant. Before crafting the research problem, revisit the title
and abstract of the target paper, to ensure it remains the focal point of your research problem
identification process. 
Now convert this idea into a concrete testable hypothesis. Remember hypothesis is a declarative statement expressing a 
relationship between two variables like independent or dependent variables or left group and rigt group in a given context.
Your hypothesis should contain the key variable or variables from your research idea.
Source paper title: {source_paper['title']}
Source paper abstract: {source_paper['abstract']}
Then, following your review of the above content, please proceed to analyze the progression of the research topic. Now output this analysis, the research idea and hypothesis with the rationale.
Your output should be a valid JSON with the following fields.  
Output a JSON object in the following format:
```json
{{
  "Analysis": {{Output a dictionary with each paper in the Related Papers as a key. For each key analyze how this paper builds upon the previous papers in the list. For example, how Paper 0 builds upon source paper and Paper 1 builds upon the concepts in Paper 0 and so on. Elaborate on specific advancements made, including the explanation behind their effectiveness in addressing previous challenges. Apply this analytical approach to each valid paper in the sequence. Ignore papers that do not build upon the previous papers and diverge from the original source paper's topic significantly.}},
  "Rationale": "Summarize the above analysis and explain how you would come up with a research idea that will advance the field of work while addressing the limitations of previous work and building upon the existing work.",
  "Research idea": "Delineate an elaborate research problem here including the key variables.",
  "Hypothesis": "Provide a concrete testable hypothesis that follows from the above research problem here"
}}
```
This JSON will be automatically parsed, so ensure the format is precise.
"""
    return system_message, user_message


import random

# Process One-Hop Relevance Task (hyper-1-hop) with priority logic
def process_one_hop_relevance(content):
    """
    Process the one-hop relevance task by prioritizing pairs with relevance 0.
    If no such pairs exist, randomly pick a pair with relevance 1 or 2.
    """
    tasks = []
    relevance_zero_pairs = []
    relevance_other_pairs = []
    
    for i in range(len(content) - 1):  # Loop through source-target pairs
        source_paper = content[i]
        target_paper = content[i + 1]
        relevance = target_paper.get("relevance", -1)

        # Separate pairs based on relevance score
        if relevance == 0:
            relevance_zero_pairs.append((source_paper, target_paper))
        else:
            relevance_other_pairs.append((source_paper, target_paper))

    # Select the priority pair(s)
    selected_pairs = relevance_zero_pairs if relevance_zero_pairs else random.sample(relevance_other_pairs, 1) if relevance_other_pairs else []

    # Create task entries
    for source_paper, target_paper in selected_pairs:
        tasks.append({
            "task_name": "hyper-1-hop",
            "instruction": "Hypotheses are frequently the starting point when undertaking the empirical portion of the scientific process. They state something that the scientific process will attempt to evaluate, corroborate, verify, or falsify. Their purpose is to guide the types of data we collect, analyses we conduct, and inferences we would like to make. You are a scientist. Your task is to evaluate the relevance of a target paper to a source paper in a one-hop citation context. You are given a source paper and a target paper that followed from it. Your job is to determine the degree of relevance between these papers based on scientific progression, logical dependence, and hypothesis inspiration.\n\nAssign a relevance score to the target paper based on the following criteria:\n- 0: No meaningful connection between the target paper and the source paper. This includes review papers (e.g., systematic reviews, meta-analyses) that summarize literature without introducing novel hypotheses or findings.\n- 1: The key hypothesis in the target paper is inspired by the hypothesis or findings from the source paper.\n- 2: The key hypothesis in the target paper directly depends on the findings of the source paper. This means the source paper contains sub-hypotheses or foundational results that contribute directly to the target paper's hypothesis.\n\nYour task:\n1. Assign a relevance score (0, 1, or 2) based on the relationship between the source and target paper.\nAnswer must follow the example output format in JSON:\n```json\n{ \"relevance_score\": <0, 1, or 2> }\n```\n DO NOT include any additional text, comments, explanations, or formatting variations.\n ONLY return the JSON object exactly as required.",
            "input": {
                "source_paper": source_paper,
                "target_paper": target_paper
            },
            "output": {
                "relevance_score": target_paper.get("relevance", -1)
            }
        })
    
    return tasks

# Function to process each dataset file
def process_dataset(file_path):
    is_valid = False
    with open(file_path, "r") as f:
        data = json.load(f)
    
    if 'valid_hyp.json' in file_path:
        is_valid = True
    
    if "train_hyp.json" in file_path:
        data = remove_2024_papers(data)

    processed_data = []
    
    for entry in data:
        file_name = entry["file_name"]
        chain_label = entry["chain_label"]
        label = "invalid" if entry["chain_label"] in {"invalid_type1", "invalid_type2"} else entry["chain_label"]
        file_path = entry["file_path"]
        # remove 2024 data?
        content = entry["content"]
        target_hypothesis_data = entry.get("target_hypothesis", {})
        if isinstance(target_hypothesis_data, dict):
            target_hypothesis_list = target_hypothesis_data.get("Target Hypotheses", ["NA"])
        else:
            target_hypothesis_list = ["NA"]  
        target_hypothesis= " ".join([f"{i+1}. {hypothesis}" for i, hypothesis in enumerate(target_hypothesis_list)])
        
        # Initialize tasks
        tasks = []
        
        # Process One-Hop Relevance Task (hyper-1-hop) with priority logic
        tasks.extend(process_one_hop_relevance(content))
            
        # Process Multi-Hop Reasoning Chain Tasks
        formatted_chain = format_reasoning_chain(content)

        '''
        # hyper-multi-hop1_c1 (Chain -> Valid/Invalid)
        tasks["hyper-multi-hop1_c1"].append({
            "instruction": "Hypotheses are frequently the starting point when undertaking the empirical portion of the scientific process. They guide the types of data collected, analyses conducted, and inferences drawn. You are a scientist. Your task is to evaluate the relevance of a reasoning chain derived from a series of related papers. Each paper in the chain builds on the previous one, starting from a source paper and ending with a final paper. A reasoning chain is considered 'valid' if the logical connections between all the papers in the chain are coherent, and the final hypothesis logically inspired from or depends on the source paper, either directly or through intermediate papers. A reasoning chain is 'invalid' if any logical break exists in the chain, such as unrelated intermediate papers, or  if the final hypothesis does not logically depend on the previous papers. Your task is to classify the given reasoning chain as either 'valid' or 'invalid'. Your evaluation should be based on the logical relevance and progression between the papers.  Answer should be only 'valid' or 'invalid'.",
            "input": {"reasoning_chain": formatted_chain},
            "output": {"validity": label}
        })
        '''
        # hyper-multi-hop2_c1 (Chain -> Valid/Invalid + Invalid Paper IDs)
        if label == "invalid":
            invalid_papers = [pid + 1 for pid, p in enumerate(content) if p.get("relevance") == 0]
            tasks.append({
                "task_name": "hyper-multi-hop2_c1",
                "instruction": "Hypotheses are frequently the starting point when undertaking the empirical portion of the scientific process. They guide the types of data collected, analyses conducted, and inferences drawn. You are a scientist. Your task is to evaluate the relevance of a reasoning chain derived from a series of related papers. Each paper in the chain builds on the previous one, starting from a source paper and ending with a final paper. \n\nA reasoning chain is considered 'valid' if:\n- All papers in the sequence are logically connected.\n- The final hypothesis in the last paper is logically inspired by or dependent on the source paper, either directly or through intermediate papers.\n\nA reasoning chain is considered 'invalid' if:\n- if any logical break exists in the chain, such as unrelated intermediate papers, or  if the final hypothesis does not logically depend on the previous papers.\n\nYour task:\n1. Determine whether the reasoning chain is 'valid' or 'invalid'.\n2. If invalid, return a list of paper IDs that break logical progression in the chain.\n\nAnswer must follow the example output format in JSON:\n- If the chain is valid, return:\n  ```json\n  { \"validity\": \"valid\", \"invalid_paper_ids\": [] }\n  ```\n- If the chain is invalid, return:\n  ```json\n  { \"validity\": \"invalid\", \"invalid_paper_ids\": [3, 4, 7] }\n  ```\n DO NOT include any additional text, comments, explanations, or formatting variations.\n ONLY return the JSON object exactly as required.",
                "input": {"reasoning_chain": formatted_chain},
                "output": {"validity": "invalid", "invalid_paper_ids": invalid_papers}
            })
        else:
            tasks.append({
                "task_name": "hyper-multi-hop2_c1",
                "instruction": "Hypotheses are frequently the starting point when undertaking the empirical portion of the scientific process. They guide the types of data collected, analyses conducted, and inferences drawn. You are a scientist. Your task is to evaluate the relevance of a reasoning chain derived from a series of related papers. Each paper in the chain builds on the previous one, starting from a source paper and ending with a final paper. \n\nA reasoning chain is considered 'valid' if:\n- All papers in the sequence are logically connected.\n- The final hypothesis in the last paper is logically inspired by or dependent on the source paper, either directly or through intermediate papers.\n\nA reasoning chain is considered 'invalid' if:\n- if any logical break exists in the chain, such as unrelated intermediate papers, or  if the final hypothesis does not logically depend on the previous papers.\n\nYour task:\n1. Determine whether the reasoning chain is 'valid' or 'invalid'.\n2. If invalid, return a list of paper IDs that break logical progression in the chain.\n\nAnswer must follow the example output format in JSON:\n- If the chain is valid, return:\n  ```json\n  { \"validity\": \"valid\", \"invalid_paper_ids\": [] }\n  ```\n- If the chain is invalid, return:\n  ```json\n  { \"validity\": \"invalid\", \"invalid_paper_ids\": [3, 4, 7] }\n  ```\n DO NOT include any additional text, comments, explanations, or formatting variations.\n ONLY return the JSON object exactly as required.",
                "input": {"reasoning_chain": formatted_chain},
                "output": {"validity": "valid", "invalid_paper_ids": []}
            })
        
        '''
        # hyper-multi-hop1_c2 (Chain + Hypothesis -> Valid/Invalid)
        tasks["hyper-multi-hop1_c2"].append({
            "instruction": "Hypotheses are frequently the starting point when undertaking the empirical portion of the scientific process. They guide the types of data collected, analyses conducted, and inferences drawn. You are a scientist. Your task is to evaluate the relevance of a reasoning chain derived from a series of related papers. Each paper in the chain builds on the previous one, starting from a source paper and ending with a final paper. A reasoning chain is considered 'valid' if the logical connections between all the papers in the chain are coherent, and the final hypotheses from the final paper logically inspired from or depends on the source paper, either directly or through intermediate papers. A reasoning chain is 'invalid' if any logical break exists in the chain, such as unrelated intermediate papers, or  if the final hypotheses do not logically depend on the previous papers. If target hypotheses is provided, the chain's validity should be evaluated based on whether it logically supports the target hypotheses. Your task is to classify the given reasoning chain as either 'valid' or 'invalid'. Your evaluation should be based on the logical relevance and progression between the papers.  Answer should be only 'valid' or 'invalid'.",
            "input": {"reasoning_chain": formatted_chain, "target_hypotheses": target_hypothesis},
            "output": {"validity": label}
        })
        '''
        
        # hyper-multi-hop2_c2 (Chain + Hypothesis -> Valid/Invalid + Invalid Paper IDs)
        if label == "invalid":
            if len(content) > 1:
                formatted_chain = format_reasoning_chain(content[:-1])
            invalid_papers = [pid + 1 for pid, p in enumerate(content) if p.get("relevance") == 0]
            tasks.append({
                "task_name": "hyper-multi-hop2_c2",
                "instruction": "Hypotheses are frequently the starting point when undertaking the empirical portion of the scientific process. They guide the types of data collected, analyses conducted, and inferences drawn. You are a scientist. Your task is to evaluate the relevance of a reasoning chain derived from a series of related papers. Each paper in the chain builds on the previous one, starting from a source paper and ending with a final paper.\n\nA reasoning chain is considered 'valid' if:\n- All papers in the sequence are logically connected.\n- The final hypotheses in the last paper is logically inspired by or dependent on the source paper, either directly or through intermediate papers.\n- The reasoning chain logically supports the given target hypotheses.\n\nA reasoning chain is considered 'invalid' if:\n- Any logical break exists in the chain, such as unrelated intermediate papers, or if the final hypotheses do not logically depend on the previous papers.\n- The reasoning chain does not logically support the given target hypotheses.\n\nYour task:\n1. Determine whether the reasoning chain supports the target hypotheses and is 'valid' or 'invalid'.\n2. If invalid, return a list of paper IDs that break logical progression in the chain.\n\nAnswer must follow the example output format in JSON:\n- If the chain is valid, return:\n  ```json\n  { \"validity\": \"valid\", \"invalid_paper_ids\": [] }\n  ```\n- If the chain is invalid, return:\n  ```json\n  { \"validity\": \"invalid\", \"invalid_paper_ids\": [3, 4, 7] }\n  ```\n DO NOT include any additional text, comments, explanations, or formatting variations.\n ONLY return the JSON object exactly as required.",
                "input": {"reasoning_chain": formatted_chain, "target_hypotheses": target_hypothesis},
                "output": {"validity": "invalid", "invalid_paper_ids": invalid_papers}
            })
        else:
            if len(content) > 1:
                formatted_chain = format_reasoning_chain(content[:-1])
            tasks.append({
                "task_name": "hyper-multi-hop2_c2",
                "instruction": "Hypotheses are frequently the starting point when undertaking the empirical portion of the scientific process. They guide the types of data collected, analyses conducted, and inferences drawn. You are a scientist. Your task is to evaluate the relevance of a reasoning chain derived from a series of related papers. Each paper in the chain builds on the previous one, starting from a source paper and ending with a final paper.\n\nA reasoning chain is considered 'valid' if:\n- All papers in the sequence are logically connected.\n- The final hypotheses in the last paper is logically inspired by or dependent on the source paper, either directly or through intermediate papers.\n- The reasoning chain logically supports the given target hypotheses.\n\nA reasoning chain is considered 'invalid' if:\n- Any logical break exists in the chain, such as unrelated intermediate papers, or if the final hypotheses do not logically depend on the previous papers.\n- The reasoning chain does not logically support the given target hypothesis.\n\nYour task:\n1. Determine whether the reasoning chain supports the target hypotheses and is 'valid' or 'invalid'.\n2. If invalid, return a list of paper IDs that break logical progression in the chain.\n\nAnswer must follow the example output format in JSON:\n- If the chain is valid, return:\n  ```json\n  { \"validity\": \"valid\", \"invalid_paper_ids\": [] }\n  ```\n- If the chain is invalid, return:\n  ```json\n  { \"validity\": \"invalid\", \"invalid_paper_ids\": [3, 4, 7] }\n  ```\n DO NOT include any additional text, comments, explanations, or formatting variations.\n ONLY return the JSON object exactly as required.",
                "input": {"reasoning_chain": formatted_chain, "target_hypotheses": target_hypothesis},
                "output": {"validity": "valid", "invalid_paper_ids": []}
            })
        
        if is_valid:
            print("True")
            formatted_chain = format_reasoning_chain(content[1:-1])
            paper=  f"Title: {content[-1]['title']}; \nAbstract: {content[-1]['abstract']}; \n\n"
            hypothesis = get_variable_topic(paper)
            print(hypothesis)
            system_prompt, user_prompt = ra_prompt_edited_for_validity(content[0], formatted_chain)
            tasks.append({
                "task_name": "hyper-hg",
                "instruction": system_prompt,
                "input": {"reasoning_chain": user_prompt},
                "output": {"target_hypotheses": hypothesis["Hypothesis"]}
            })

        
        # Save structured entry
        for task in tasks:
            processed_data.append({
                "file_name": file_name,
                "chain_label": chain_label,
                "file_path": file_path,
                "task_name": task["task_name"],
                "instruction": task["instruction"],
                "input": task["input"],
                "output": task["output"],
                "generated_from_split": False
            })
    
    return processed_data

# Process all dataset splits
splits = ["train_hyp.json", "valid_hyp.json", "test_hyp.json"]

for split in splits:
    input_path = os.path.join(DATASET_PATH, split)
    processed_output = process_dataset(input_path)

    # Save the processed dataset
    output_file = os.path.join(OUTPUT_PATH, split)
    with open(output_file, "w") as f:
        json.dump(processed_output, f, indent=2)

    print(f"Processed and saved: {output_file}")
