import os
import sys
sys.path.append('/srv/scratch1/rosni/scigen/')
import random
import json
import asyncio
from fuzzywuzzy import process
from ground_truth_path.utils import remove_numbering, search_papers
from ground_truth_path.RAG_temporal_extension_llama import load_global_llama, build_temporal_chain
from ground_truth_path.RAG_temporal_extension_llama import extract_linear_chain
REVIEW_ID = None
SEED = 42
IDX = 0
valid_chain_dir = "ground_truth_path/result_chains/"
intermediate_dir = "ground_truth_path/intermediate_chains/"
invalid_chain_dir = "ground_truth_path/invalid_chains_type2/"
os.makedirs(invalid_chain_dir, exist_ok=True)
skipped_file_path = os.path.join(invalid_chain_dir, "skipped_files.txt")
skipped_files = []

def initialize_globals(review_id, seed, idx):
    assert review_id is not None, "REVIEW_ID must be initialized"
    assert seed is not None, "SEED must be initialized"
    assert idx is not None, "IDX must be initialized"
    global REVIEW_ID, SEED, IDX
    REVIEW_ID = review_id
    SEED = seed  
    IDX = idx
    
def pick_non_adjacent_nodes(intermediate_papers):
    num_nodes = len(intermediate_papers)
    if num_nodes == 0:
        raise ValueError("No intermediate nodes to pick from.")
    if num_nodes < 3:
        index_1 = random.randint(0, num_nodes - 1)
        return [intermediate_papers[index_1]]
    
    # Select the first random index
    index_1 = random.randint(0, num_nodes - 1)
    # Define valid non-adjacent indices
    valid_indices = [i for i in range(num_nodes) if abs(i - index_1) > 1]
    # If no valid indices are found, return one node
    if not valid_indices:
        return [intermediate_papers[index_1]]
    # Randomly select a non-adjacent index
    index_2 = random.choice(valid_indices)
    print(f"Selected non-adjacent nodes: {index_1}, {index_2}")
    return [intermediate_papers[index_1], intermediate_papers[index_2]]


async def generate_valid_chain_until_2024(source_paper):
    global REVIEW_ID, SEED, IDX
    try:
        # Load the global Llama pipeline if not already loaded
        load_global_llama()
        with open("temporary_data/output.json", "r") as f:
            data = json.load(f)
        with open("temporary_data/gpt4_output", "r") as f:
            evaluations = json.load(f)
        
        few_shot_papers = data[1:3]
        few_shot_evaluations = list(evaluations.items())[1:3]

        few_shot_prompt = "Examples:\n\n"
        for idx, paper in enumerate(few_shot_papers):
            few_shot_prompt += f"{idx + 1}. Title: {paper['Title']} Abstract: {paper['Abstract']}; ({paper['Year']})\n"

        # Add example JSON evaluations
        few_shot_prompt += "\nExample evaluations in JSON format:\n```json\n{\n"
        for idx, (eval_title, eval_details) in enumerate(few_shot_evaluations):
            few_shot_prompt += f'    "{eval_title}": {{\n'
            few_shot_prompt += f'        "explanation": "{eval_details["explanation"]}",\n'
            few_shot_prompt += f'        "relevance": {eval_details["relevance"]}\n'
            few_shot_prompt += f'    }}'
            if idx < len(few_shot_evaluations) - 1:
                few_shot_prompt += ",\n"
        few_shot_prompt += "\n}\n```"
        
        # Build the hierarchical temporal chain
        full_chain = await build_temporal_chain(source_paper, few_shot_prompt, source_paper['year'], end_year=2024, REVIEW_ID=REVIEW_ID, SEED=SEED, IDX=IDX)

        # Extract and return the linear chain
        linear_chain = extract_linear_chain(full_chain)
        return linear_chain

    except Exception as e:
        print(f"Error generating valid chain for source paper {source_paper['title']}: {e}")
        return []
        
        
def negative_sampling(valid_chain_file, intermediate_file):
    with open(valid_chain_file, 'r') as f:
        temporal_chain = json.load(f)
    with open(intermediate_file, 'r') as f:
        intermediate_output = json.load(f) 
    result_dict = {}
    threshold = 98
    
    # Iterate through papers in the valid chain
    for paper in temporal_chain[1:-1]:
        paper_title = paper.get('title')
        paper_id = paper.get('paperId')
        if not paper_title:
            continue
        #print(f"Processing paper: {paper_title}")
        # Search in intermediate data for entries containing the paper title in `paper_list`
        for entry in intermediate_output:
            paper_list = entry.get('llama_output', {}).get('paper_list', {})
            paper_list_titles = list(paper_list.keys())
            best_match = process.extractOne(paper_title, paper_list_titles)  
            if best_match and best_match[1] >= threshold:  
                #print(f"Best Match: {best_match}")
                for title, details in paper_list.items():
                    if details.get('relevance') == 0:
                        key = (paper_title, paper_id)
                        cleaned_title = title.strip().lower()
                        if key not in result_dict:
                            result_dict[key] = []
                        result_dict[key].append({"title": cleaned_title,
                                                 "relevance": details.get('relevance'),
                                                 "explanation": details.get('explanation')
                                                 }) #todo: also add the relevancy_score and explanation?
                break
                        
    return result_dict 


async def create_hard_negative_chain_split(valid_chain_file, relevance_0_candidates, output_dir):
    with open(valid_chain_file, 'r') as f:
        valid_chain = json.load(f)

    # Skip chains of length 2
    if len(valid_chain) <= 2:
        print(f"Skipping chain with length <= 2: {valid_chain_file}")
        skipped_files.append(valid_chain_file)
        return

    first_paper = valid_chain[0]
    intermediate_papers = valid_chain[1:-1]

    split_replacements = {1: False, 2: False}
    # Attempt to find valid selected nodes with replaceable papers
    for attempt in range(5):  # Limit attempts to prevent infinite loops
        selected_nodes = pick_non_adjacent_nodes(intermediate_papers)
        print(selected_nodes)
        if not selected_nodes:
            print(f"No valid non-adjacent nodes found for {valid_chain_file}. Skipping file.")
            skipped_files.append(valid_chain_file)
            return

        # Check if replaceable papers are available for the selected nodes
        replaceable_papers = [
            node for node in selected_nodes
            if (node['title'], node['paperId']) in relevance_0_candidates and relevance_0_candidates[(node['title'], node['paperId'])]
        ]
   
        if replaceable_papers:
            break  # Valid combination found, proceed with the chain creation
        else:
            print(f"Attempt {attempt + 1}: No replaceable papers for selected nodes. Retrying...")
            continue

    # If after all attempts no valid combination is found, skip the file
    if not replaceable_papers:
        print(f"No replaceable papers with relevance 0 candidates after multiple attempts in {valid_chain_file}. Skipping file.")
        skipped_files.append(valid_chain_file)
        return 

    for split_index, selected_node in enumerate(selected_nodes, start=1):
        invalid_chain = [first_paper]  # Start a new invalid chain for the current split
        valid_chain_created = False
        for paper in intermediate_papers:
            if paper == selected_node:
                # Replace the selected paper with a relevance 0 candidate
                key = (paper['title'], paper['paperId'])
                replacement_candidates = relevance_0_candidates.get(key, [])
                if replacement_candidates:
                    # Select a random irrelevant paper
                    replacement_choice = random.choice(replacement_candidates)
                    replacement_title = replacement_choice["title"]
                    explanation = replacement_choice["explanation"]
                    relevance = replacement_choice["relevance"]

                    # Find a valid replacement paper using the title
                    replacement_title = remove_numbering(replacement_title)
                    replacement_paper = search_papers(replacement_title)

                    if replacement_paper and len(replacement_paper) > 0:
                        replacement_node = {
                            "paperId": replacement_paper[0]["paperId"],
                            "title": replacement_paper[0]["title"],
                            "abstract": replacement_paper[0].get("abstract"),
                            "year": replacement_paper[0].get("year"),
                            "citation_count": replacement_paper[0].get("citationCount"),
                            "relevance": relevance,
                            "explanation": explanation
                        }
                        #invalid_chain.append(replacement_node)
                        split_replacements[split_index] = True
                        # Generate a valid chain from this node till 2024
                        valid_subchain = await generate_valid_chain_until_2024(replacement_node)
                        if not valid_subchain:
                            print(f"Warning: Failed to generate valid chain from {replacement_node['title']} in {valid_chain_file}. Skipping this split.")
                            break 
                        invalid_chain.extend(valid_subchain)
                        valid_chain_created = True
                        break  # End processing for this split once the valid chain is created
                    else:
                        # Fallback: Keep the original paper if no valid replacement found
                        invalid_chain.append(paper)
                else:
                    # If no replacement candidates, keep the paper as is
                    invalid_chain.append(paper)
            else:
                # Keep the paper if it's not the selected node
                invalid_chain.append(paper)

        # If no valid chain was created, skip saving this split
        if not valid_chain_created:
            print(f"No valid chain created for split {split_index} in {valid_chain_file}. Skipping.")
            continue

        # Save the invalid chain for the current split
        split_suffix = f"split_{split_index}"
        base_filename = os.path.basename(valid_chain_file)
        filename_without_extension = os.path.splitext(base_filename)[0]
        invalid_filename = f"{filename_without_extension}_hard_negative_{split_suffix}_with_valid_chain.json"
        invalid_file_path = os.path.join(output_dir, invalid_filename)

        with open(invalid_file_path, 'w') as f:
            json.dump(invalid_chain, f, indent=4)

        print(f"Hard negative chain with {split_suffix} saved: {invalid_file_path}")


def main():
    global skipped_files
    processed_identifiers = set()

    for processed_file in os.listdir(invalid_chain_dir):
        if "_hard_negative_split_" in processed_file and processed_file.endswith("_with_valid_chain.json"):
            # Extract everything from the beginning up to `p-1`
            identifier = processed_file.split("_hard_negative_split_")[0]
            processed_identifiers.add(identifier)
        
    for valid_chain_file in os.listdir(valid_chain_dir):
        if valid_chain_file.endswith(".json"):
            identifier = os.path.splitext(valid_chain_file)[0]  # Remove `.json`
            if identifier in processed_identifiers:
                print(f"Skipping already processed file: {valid_chain_file}")
                continue
            print(f"Processing file: {valid_chain_file}")
            
            review_id = os.path.splitext(valid_chain_file)[0].split('_')[2]
            valid_chain_file_path = os.path.join(valid_chain_dir, valid_chain_file)
            intermediate_file_path = os.path.join(intermediate_dir, f"llama_outputs_{review_id}_p-1_log.json")
            initialize_globals(review_id, 42, 0)
            if not os.path.exists(intermediate_file_path):
                print(f"Intermediate file for {review_id} not found!")
                continue

            relevance_0_candidates = negative_sampling(valid_chain_file_path, intermediate_file_path)
            asyncio.run(create_hard_negative_chain_split(valid_chain_file_path, relevance_0_candidates, invalid_chain_dir))
            
    # Save skipped files to a text file
    with open(skipped_file_path, 'a') as f:
        for skipped_file in skipped_files:
            f.write(f"{skipped_file}\n")

    print(f"Skipped files saved to {skipped_file_path}")


if __name__ == "__main__":
    main()
