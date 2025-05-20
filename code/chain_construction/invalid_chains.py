import os
import sys
sys.path.append('path to project/')
import random
import json
from fuzzywuzzy import process
from ground_truth_path.utils import remove_numbering, search_papers

valid_chain_dir = "dataset/result_chains/"
intermediate_dir = "dataset/intermediate_chains/"
invalid_chain_dir = "dataset/invalid_chains_type1/"
os.makedirs(invalid_chain_dir, exist_ok=True)
skipped_file_path = os.path.join(invalid_chain_dir, "skipped_files.txt")
skipped_files = []

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

def create_multiple_invalid_chains(valid_chain_file, relevance_0_candidates, output_dir):

    with open(valid_chain_file, 'r') as f:
        valid_chain = json.load(f)

    # Skip chains of length 2
    if len(valid_chain) <= 2:
        print(f"Skipping chain with length <= 2: {valid_chain_file}")
        skipped_files.append(valid_chain_file)
        return

    first_paper = valid_chain[0]
    last_paper = valid_chain[-1]
    intermediate_papers = valid_chain[1:-1]
    num_intermediate_papers = len(intermediate_papers)

    # Calculate maximum replacements (up to 50% of intermediate nodes)
    max_replacements = max(1, num_intermediate_papers // 2)

    for num_to_replace in range(1, max_replacements + 1):
        # Filter papers that have relevance 0 candidates
        replaceable_papers = [
            paper for paper in intermediate_papers
            if (paper['title'], paper['paperId']) in relevance_0_candidates
        ]

        # If no replaceable papers, skip this level of replacement
        if not replaceable_papers:
            print(f"No replaceable papers with relevance 0 candidates in {valid_chain_file}")
            skipped_files.append(valid_chain_file)
            return

        # Ensure the number of replacements does not exceed the available candidates
        num_to_replace = min(num_to_replace, len(replaceable_papers))

        # Select nodes to replace from replaceable_papers
        papers_to_replace = random.sample(replaceable_papers, num_to_replace)

        invalid_chain = [first_paper]  # Start the invalid chain with the first paper

        for paper in intermediate_papers:
            if paper in papers_to_replace:
                key = (paper['title'], paper['paperId'])
                replacement_candidates = relevance_0_candidates[key]
                replacement_choice = random.choice(replacement_candidates)
                replacement_title = replacement_choice["title"]
                explanation = replacement_choice["explanation"]
                relevance = replacement_choice["relevance"]

                # Attempt to find a valid replacement paper
                replacement_title = remove_numbering(replacement_title)
                replacement_paper = search_papers(replacement_title)

                if replacement_paper and len(replacement_paper) > 0:
                    invalid_chain.append({
                        "paperId": replacement_paper[0]["paperId"],
                        "title": replacement_paper[0]["title"],
                        "abstract": replacement_paper[0].get("abstract"),
                        "year": replacement_paper[0].get("year"),
                        "citation_count": replacement_paper[0].get("citationCount"),
                        "relevance": relevance,
                        "explanation": explanation
                    })
                else:
                    # Fallback to the original paper if no valid replacement is found
                    invalid_chain.append(paper)
            else:
                invalid_chain.append(paper)  # No replacement for this paper

        invalid_chain.append(last_paper)  # Add the last paper to the chain

        # Save the invalid chain with the replacement count in the filename
        base_filename = os.path.basename(valid_chain_file)
        filename_without_extension = os.path.splitext(base_filename)[0]
        invalid_filename = f"{filename_without_extension}_invalid_{num_to_replace}_replacements.json"
        invalid_file_path = os.path.join(output_dir, invalid_filename)

        with open(invalid_file_path, 'w') as f:
            json.dump(invalid_chain, f, indent=4)

        print(f"Invalid chain with {num_to_replace} replacements saved: {invalid_file_path}")

def main():
    global skipped_files
    
    for valid_chain_file in os.listdir(valid_chain_dir):
        if valid_chain_file.endswith(".json"):
            review_id = os.path.splitext(valid_chain_file)[0].split('_')[2]
            valid_chain_file_path = os.path.join(valid_chain_dir, valid_chain_file)
            intermediate_file_path = os.path.join(intermediate_dir, f"llama_outputs_{review_id}_p-1_log.json")

            if not os.path.exists(intermediate_file_path):
                print(f"Intermediate file for {review_id} not found!")
                continue

            relevance_0_candidates = negative_sampling(valid_chain_file_path, intermediate_file_path)
            create_multiple_invalid_chains(valid_chain_file_path, relevance_0_candidates, invalid_chain_dir)
            
    # Save skipped files to a text file
    with open(skipped_file_path, 'a') as f:
        for skipped_file in skipped_files:
            f.write(f"{skipped_file}\n")

    print(f"Skipped files saved to {skipped_file_path}")


if __name__ == "__main__":
    main()
