import os
# add gpu support
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
from dotenv import load_dotenv
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
from huggingface_hub import login
login(token=huggingface_token)
import json
import sys
sys.path.append('path to project')
import asyncio
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("S2_API_KEY", None)
from langdetect import detect, DetectorFactory, LangDetectException
DetectorFactory.seed = 0
from utils import clean_gpt_output, evaluate_papers_with_llama, load_llama_model
from utils import get_paper_data, get_paper_by_id, relevancy_prompt
from utils import remove_numbering, split_into_chunks, search_papers, extract_linear_chain, save_json
import torch.nn.functional as F
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils import set_seed
import argparse
import logging
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"pipeline_llama_HN.log")
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
SEED = None
REVIEW_ID = None
IDX = None
llama_pipeline = None
saved_top_papers = False

def parse_args():
    parser = argparse.ArgumentParser(description="Run script with a specific seed.")
    parser.add_argument("--seed", type=int, required=True, help="Seed value for the run.")
    parser.add_argument("--review_id", type=str, required=True, help="Review ID to process.")
    return parser.parse_args()

def set_global_seed(seed):
    """Set the global seed."""
    global SEED
    SEED = seed
    set_seed(seed)
    logging.info(f"Global seed set to: {SEED}")

def initialize_globals(review_id, seed, idx):
    """Initialize globals for the first time."""
    global REVIEW_ID, SEED, IDX
    REVIEW_ID = review_id
    SEED = seed  
    IDX = idx
    
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"pipeline_llama_HN.log")
    logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
      
def load_global_llama():
    global llama_pipeline
    if llama_pipeline is None:
        llama_pipeline = load_llama_model()

def log_to_json_file(data, filename="./intermediate_chains_HN/llama_outputs.json"):
    """Append data to a JSON file."""
    global REVIEW_ID
    filename = filename.replace(".json", f"_{REVIEW_ID}_p-{IDX}_log.json")
    try:
        if os.path.exists(filename):
            with open(filename, "r") as infile:
                existing_data = json.load(infile)
        else:
            existing_data = []
        data["seed"] = SEED
        
        existing_data.append(data)
        with open(filename, "w") as outfile:
            json.dump(existing_data, outfile, indent=4)
    except Exception as e:
        logging.error(f"Error saving to JSON file: {e}")
        
def filter_papers(citation_details, current_year):
    year_wise_citations = {}
    for cited_paper in citation_details.get("citations", []):
        cited_year = cited_paper.get("year")
        title = cited_paper.get("title", "")
        abstract = cited_paper.get("abstract", "")
        citation_count = cited_paper.get("citationCount", None)

        if cited_year and abstract and (cited_year > current_year):
            title_is_english = detect(title) == 'en' if title else False
            valid_year = cited_year is not None and isinstance(cited_year, int)
            valid_citation_count = (cited_year >= current_year) or (citation_count is not None)
            
            if cited_year not in [2023, 2024] and cited_paper.get("citationCount", 0) <= 0:
                continue
            
            if title_is_english and valid_year and valid_citation_count:
                if cited_year not in year_wise_citations:
                    year_wise_citations[cited_year] = []
                year_wise_citations[cited_year].append({
                    "paperId": cited_paper.get("paperId"),
                    "title": cited_paper.get("title"),
                    "abstract": cited_paper.get("abstract", None),
                    "citationCount": cited_paper.get("citationCount", 0)
                })

    return year_wise_citations
 
def parse_llama_output(response):
    llama_output = {}
    for item in response:
        if item['role'] == 'assistant':
            try:
                return clean_gpt_output(item['content'])
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing LLAMA output: {e}")
                logging.debug(f"Problematic LLAMA content: {item['content']}")
    return llama_output

def extract_top_papers(llama_output):
    top3_papers = llama_output.get("top3_relevant_papers", [])
    logging.info(f"Top 3 papers: {top3_papers}")
    extracted_papers = []

    for top_paper in top3_papers:
        top = remove_numbering(top_paper)
        result = search_papers(top)
        if result:
            paper = {
                "paperId": result[0]["paperId"],
                "title": result[0]["title"],
                "abstract": result[0]["abstract"],
                "year": result[0]["year"],
                "citation_count": result[0]["citationCount"],
                "relevance": top3_papers[top_paper]["relevance"],
                "explanation": top3_papers[top_paper]["explanation"]
            }
            extracted_papers.append(paper)
        else:
            logging.warning(f"Paper not found: {top_paper}")
    return extracted_papers

def process_yearly_citations(year, year_wise_citations, current_paper, few_shot_prompt, llama_pipeline):
    if year not in year_wise_citations:
        return []

    chunks = split_into_chunks(year_wise_citations[year], 10)
    relevant_papers_for_year = []

    for chunk_index, chunk in enumerate(chunks):
        logging.info(f"Processing chunk {chunk_index + 1}/{len(chunks)} for year {year}")
        prompt = relevancy_prompt(current_paper, few_shot_prompt, year, chunk)
        try:
            response = evaluate_papers_with_llama(prompt, llama_pipeline)
            llama_output = parse_llama_output(response)
            logging.info(f"Year: {year}, Chunk {chunk_index + 1} Llama Output: {llama_output}")
            log_to_json_file({
                "year": year,
                "source_paper": current_paper,
                "llama_output": llama_output
            })
            relevant_papers_for_year.extend(extract_top_papers(llama_output))
        except Exception as e:
            logging.error(f"Error processing chunk {chunk_index + 1} for year {year}: {e}")
    return relevant_papers_for_year

def save_relevant_papers_to_file(papers, filename="./intermediate_chains_HN/all_relevant_papers.json"):
    """Save relevant papers to a JSON file."""
    global REVIEW_ID
    global IDX
    filename = filename.replace(".json", f"_{REVIEW_ID}_pk-{IDX}.json")
    with open(filename, "w") as outfile:
        json.dump(papers, outfile, indent=4)
    logging.info(f"Relevant papers saved to '{filename}'.")

def compute_score(paper, max_citation_count, w_r=0.7, w_c=0.3):
    """Calculate a paper's heuristic score."""
    relevance = paper.get("relevance", 0)
    citation_count = paper.get("citation_count", 0)
    normalized_citation = citation_count / max_citation_count if max_citation_count > 0 else 0
    return (w_r * relevance) + (w_c * normalized_citation)

def find_best_paper(relevant_papers):
    """Find the best paper based on heuristic scoring."""
    if not relevant_papers:
        return None

    max_citation_count = max(paper.get("citation_count", 1) for paper in relevant_papers)
    return max(relevant_papers, key=lambda paper: compute_score(paper, max_citation_count))

def process_top_papers(top_papers_across_years):
    """Process top papers to find the most relevant one."""
    if not top_papers_across_years:
        return None

    # Filter papers with relevance scores 1 or 2
    relevant_papers = [
        paper for paper in top_papers_across_years if paper.get("relevance", 0) in [1, 2]
    ]

    # Save relevant papers globally
    global saved_top_papers
    if not saved_top_papers:
        save_relevant_papers_to_file(relevant_papers)
        saved_top_papers = True

    logging.info(f"Relevant papers: {relevant_papers}")
    # Find and return the best paper
    return find_best_paper(relevant_papers) if relevant_papers else None

        
async def build_temporal_chain(source_paper, few_shot_prompt, current_year, end_year=2024, REVIEW_ID=None, SEED=42, IDX=1):
    initialize_globals(REVIEW_ID, SEED, IDX)
    global llama_pipeline
    if llama_pipeline is None:
        raise ValueError("Llama pipeline is not loaded.")
    
    chains = {}
    chains[current_year] = [{"paper": source_paper}]
    papers_to_process = [(source_paper, current_year)]

    while papers_to_process:
        logging.info(f"Current papers to process: {papers_to_process}")
        top_papers_across_years = []
        current_paper, current_year = papers_to_process.pop(0)
        if current_year > end_year:
            continue
        logging.info(f"Processing source paper: {current_paper}")

        # Load citations for the paper
        citation_details = await get_paper_by_id(current_paper['paperId'])
        year_wise_citations = filter_papers(citation_details, current_year)
        #logging.info(f"Year-wise citations: {year_wise_citations}")
        for year in range(current_year + 1, current_year + 3):
            logging.info(f"Processing citations for year {year}")
            try:
                relevant_papers_for_year = process_yearly_citations(
                    year, year_wise_citations, current_paper, few_shot_prompt, llama_pipeline
                )
                if relevant_papers_for_year:
                    top_papers_across_years.extend(relevant_papers_for_year)
            except Exception as e:
                logging.error(f"Error processing citations for year {year}: {e}")

        ## add here sys.exit to and run 10 times with different seeds?
        #sys.exit()
        
        if top_papers_across_years:
            best_paper = process_top_papers(top_papers_across_years)
            if best_paper:
                logging.info(f"Selected best paper using heuristic scoring: {best_paper}")
        else:
            best_paper = None

        if best_paper:
            logging.info(f"Adding best paper to process queue: {best_paper}")
            papers_to_process.append((best_paper, best_paper['year']))
            if best_paper['year'] not in chains:
                chains[best_paper['year']] = []
            chains[best_paper['year']].append({"paper": best_paper})
            logging.info(f"Appended chain for best paper {best_paper['title']} in year {best_paper['year']}")
            logging.info(f"Chain: {chains}")
        else:
            logging.info(f"No best paper found, incrementing year for current paper: {current_paper['title']}")
            papers_to_process.append((current_paper, current_year + 1))   
            
    return chains
    
async def extend_temporal_path_with_chains(source_paper, few_shot_prompt):
    
    paper_details = await get_paper_data(source_paper['pmid'])
    if paper_details is None:
        logging.error(f"Failed to fetch data for source paper with PMID: {source_paper['pmid']}. Skipping...")
        return
    
    source_paper = {
    "paperId": paper_details["paperId"],
    "pmid": paper_details["externalIds"].get("PubMed", "None"),
    "title": paper_details.get("title"),
    "abstract": paper_details.get("abstract"),
    "year": paper_details.get("year"),
    "citation_count": paper_details.get("citationCount","None")
    }
    load_global_llama()
    global REVIEW_ID
    global IDX
    # Start building the temporal chain from the source paper
    full_chain = await build_temporal_chain(source_paper, few_shot_prompt, source_paper['year'], REVIEW_ID, 42, IDX)
    
    # Save the output in hierarchical format
    #save_json(full_chain, "temporal_chains_llama.json")    
    linear_chain = extract_linear_chain(full_chain)
    outfile = f"./result_chains_all/temporal_chain_{REVIEW_ID}_p-{IDX}.json"
    save_json(linear_chain, outfile)
    logging.info(f"Linear chain saved to {outfile}.")

def main():
    args = parse_args()
    global IDX
    global REVIEW_ID
    REVIEW_ID = args.review_id
    set_global_seed(args.seed)
    
    with open("example_data/output.json", "r") as f:
        data = json.load(f)
    with open("example_data/gpt4_output", "r") as f:
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
    
    
    '''
    review_file_path = f"ground_truth_path/data/{REVIEW_ID}.json"
    if not os.path.exists(review_file_path):
        raise FileNotFoundError(f"Review file not found: {review_file_path}")
    
    with open(review_file_path, "r") as f:
        source = json.load(f)
    
    if not source:
        raise ValueError(f"The source JSON file '{review_file_path}' is empty.")
    
    source_paper = max(source, key=lambda x: x.get("CitationCount", 0))
    '''
    temporal_chain_path = f"./result_chains/temporal_chain_{REVIEW_ID}_p-1.json"
    with open(temporal_chain_path, "r") as f:
        temporal_chain = json.load(f)
    second_item = temporal_chain[1]
    second_item_id = second_item.get("paperId")
    relevant_papers_path = f"./intermediate_chains/all_relevant_papers_{REVIEW_ID}_pk-1.json"
    with open(relevant_papers_path, "r") as f:
        relevant_papers = json.load(f)
    new_source_papers = [paper for paper in relevant_papers if paper.get("paperId") != second_item_id]
    if not new_source_papers:
        logging.error("No new source papers found.")
        return
    
    for idx, source_paper in enumerate(new_source_papers):
        IDX = idx+2
        source_details = {
            "pmid": source_paper.get("paperId"),
            "title": source_paper.get("Title"),
            "abstract": source_paper.get("Abstract"),
            "year": source_paper.get("Year")
        }
        logging.info(f"Processing review ID: {REVIEW_ID} with seed: {args.seed}")
        asyncio.run(extend_temporal_path_with_chains(source_details, few_shot_prompt))
        
    #print(source_details)
    #sys.exit()

if __name__ == "__main__":
    main()