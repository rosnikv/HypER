def ra_prompt_edited(source_paper:dict, citing_paper_list:str):
    role = "an AI assistant"
    system_message = """You are {role} whose primary goal is to identify promising, new, and key scientific
problems based on existing scientific literature, in order to aid researchers in discovering novel
and significant research opportunities that can advance the field."""
    user_message = f"""You are going to generate a research problem that should be original, clear, feasible, relevant, and
significant to its field. This will be based on the title and abstract of the source paper, those of
{len(citing_paper_list)} related papers in the existing literature.
Understanding of the source paper and related papers is essential:
- The source paper is the primary research study you aim to enhance or build upon through future
research, serving as the central source and focus for identifying and developing the specific
research problem.
- The related papers that are connected to the source paper by citation chain, indicating their direct relevance
and connection to the primary research topic you are focusing on, and providing additional context
and insights that are essential for understanding and expanding upon the source paper.
Your approach should be systematic:
- Start by thoroughly reading the title and abstract of the source paper to understand its core focus.
- Next, proceed to read the titles and abstracts of the related papers to gain a broader perspective
and insights relevant to the primary research topic. The related papers are provided as an enumerated list of
Title and Abstract tuple.
I am going to provide the source paper and related papers, as follows:
Source paper title: {source_paper['title']}
Source paper abstract: {source_paper['abstract']}
Related paper: {citing_paper_list}
With the provided source paper, and the related papers, your objective now is to formulate a
research problem that not only builds upon these existing studies but also strives to be original,
clear, feasible, relevant, and significant. Before crafting the research problem, revisit the title
and abstract of the source paper, to ensure it remains the focal point of your research problem
identification process. 

Now convert this idea into a concrete testable hypothesis. Remeber hypothesis is a declarative statement expressing a 
relationship between two variables like independent or dependent variables or left group and rigt group in a given context.
Your hypothesis should contain the key variable or variables from your research idea.

Source paper title: {source_paper['title']}
Source paper abstract: {source_paper['abstract']}
Then, following your review of the above content, please proceed to generate one research
problem and hypothesis with the rationale, in the format of
Rationale:
Problem:
Hypothesis:
"""
    return system_message, user_message

def ra_prompt_edited_for_validity(source_paper:dict, citing_paper_list:str):
    role = "an AI assistant"
    system_message = """You are {role} whose primary goal is to identify promising, new, and key scientific
problems based on existing scientific literature, in order to aid researchers in discovering novel
and significant research opportunities that can advance the field."""
    user_message = f"""You are going to generate a research problem that should be original, clear, feasible, relevant, and significant to its field. This will be based on the title and abstract of the source paper, those of {len(citing_paper_list)} related papers in the existing literature.
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
  "Analysis": {{Output a dictionary with each paper in the Related Papers as a key. For each key (paper) analyze how this paper builds upon the previous papers in the list. For example, how Paper 0 builds upon source paper and Paper 1 builds upon the concepts in Paper 0 and so on. Elaborate on specific advancements made, including the explanation behind their effectiveness in addressing previous challenges. Apply this analytical approach to each valid paper in the sequence, adding the analysis as the value for each key in a few sentences. Ignore papers that do not build upon the previous papers and diverge from the original source paper's topic significantly.}},
  "Rationale": "Summarize the above analysis and explain how you would come up with a research idea that will advance the field of work while addressing the limitations of previous work and building upon the existing work.",
  "Research idea": "Delineate an elaborate research problem here including the key variables.",
  "Hypothesis": "Provide a concrete testable hypothesis that follows from the above research problem here"
}}
```
This JSON will be automatically parsed, so ensure the format is precise.
"""
    return system_message, user_message

import json 

def ra_prompt_edited_for_validity_fewshot(source_paper:dict, citing_paper_list:str):
    with open('./code/prompts/fewshot_example_hypothesis_generation_3_corrected.json') as f:
        one_shot = json.load(f)
    
    system_message = """You are an AI assistant whose primary goal is to identify promising, new, and key scientific problems based on existing scientific literature, in order to aid researchers in discovering novel and significant research opportunities that can advance the field."""
    user_message = f"""You are going to generate a research problem that should be original, clear, feasible, relevant, and significant to its field. This will be based on the title and abstract of the source paper, those of {len(citing_paper_list)} related papers in the existing literature.
    Understanding of the target paper, and the related papers is essential:
    - The source paper is the primary research study you aim to enhance or build upon through future
    research, serving as the central source and focus for identifying and developing the specific
    research problem.
    - The related papers are arranged in temporal order of citation, such that paper 0 cites the source paper, 
    2 cites paper 1 and paper 3 cites paper 2 and so on. The relevant papers provide additional context and insights 
    that are essential for understanding and expanding upon the target paper. However, all the papers in the list may 
    not be relevant to the primary research you are focusing on. Identify the most relevant papers from the list in your 
    analysis and only use those for research idea generation.
    Your approach should be systematic:
    - Start by thoroughly reading the title and abstract of the source paper to understand its core focus.
    - Next, proceed to read the titles and abstracts of the related papers in the order in which they appear in the list.
    Identify the papers that form a logical reasoning chain starting from the source paper.
    - Use only these papers to gain a broader perspective about the progression of the primary research topic over time. 

    ### **Example Task & Expected Output**
    #### **Example Input:**
    ###
    Example Input:
    Source paper title: {one_shot['source paper']['title']}
    Source paper abstract: {one_shot['source paper']['abstract']}
    Source paper year of publication: {one_shot['source paper']['year']}
    Related papers: {one_shot['related papers']}

    #### **Example Output (Valid JSON Format):**
    ```json
    {{
    "Analysis": {one_shot['output']['<analysis>']},
    "Rationale": "{one_shot['output']['<motivation>']}",
    "Research idea": "{one_shot['output']['<research idea>']}",
    "Hypothesis": "{one_shot['output']['<hypothesis>']}"
    }}
    ```
    ###
    ### **Important:  **Do not copy from the example above.** Instead, based on the provided source and related papers to generate a research problem that should be original, clear, feasible, relevant, and significant to its field. 

    I am going to provide the source paper and related papers as an enumerated list of Title, Abstract and Year of publication triple, as follows:
    Source paper title: {source_paper['title']}
    Source paper abstract: {source_paper['abstract']}
    Source paper year of publication: {source_paper['year']}
    Related papers: {citing_paper_list}
    
    With the provided source paper, and the related papers, your objective now is to formulate a
    research problem that not only builds upon these existing studies but also strives to be original,
    clear, feasible, relevant, and significant. Before crafting the research problem, revisit the title
    and abstract of the source paper, to ensure it remains the focal point of your research problem
    identification process. Your research problem will be scored for clarity. It should contain a short description 
    of the general research idea and it's impact followed by more details on all the variables and how they will be measured. 
    If possible include PICO elements which stands for Population, Intervention, Control and Outcome. 
    State clearly how the outcome could potentially be measured.

    Now convert this idea into a concrete testable hypothesis. Remember hypothesis is a declarative statement expressing a 
    relationship between two variables like independent or dependent variables or left group and rigt group in a given context.
    Your hypothesis should contain the key variable or variables from your research idea and how they will be measured.
    Your hypothesis will be scored on clarity and novelty.
    
    Source paper title: {source_paper['title']}
    Source paper abstract: {source_paper['abstract']}

    Then, following your review of the above content and example, please proceed to analyze the progression of the research topic. For analysis, Output a dictionary with each paper in the Related Papers as a key. For each key (paper) analyze how this paper builds upon the previous papers in the list. For example, how Paper 0 builds upon source paper and Paper 1 builds upon the concepts in Paper 0 and so on. Elaborate on specific advancements made, including the explanation behind their effectiveness in addressing previous challenges. Apply this analytical approach to each valid paper in the sequence, adding the analysis as the value for each key in a few sentences. Ignore papers that do not build upon the previous papers and diverge from the original source paper's topic significantly.
    Now output this analysis, the research problem and hypothesis with the rationale. Your output should be a valid JSON with the following fields.  
    
    Output a JSON object in the following format:
    ```json
    {{
    "Analysis": {{Output a dictionary with each paper in the Related Papers as a key. For each key (paper) analyze how this paper builds upon the previous papers in the list.}},
    "Rationale": "Summarize the above analysis and explain how you would come up with a research idea that will advance the field of work while addressing the limitations of previous work and building upon the existing work.",
    "Research idea": "Delineate an elaborate research problem here including the key variables.",
    "Hypothesis": "Provide a concrete testable hypothesis that follows from the above research problem here"
    }}
    ```
    This JSON will be automatically parsed, so ensure the format is precise. DO NOT leave any field empty. If you cannot generate a specific part, provide a best guess.
    """
    return system_message, user_message

def ra_prompt_edited_for_validity_fewshot_v2(source_paper:dict, citing_paper_list:str):
    with open('./code/prompts/fewshot_example_hypothesis_generation_3_corrected.json') as f:
        one_shot = json.load(f)
    
    system_message = """You are an AI assistant whose primary goal is to identify promising, new, and key scientific problems based on existing scientific literature, in order to aid researchers in discovering novel and significant research opportunities that can advance the field."""
    user_message = f"""You are going to generate a research problem that should be original, clear, feasible, relevant, and significant to its field. This will be based on the title and abstract of the source paper, those of {len(citing_paper_list)} related papers in the existing literature.
    Understanding of the target paper, and the related papers is essential:
    - The source paper is the primary research study you aim to enhance or build upon through future
    research, serving as the central source and focus for identifying and developing the specific
    research problem.
    - The related papers are arranged in temporal order of citation, such that paper 0 cites the source paper, 
    2 cites paper 1 and paper 3 cites paper 2 and so on. The relevant papers provide additional context and insights 
    that are essential for understanding and expanding upon the target paper.
    Your approach should be systematic:
    - Start by thoroughly reading the title and abstract of the source paper to understand its core focus.
    - Next, proceed to read the titles and abstracts of the related papers in the order in which they appear in the list.
    Identify the papers that form a logical reasoning chain starting from the source paper.
    - Use only these papers to gain a broader perspective about the progression of the primary research topic over time. 

    ### **Example Task & Expected Output**
    #### **Example Input:**
    ###
    Example Input:
    Source paper title: {one_shot['source paper']['title']}
    Source paper abstract: {one_shot['source paper']['abstract']}
    Source paper year of publication: {one_shot['source paper']['year']}
    Related papers: {one_shot['related papers']}

    #### **Example Output (Valid JSON Format):**
    ```json
    {{
    "Analysis": {one_shot['output']['<analysis>']},
    "Rationale": "{one_shot['output']['<motivation>']}",
    "Research idea": "{one_shot['output']['<research idea>']}",
    "Hypothesis": "{one_shot['output']['<hypothesis>']}"
    }}
    ```
    ###
    ### **Important:  **Do not copy from the example above.** Instead, based on the provided source and related papers to generate a research problem that should be original, clear, feasible, relevant, and significant to its field. 

    I am going to provide the source paper and related papers as an enumerated list of Title, Abstract and Year of publication triple, as follows:
    Source paper title: {source_paper['title']}
    Source paper abstract: {source_paper['abstract']}
    Source paper year of publication: {source_paper['year']}
    Related papers: {citing_paper_list}
    
    With the provided source paper, and the related papers, your objective now is to formulate a
    research problem that not only builds upon these existing studies but also strives to be original,
    clear, feasible, relevant, and significant. Before crafting the research problem, revisit the title
    and abstract of the source paper, to ensure it remains the focal point of your research problem
    identification process. Your research problem will be scored for clarity. It should contain a short description 
    of the general research idea and it's impact followed by more details on all the variables and how they will be measured. 
    If possible include PICO elements which stands for Population, Intervention, Control and Outcome. 
    State clearly how the outcome could potentially be measured.

    Now convert this idea into a concrete testable hypothesis. Remember hypothesis is a declarative statement expressing a 
    relationship between two variables like independent or dependent variables or left group and rigt group in a given context.
    Your hypothesis should contain the key variable or variables from your research idea and how they will be measured.
    Your hypothesis will be scored on clarity and novelty.
    
    Source paper title: {source_paper['title']}
    Source paper abstract: {source_paper['abstract']}

    Then, following your review of the above content and example, please proceed to analyze the progression of the research topic. For analysis, Output a dictionary with each paper in the Related Papers as a key. For each key (paper) analyze how this paper builds upon the previous papers in the list. For example, how Paper 0 builds upon source paper and Paper 1 builds upon the concepts in Paper 0 and so on. Elaborate on specific advancements made, including the explanation behind their effectiveness in addressing previous challenges. Apply this analytical approach to each valid paper in the sequence, adding the analysis as the value for each key in a few sentences. Ignore papers that do not build upon the previous papers and diverge from the original source paper's topic significantly.
    Now output this analysis, the research problem and hypothesis with the rationale. Your output should be a valid JSON with the following fields.  
    
    Output a JSON object in the following format:
    ```json
    {{
    "Analysis": {{Output a dictionary with each paper in the Related Papers as a key. For each key (paper) analyze how this paper builds upon the previous papers in the list.}},
    "Rationale": "Summarize the above analysis and explain how you would come up with a research idea that will advance the field of work while addressing the limitations of previous work and building upon the existing work.",
    "Research idea": "Delineate an elaborate research problem here including the key variables.",
    "Hypothesis": "Provide a concrete testable hypothesis that follows from the above research problem here"
    }}
    ```
    This JSON will be automatically parsed, so ensure the format is precise. DO NOT leave any field empty. If you cannot generate a specific part, provide a best guess.
    """
    return system_message, user_message


def ra_prompt_priming_2024_paper(source_paper:dict, citing_paper_list:str, paper2024:str):
    role = "an AI assistant"
    system_message = """You are {role} whose primary goal is to identify promising, new, and key scientific
problems based on existing scientific literature, in order to aid researchers in discovering novel
and significant research opportunities that can advance the field."""
    user_message = f"""You are going to generate a research problem that should be original, clear, feasible, relevant, and
significant to its field. This will be based on the title and abstract of the source paper, those of
{len(citing_paper_list)} related papers in the existing literature, and insights from the given research topic.
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
triple, and a research topic as follows:
Source paper title: {source_paper['title']}
Source paper abstract: {source_paper['abstract']}
Source paper year of publication: {source_paper['year']}
Related papers: {citing_paper_list}
Research topic: {paper2024}
With the provided source paper, and the related papers, and the research topic, your objective now is to formulate a
research problem that not only builds upon these existing studies but also strives to be original,
clear, feasible, relevant, and significant. Before crafting the research problem, revisit the title
and abstract of the target paper, to ensure it remains the focal point of your research problem
identification process. 

Now convert this idea into a concrete testable hypothesis. Remember hypothesis is a declarative statement expressing a 
relationship between two variables like independent or dependent variables or left group and rigt group in a given context.
Your hypothesis should contain the key variable or variables from your research idea.

Source paper title: {source_paper['title']}
Source paper abstract: {source_paper['abstract']}

Then, following your review of the above content, please proceed to analyze the progression of the research topic. Now output this
analysis, the research idea and hypothesis with the rationale. 
Your output should be a valid JSON with the following fields. This JSON will be automatically parsed, so ensure the format is precise. 
  
Output a JSON object in the following format:
```json
{{
  "Analysis": "Output a dictionary with each paper in the Related Papers as a key. For each key analyze how this paper builds upon the previous papers in the list. For example, how Paper 0 builds upon source paper and Paper 1 builds upon the concepts in Paper 0 and so on. Elaborate on specific advancements made, including the rationale behind their effectiveness in addressing previous challenges. Apply this analytical approach to each valid paper in the sequence. Ignore papers that do not build upon the previous papers and diverge from the original source paper's topic significantly.",
  "Rationale": "Summarize the above analysis and explain how you would come up with a research idea that will advance the field of work while addressing the limitations of previous work and building upon the existing work.",
  "Research idea": "Delineate an elaborate research problem here including the key variables.",
  "Hypothesis": "Provide a concrete testable hypothesis that follows from the above research problem here"
}}
```
"""
    return system_message, user_message



def ra_prompt_subgraph(source_paper:dict, hypothesis_subgraph:str, paper_count:int):
    paper_count = paper_count-3
    role = "an AI assistant"
    system_message = """You are {role} whose primary goal is to identify promising, new, and key scientific
problems based on existing scientific literature, in order to aid researchers in discovering novel
and significant research opportunities that can advance the field."""
    user_message = f"""You are going to generate a research problem that should be original, clear, feasible, relevant, and
significant to its field. This will be based on the title and abstract of the source paper, those of
{paper_count} related papers in the existing literature given as a subgraph.
Understanding of the target paper, and the related papers is essential:
- The source paper is the primary research study you aim to enhance or build upon through future
research, serving as the central source and focus for identifying and developing the specific
research problem.
- The related papers are arranged as a subgraph consisting of multiple chains with source paper serving as the starting research. The related papers from each chain are arranged in temporal order of citation, such that paper 2 cites paper 1 and 
paper 3 cites paper 2 and so on. The subgraph provide additional context and insights that are essential for 
understanding and expanding upon the target paper. However, all the papers in the subgraph may not be relevant to the primary 
research you are focusing on. 
Your approach should be systematic:
- Start by thoroughly reading the title and abstract of the source paper to understand its core focus.
- Next, proceed to read the titles and abstracts of the related papers in the order in which they appear in the chains.
Identify the valid chain from the subgraph that forms a logical reasoning chain starting from the source paper. The valid chain in the subgraph contains the only papers that follow a logical progression from the source paper.
- Use only these papers to gain a broader perspective about the progression of the primary research topic over time. Papers that do not contribute to this logical progression should be ignored in the analysis.


I am going to provide the source paper and the subgraph, where related papers in the subgraph chain as an enumerated list of Title, Topic, Hypothesis and Year of publication 
quadruple as follows:
Source paper title: {source_paper['title']}
Source paper abstract: {source_paper['abstract']}
Source paper year of publication: {source_paper['year']}
Subgraph: {hypothesis_subgraph}
With the provided source paper, and the related papers from the subgraph, your objective now is to formulate a
research problem that not only builds upon these existing studies but also strives to be original,
clear, feasible, relevant, and significant. Before crafting the research problem, revisit the title
and abstract of the target paper, to ensure it remains the focal point of your research problem
identification process. 

Now convert this idea into a concrete testable hypothesis. Remember hypothesis is a declarative statement expressing a 
relationship between two variables like independent or dependent variables or left group and rigt group in a given context.
Your hypothesis should contain the key variable or variables from your research idea.

Source paper title: {source_paper['title']}
Source paper abstract: {source_paper['abstract']}

Then, following your review of the above content, please proceed to analyze the progression of the research topic. Now output this
analysis, the research idea and hypothesis with the rationale. 
Your output should be a valid JSON with the following fields. This JSON will be automatically parsed, so ensure the format is precise. 
  
Output a JSON object in the following format:
```json
{{
  "Analysis": "Output a dictionary where each key is a valid related paper's title from the valid chain, where each value is a clear analysis of how that paper builds from the previous one. For each key analyze how this paper builds upon the previous papers in the valid chain. For example, how Paper 0 builds upon source paper and Paper 1 builds upon the concepts in Paper 0 and so on. Elaborate on specific advancements made, including the rationale behind their effectiveness in addressing previous challenges. Apply this analytical approach to each valid paper in the sequence. Ignore papers that do not build upon the previous papers and diverge from the original source paper's topic significantly.",
  "Rationale": "Summarize the above analysis and explain how you would come up with a research idea that will advance the field of work while addressing the limitations of previous work and building upon the existing work.",
  "Research idea": "Delineate an elaborate research problem here including the key variables.",
  "Hypothesis": "Provide a concrete testable hypothesis that follows from the above research problem here"
}}
```
"""
    return system_message, user_message

def reviewagent(scoring_protocol, metric, related_papers, source_paper, research_idea):
    system_message = """You are an AI assistant whose primary goal is to assess the quality and validity of scientific
problems across diverse dimensions, in order to aid researchers in refining their problems based
on your evaluations and feedback, thereby enhancing the impact and reach of their work. Your response must be in JSON format"""
    user_message = f"""You are going to evaluate a research problem for its {metric}, focusing on how well it is defined
in a clear, precise, and understandable manner.
As part of your evaluation, you can refer to the existing studies that may be related to the problem,
which will help in understanding the context of the problem for a more comprehensive assessment.
- The existing studies refer to the target paper that has been pivotal in identifying the problem, as
well as the related papers that have been additionally referenced in the discovery phase of the
problem.
The existing studies (target paper & related papers) are as follows:
Target paper title: {source_paper['title']}
Target paper abstract: {source_paper['abstract']}
Related papers: {related_papers}
Now, proceed with your {metric} evaluation approach that should be systematic:
- Start by thoroughly reading the research problem and its rationale, keeping in mind the context
provided by the existing studies mentioned above.
- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing
on the {metric} of the problem.
- Finally, provide a score for the Hypothesis on a 5-point Likert scale, with 1 being the lowest. Be a harse critic. Please ensuring a discerning and critical evaluation and avoid uniformly high ratings (4-5) unless fully justified. 
Following are the judging criteria for each rating number:
{scoring_protocol[metric]}
I am going to provide the research problem with its rationale, as follows:
Research problem: {research_idea['Research idea']}
Rationale: {research_idea['Rationale']}
Hypothesis: {research_idea['Hypothesis']}
After your evaluation of the above content, please provide your review, feedback, and rating. 
Your output should be structured as follows:
RESPONSE:
```json
<JSON>
```
In <JSON>, respond in JSON format with ONLY the following field:
- "Review": Your review of the research problem.
- "Feedback": Your constructive feedback for improvement.
- "Rating (1-5) for Hypothesis": only output a rating number here.
This JSON will be automatically parsed, so ensure the format is precise.
"""
    return system_message, user_message

def fewshotagent(scoring_protocol, metric, related_papers, source_paper, research_idea): 
    system_message = """You are an AI assistant specialized in generating research problems with varying levels of quality across a specific evaluation metric. Your task is to create multiple versions of a research problem, rationale, and hypothesis, each corresponding to a different rating level for the given metric. Your response must be in JSON format."""
    user_message = f"""You are going to evaluate a research problem for its {metric}, focusing on how well it is defined
in a clear, precise, and understandable manner. Your task is to generate different versions of a research problem, rationale, and hypothesis for the metric {metric}, ensuring systematic variations in quality based on a Likert scale of 1 to 5.
As part of your evaluation, you can refer to the existing studies that may be related to the problem,
which will help in understanding the context of the problem for a more comprehensive assessment.
- The existing studies refer to the target paper that has been pivotal in identifying the problem, as
well as the related papers that have been additionally referenced in the discovery phase of the
problem.
The existing studies (target paper & related papers) are as follows:
Target paper title: {source_paper['title']}
Target paper abstract: {source_paper['abstract']}
Related papers: {related_papers}
Now, proceed with your {metric} evaluation approach that should be systematic:
- Start by thoroughly reading the research problem and its rationale, keeping in mind the context
provided by the existing studies mentioned above.
- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing
on the {metric} of the problem.
- Finally, provide a score for the Hypothesis on a 5-point Likert scale, with 1 being the lowest. Ensure a critical and discerning evaluation. Please ensuring a discerning and critical evaluation and avoid uniformly high ratings (4-5) unless fully justified. 
Following are the judging criteria for each rating number:
{scoring_protocol[metric]}
I am going to provide the research problem with its rationale, as follows:
Research problem: {research_idea['Research idea']}
Rationale: {research_idea['Rationale']}
Hypothesis: {research_idea['Hypothesis']}
After your evaluation of the above content, please provide your review, feedback, and rating.
Once you have completed the evaluation, generate alternative research problems, rationales, and hypotheses for different rating levels (1, 2, 3, 4, and 5). For each rating (1 to 5), create a distinct research problem, rationale, and hypothesis that align with the given rating level. Each variant should be meaningfully different in approach, scope, or complexity.
Your output should be structured as follows:
RESPONSE:
```json
{{
    "Evaluation": {{
        "Review": "Your review of the research problem.",
        "Feedback": "Your constructive feedback for improvement.",
        "Rating (1-5) for Hypothesis": "only output a rating number here"
    }},
    "Generated Variants": {{
        "Rating 1": {{
            "Research problem": "Example research idea...",
            "Rationale": "Example rationale...",
            "Hypothesis": "Example hypothesis...",
            "Review": "Your review of the research problem",
            "Feedback": "Your constructive feedback for improvement"
        }},
        "Rating 2": {{
            "Research problem": "Example research idea...",
            "Rationale": "Example rationale...",
            "Hypothesis": "Example hypothesis...",
            "Review": "Your review of the research problem",
            "Feedback": "Your constructive feedback for improvement"
        }},
        ...
        "Rating 5": {{
            "Research problem": "Example research idea...",
            "Rationale": "Example rationale...",
            "Hypothesis": "Example hypothesis...",
            "Review": "Your review of the research problem",
            "Feedback": "Your constructive feedback for improvement"
        }}
    }} 
}}
```
This JSON will be automatically parsed, so ensure the format is precise.
"""
    return system_message, user_message    



def fewshot_reviewagent(scoring_protocol, metric, related_papers, source_paper, research_idea):
    with open("./code/prompts/fewshots_rating.json", "r") as f:
        fewshots_rating = json.load(f)
    fewshots = fewshots_rating[0]['few_shot_results'][metric]["Generated Variants"]
    
    system_message = """You are an AI assistant whose primary goal is to assess the quality and validity of scientific
problems across diverse dimensions, in order to aid researchers in refining their problems based
on your evaluations and feedback, thereby enhancing the impact and reach of their work. Your response must be in JSON format"""
    user_message = f"""You are going to evaluate a research problem for its {metric}, focusing on how well it is defined
in a clear, precise, and understandable manner.
As part of your evaluation, you can refer to the existing studies that may be related to the problem,
which will help in understanding the context of the problem for a more comprehensive assessment.
- The existing studies refer to the target paper that has been pivotal in identifying the problem, as
well as the related papers that have been additionally referenced in the discovery phase of the
problem.
The existing studies (target paper & related papers) are as follows:
Target paper title: {source_paper['title']}
Target paper abstract: {source_paper['abstract']}
Related papers: {related_papers}

Now, proceed with your {metric} evaluation approach that should be systematic:
- Start by thoroughly reading the research problem and its rationale, keeping in mind the context
provided by the existing studies mentioned above.
- Next, generate a review and feedback that should be constructive, helpful, and concise, focusing
on the {metric} of the problem.
- Finally, provide a score for the Hypothesis on a 5-point Likert scale, with 1 being the lowest. Be a harse critic. Please ensuring a discerning and critical evaluation and avoid uniformly high ratings (4-5) unless fully justified. 
Following are the judging criteria for each rating number:
{scoring_protocol[metric]}

 ### **Example Input & Expected Output**
    #### **Example Input:**
    ###
    Example Input for Rating 1:
    Research problem: {fewshots['Rating 1']["Research problem"]}
    Rationale: {fewshots['Rating 1']['Rationale']}
    Hypothesis: {fewshots['Rating 1']['Hypothesis']}
    #### **Example Output (Valid JSON Format):**
    ```json
    {{
    "Review": "{fewshots['Rating 1']['Review']}",
    "Feedback": "{fewshots['Rating 1']['Feedback']}"
    "Rating (1-5) for Hypothesis": 1
    }}
    ```
    ###
    ###
    Example Input for Rating 3:
    Research problem: {fewshots['Rating 3']["Research problem"]}
    Rationale: {fewshots['Rating 3']['Rationale']}
    Hypothesis: {fewshots['Rating 3']['Hypothesis']}
    #### **Example Output (Valid JSON Format):**
    ```json
    {{
    "Review": "{fewshots['Rating 3']['Review']}",
    "Feedback": "{fewshots['Rating 3']['Feedback']}"
    "Rating (1-5) for Hypothesis": 3
    }}
    ```
    ###
    ###
    Example Input for Rating 5:
    Research problem: {fewshots['Rating 5']["Research problem"]}
    Rationale: {fewshots['Rating 5']['Rationale']}
    Hypothesis: {fewshots['Rating 5']['Hypothesis']}
    #### **Example Output (Valid JSON Format):**
    ```json
    {{
    "Review": "{fewshots['Rating 5']['Review']}",
    "Feedback": "{fewshots['Rating 5']['Feedback']}"
    "Rating (1-5) for Hypothesis": 5
    }}
    ```
    ###

I am going to provide the research problem with its rationale, as follows:
Research problem: {research_idea['Research idea']}
Rationale: {research_idea['Rationale']}
Hypothesis: {research_idea['Hypothesis']}
After your evaluation of the above content, please provide your review, feedback, and rating. 
Your output should be structured as follows:
RESPONSE:
```json
<JSON>
```
In <JSON>, respond in JSON format with ONLY the following field:
- "Review": Your review of the research problem.
- "Feedback": Your constructive feedback for improvement.
- "Rating (1-5) for Hypothesis": only output a rating number here.
This JSON will be automatically parsed, so ensure the format is precise.
"""
    return system_message, user_message