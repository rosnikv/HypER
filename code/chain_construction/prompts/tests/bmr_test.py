import os
# add gpu support
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
from transformers import AutoModel, AutoTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "BMRetriever/BMRetriever-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

from torch.nn.functional import cosine_similarity
import torch
import torch.nn.functional as F
from torch import Tensor

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        embedding = last_hidden[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden.shape[0]
        embedding = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    return embedding

def get_detailed_instruct_query(task_description: str, query: str) -> str:
    return f'{task_description}\nQuery: {query}'

def get_detailed_instruct_passage(passage: str) -> str:
    return f'Represent this passage\npassage: {passage}'

# Function to compute weighted similarity
def compute_weighted_similarity(source_embedding: Tensor, doc_embeddings: Tensor, weights: Tensor) -> Tensor:
    similarities = cosine_similarity(source_embedding.unsqueeze(0), doc_embeddings)
    weighted_similarities = similarities * weights
    return weighted_similarities.sum()

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a scientific abstract, retrieve documents that support or refute the claims from this abstract'

# Tokenize and encode inputs
source_abstract = "BACKGROUND\nThe prophylactic administration of indomethacin reduces the frequency of patent ductus arteriosus and severe intraventricular hemorrhage in very-low-birth-weight infants (those with birth weights below 1500 g). Whether prophylaxis with indomethacin confers any long-term benefits that outweigh the risks of drug-induced reductions in renal, intestinal, and cerebral blood flow is not known.\n\n\nMETHODS\nSoon after they were born, we randomly assigned 1202 infants with birth weights of 500 to 999 g (extremely low birth weight) to receive either indomethacin (0.1 mg per kilogram of body weight) or placebo intravenously once daily for three days. The primary outcome was a composite of death, cerebral palsy, cognitive delay, deafness, and blindness at a corrected age of 18 months. Secondary long-term outcomes were hydrocephalus necessitating the placement of a shunt, seizure disorder, and microcephaly within the same time frame. Secondary short-term outcomes were patent ductus arteriosus, pulmonary hemorrhage, chronic lung disease, ultrasonographic evidence of intracranial abnormalities, necrotizing enterocolitis, and retinopathy.\n\n\nRESULTS\nOf the 574 infants with data on the primary outcome who were assigned to prophylaxis with indomethacin, 271 (47 percent) died or survived with impairments, as compared with 261 of the 569 infants (46 percent) assigned to placebo (odds ratio, 1.1; 95 percent confidence interval, 0.8 to 1.4; P=0.61). Indomethacin reduced the incidence of patent ductus arteriosus (24 percent vs. 50 percent in the placebo group; odds ratio, 0.3; P<0.001) and of severe periventricular and intraventricular hemorrhage (9 percent vs. 13 percent in the placebo group; odds ratio, 0.6; P=0.02). No other outcomes were altered by the prophylactic administration of indomethacin.\n\n\nCONCLUSIONS\nIn extremely-low-birth-weight infants, prophylaxis with indomethacin does not improve the rate of survival without neurosensory impairment at 18 months, despite the fact that it reduces the frequency of patent ductus arteriosus and severe periventricular and intraventricular hemorrhage."
candidate_documents = [
    "Title: Perfusion Index as a Diagnostic Tool for Patent Ductus Arteriosus in Preterm Infants Abstract: Background: Perfusion index (PI) could reflect peripheral flow. Preterm infants with hemodynamically significant patent ductus arteriosus (hsPDA) will have left-to-right shunt across PDA causing less blood flow to the lower legs. Objective: To evaluate pre- and postductal PI differences (ΔPI) in hsPDA. Methods: Preterm infants with gestational age <34 weeks were assessed for ΔPI on days 1, 3, and 7 of life with simultaneous echocardiography. Based on echocardiography, each infant was categorized into hsPDA, non-hsPDA, and no PDA. Results: Thirty infants (16 males), median age 31 weeks (interquartile range, IQR, 29-32) and weight 1,490 g (IQR 1,100-1,670) were enrolled. On days 1 and 3 of life, the ΔPI of infants with hsPDA (1.57%, IQR 0.28-2.32, n = 14, and 1.32%, IQR 0.28-1.83, n = 10) were significantly higher than those without hsPDA (0.14%, IQR -0.03 to 0.30, n = 16, and 0.08%, IQR -0.07 to 0.26, n = 20), p = 0.009 and 0.005, respectively. At all time points (days 1, 3, and 7 of life, n = 84), ΔPI >1.05% had sensitivity, specificity, positive predictive value, and negative predictive value of 66.7, 100, 100, and 86.4%, respectively, to detect hsPDA. Conclusion: The pre- and postductal PI differences were significantly related to the hemodynamic changes of PDA and might be useful to detect hemodynamically significant PDA.",
    "Title: Reduction of Severe Intraventricular Hemorrhage in Preterm Infants: A Quality Improvement Project. Abstract: OBJECTIVES\nThe aim of this quality improvement project was to reduce the rate of severe intraventricular hemorrhage (sIVH) by 50% within 3 years for extremely preterm infants born at a children\'s teaching hospital.\n\n\nMETHODS\nA multidisciplinary team developed key drivers for the development of intraventricular hemorrhage in preterm infants. Targeted interventions included the development of potentially better practice guidelines, promoting early noninvasive ventilation, consistent use of rescue antenatal betamethasone, and risk-based indomethacin prophylaxis. The outcome measure was the rate of sIVH. Process measures included the rate of intubation within 24 hours and receipt of rescue betamethasone and risk-based indomethacin prophylaxis. Common markers of morbidity were balancing measures. Data were collected from a quarterly chart review and analyzed with statistical process control charts. The preintervention period was from January 2012 to March 2016, implementation period was from April 2016 to December 2018, and sustainment period was through June 2020.\n\n\nRESULTS\nDuring the study period, there were 268 inborn neonates born at <28 weeks\' gestation or <1000 g (127 preintervention and 141 postintervention). The rate of sIVH decreased from 14% to 1.2%, with sustained improvement over 2 and a half years. Mortality also decreased by 50% during the same time period. This was associated with adherence to process measures and no change in balancing measures.\n\n\nCONCLUSIONS\nA multipronged quality improvement approach to intraventricular hemorrhage prevention, including evidence-based practice guidelines, consistent receipt of rescue betamethasone and indomethacin prophylaxis, and decreasing early intubation was associated with a sustained reduction in sIVH in extremely preterm infants."
]


# Tokenize and encode the source abstract
source_input = tokenizer(source_abstract, max_length=512, padding=True, truncation=True, return_tensors='pt')
source_input = {key: value.to(device) for key, value in source_input.items()}  # Move all inputs to GPU(s)
source_embedding = last_token_pool(
    model(**source_input).last_hidden_state, source_input['attention_mask']
)

# Encode the candidate documents
doc_embeddings = []
for doc in candidate_documents:
    doc_input = tokenizer(doc, max_length=512, padding=True, truncation=True, return_tensors='pt')
    doc_input = {key: value.to(device) for key, value in doc_input.items()}  # Move all inputs to GPU(s)
    doc_embedding = last_token_pool(
        model(**doc_input).last_hidden_state, doc_input['attention_mask']
    )
    doc_embeddings.append(doc_embedding)

# Ensure correct concatenation of document embeddings
doc_embeddings = torch.stack(doc_embeddings, dim=0).to(device)

# Compute cosine similarity
similarities = cosine_similarity(source_embedding.unsqueeze(0), doc_embeddings).cpu()  # Move results to CPU for evaluation

# Get the most relevant document
best_doc_idx = similarities.argmax()
print(f"Most relevant document index: {best_doc_idx}")
