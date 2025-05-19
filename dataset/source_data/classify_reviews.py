import sys
sys.path.append('/home/user/rosni/scigen')
import pandas as pd

train_targets_df = pd.read_csv("./backup/RCT-summarization-data/train-targets.csv")  
train_targets_df.head()
docs = train_targets_df["Target"].tolist() 

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,6"
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
from huggingface_hub import login
login(token=huggingface_token)

from transformers import pipeline
import torch

model_id = "meta-llama/Llama-3.1-8B-Instruct"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

specialties = [
    "Anesthesiology", "Cardiac and Thoracic Surgery", "Cardiology", "Dentistry",
    "Dermatology", "Emergency Medicine", "Endocrinology", "Family Medicine",
    "Gastroenterology and Hepatology", "General Surgery", "Hematology and Oncology",
    "Infectious Diseases", "Nephrology", "Neurology", "Neurosurgery", 
    "Obstetrics and Gynecology", "Ophthalmology", "Orthopedic Surgery", 
    "Otolaryngology", "Pediatrics", "Plastic Surgery", "Psychiatry", 
    "Pulmonology", "Radiology", "Rehabilitation Medicine", "Rheumatology", "Urology"
]


import json

system_prompt = """
You are a helpful assistant trained to classify medical documents into specific medical specialties. Your task is to return the result in a strict JSON format without any additional explanation or extra text.

Make sure the JSON is valid and includes only the predicted specialty.
"""

results = []

for doc in docs:
    prompt = f"""
        Classify the following document summary into one of the medical specialties:
        "{doc}"

        Specialties: {', '.join(specialties)}.

        Please provide the answer in a strict JSON object:
        ```json
        {{
            "specialty": "<predicted specialty>"
        }}
        ```
        """
    
    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}
    ]

    outputs = pipe(
        messages,
        max_new_tokens=150,  
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    response_text = outputs[0]["generated_text"]
    for item in response_text:
        if item['role'] == 'assistant':
            try:
                json_part = item['content'].split("```json")[-1].strip().strip("```")
                
                result = json.loads(json_part)
                predicted_specialty = result.get("specialty", "Unknown")
            except json.JSONDecodeError:
                predicted_specialty = "Unknown"

            results.append({
                "document": doc,
                "predicted_specialty": predicted_specialty
            })
            print(f"Predicted specialty: {item['content']}")

with open("./data/document_predictions.json", "w") as f:
    json.dump(results, f, indent=4)

print("Predictions saved to 'document_predictions.json'")