import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from sentence_transformers import SentenceTransformer, CrossEncoder 
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
import time
import os
import pickle
import pandas as pd
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import re
from vllm import LLM, SamplingParams
import gc

from codecarbon import track_emissions


#os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


nltk.download('punkt')
nltk.download('punkt_tab')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rer_model = "cross-encoder/ms-marco-MiniLM-L12-v2"

class EvidenceRetriever:
    def __init__(self, emb_model_name="intfloat/e5-large-v2", device=device):
        self.bm25_top_k = 1500
        self.cos_sim_top_k = 150
        self.embedding_model = SentenceTransformer(emb_model_name, device=device)
        self.corpus = None
        self.corpus_embeddings = None
        self.tokenized_corpus = None
        self.bm25 = None
        self.raranker = CrossEncoder(rer_model, device=device)
    
    
    def rerank_candidates(self, claim, candidate_abstracts):
        pairs = [(claim, ab) for ab in candidate_abstracts]
        scores = self.raranker.predict(pairs)
        return np.argsort(scores)[::-1]

    def load_corpus(self, corpus_path):
        corpus_df = pd.read_parquet(corpus_path)
        corpus_df = corpus_df[corpus_df["abstract"].notnull()]
        corpus_df["abstract"] = "passage: " + corpus_df["abstract"]

        self.corpus = {
            "abstracts": corpus_df["abstract"].tolist(),
            "ids": corpus_df["abstract_id"].tolist()
        }

        cache_path = "tokenized_corpus.pkl"
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.tokenized_corpus = pickle.load(f)
        else:
            print(f"No {cache_path} found. Starting tokenezation with nltk, this can take while...")
            #token_re = re.compile(r"\b\w+\b")
            #self.tokenized_corpus = [token_re.findall(a.lower()) for a in self.corpus["abstracts"]]
            self.tokenized_corpus = [nltk.word_tokenize(abstract) for abstract in self.corpus["abstracts"]]

            with open(cache_path, "wb") as f:
                pickle.dump(self.tokenized_corpus, f)
   
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print("Encoding corpus abstracts...")
        self.corpus_embeddings = self.embedding_model.encode(self.corpus["abstracts"], show_progress_bar=True, batch_size=256)
        self.corpus_embeddings = np.array(self.corpus_embeddings)

    def retrieve_evidence(self, claim):
        bm25_results = self.bm25_retrieve(claim, top_k=self.bm25_top_k)
        claim = f"query: {claim}"
        semantic_results = self.semantic_retrieve(claim, bm25_results)
        reranked_indices = self.rerank_candidates(claim, [r[1] for r in semantic_results])
        reranked = [semantic_results[i] for i in reranked_indices]
        return reranked[:10]

    def bm25_retrieve(self, query, top_k):
        tokenized_query = nltk.word_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return {
            "indices": top_indices,
            "abstracts": [self.corpus["abstracts"][i] for i in top_indices],
            "ids": [self.corpus["ids"][i] for i in top_indices]
        }

    def semantic_retrieve(self, query, bm25_results):
        query_embedding = self.embedding_model.encode([query], batch_size=256)
        bm25_embeddings = self.corpus_embeddings[bm25_results["indices"]]
        similarities = cosine_similarity(query_embedding, bm25_embeddings)[0]
        results = list(zip(
            bm25_results["ids"],
            bm25_results["abstracts"],
            similarities
        ))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:self.cos_sim_top_k]


TAXONOMY = {
    "0_0": "No disinformation narrative",
    "1_0": "Global warming is not happening",
    "1_1": "Ice/permafrost/snow cover isn't melting",
    "1_2": "We're heading into an ice age/global cooling",
    "1_3": "Weather is cold/snowing",
    "1_4": "Climate hasn't warmed/changed over the last (few) decade(s)",
    "1_5": "Oceans are cooling/not warming",
    "1_6": "Sea level rise is exaggerated/not accelerating",
    "1_7": "Extreme weather isn't increasing/has happened before/isn't linked to climate change",
    "1_8": "They changed the name from 'global warming' to 'climate change'",
    "2_0": "Human greenhouse gases are not causing climate change",
    "2_1": "It's natural cycles/variation",
    "2_2": "It's non-greenhouse gas human climate forcings (aerosols, land use)",
    "2_3": "There's no evidence for greenhouse effect/carbon dioxide driving climate change",
    "2_4": "CO2 is not rising/ocean pH is not falling",
    "2_5": "Human CO2 emissions are miniscule/not raising atmospheric CO2",
    "3_0": "Climate impacts/global warming is beneficial/not bad",
    "3_1": "Climate sensitivity is low/negative feedbacks reduce warming",
    "3_2": "Species/plants/reefs aren't showing climate impacts yet/are benefiting from climate change",
    "3_3": "CO2 is beneficial/not a pollutant",
    "3_4": "It's only a few degrees (or less)",
    "3_5": "Climate change does not contribute to human conflict/threaten national security",
    "3_6": "Climate change doesn't negatively impact health",
    "4_0": "Climate solutions won't work",
    "4_1": "Climate policies (mitigation or adaptation) are harmful",
    "4_2": "Climate policies are ineffective/flawed",
    "4_3": "It's too hard to solve",
    "4_4": "Clean energy technology/biofuels won't work",
    "4_5": "People need energy (e_g_, from fossil fuels/nuclear)",
    "5_0": "Climate movement/science is unreliable",
    "5_1": "Climate-related science is uncertain/unsound/unreliable (data, methods & models)",
    "5_2": "Climate movement is alarmist/wrong/political/biased/hypocritical (people or groups)",
    "5_3": "Climate change (science or policy) is a conspiracy (deception)"
}

taxonomy_str = "\n".join([f"{k}: {v}" for k, v in TAXONOMY.items()])

def build_prompt(claim):
    return f"""/no_think You are an expert in detecting climate change related disinformation. You get a claim and your task is to classify the claim using the taxonomy codes provided. 
Rules:
- If the claim aligns with narratives from the taxonomy, list applicable codes separated by a semicolon (;).
- If no disinformation is found, return '0_0'.
- You MUST OUTPUT ONLY one line containing codes and NOTHING ELSE.

Taxonomy:
{taxonomy_str}

Claim: "{claim}"
Codes:"""

def clean_and_validate(pred_text):
    pred_text = re.sub(r'<think>.*?</think>', '', pred_text, flags=re.DOTALL).strip()
    found_codes = re.findall(r'\d_\d', pred_text)
    valid_found = [c for c in found_codes if c in TAXONOMY]
#    if not valid_found:
#        return "0_0"
    return ";".join(sorted(list(set(valid_found))))

@track_emissions(project_name="ClimateCheck2026", save_to_api=True)
def main():
    results_label = "predictions"

    # Task 1.1
    EMB_MODEL = "xplainlp/e5-large-v2-climatecheck"
    retriever = EvidenceRetriever(emb_model_name=EMB_MODEL)

    CORPUS_PATH = "climatecheck_publications_corpus.parquet"
    TEST_DF_PATH = "test-00000-of-00001.parquet"
    test_df = pd.read_parquet(TEST_DF_PATH)
    test_df = test_df[test_df["claim"].notnull()]
    retriever.load_corpus(CORPUS_PATH)
    
    records = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating Predictions"):
        claim_id = row["claim_id"]
        claim_text = row["claim"]

        retrieved = retriever.retrieve_evidence(claim_text)
        for rank, (abstract_id, _, _) in enumerate(retrieved, start=1):
            records.append({
                "claim_id": claim_id,
                "abstract_id": abstract_id,
                "rank": rank,
            })

    submission_df = pd.DataFrame(records)
    submission_df.to_csv(f"{results_label}.csv", index=False)
    
    print(f"Submission file {results_label} is ready.")
    
    # Task 1.2
    model_path = "xplainlp/DeBERTa-v3-large-mnli-fever-anli-ling-wanli-climatecheck"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    with open("label_mapping.json") as f:
        label_map = json.load(f)

    predictions_df = pd.read_csv(f"{results_label}.csv")
    test_df = pd.read_parquet(TEST_DF_PATH)
    abstracts_df = pd.read_parquet(CORPUS_PATH)
    claim_map = test_df.set_index("claim_id")["claim"].to_dict()
    abstract_map = abstracts_df.set_index("abstract_id")["abstract"].to_dict()
    
    def predict_label(claim, abstract):
        text = claim + "[SEP]" + abstract
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        return label_map[str(prediction)]


    labels = []
    for _, row in tqdm(predictions_df.iterrows(), total=len(predictions_df), desc="Predicting labels"):
        claim = claim_map[row["claim_id"]]
        abstract = abstract_map[row["abstract_id"]]
        label = predict_label(claim, abstract)
        labels.append(label)
    
    predictions_df["label"] = labels
    predictions_df.to_csv(f"{results_label}_both.csv", index=False)
    print(f"Saved predictions with labels to {results_label}_both.csv")
    
    # Task 2
    MODEL_DIR = "xplainlp/Qwen3-8B-ClimateCheck"
    OUTPUT_CSV = f"{results_label}_narr.csv"

    if 'raranker' in locals():
        del raranker
    if 'retriever' in locals():
        del retriever
    if 'model' in locals():
        del model
    if 'tokenizer' in locals():
        del tokenizer
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    print("VRAM cleared. Proceeding to vLLM initialization.")
    
    llm = LLM(
        model=MODEL_DIR, 
        trust_remote_code=True
    )

    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        stop=["<|im_end|>", "<|endoftext|>"] 
    )

    test_df = pd.read_parquet(TEST_DF_PATH)
    predictions_df = pd.read_csv(f"{results_label}_both.csv")
    claims = test_df["claim"].astype(str).tolist()

    formatted_prompts = []
    for c in claims:
        user_content = build_prompt(c) + " /no_think"
        messages = [{"role": "user", "content": user_content}]
        
        prompt_with_template = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        formatted_prompts.append(prompt_with_template)

    print(f"Processing {len(formatted_prompts)} claims in batch...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    final_predictions = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        final_predictions.append(clean_and_validate(generated_text))
    
    test_df["narrative"] = final_predictions
    test_df.to_csv("narrative_task_only.csv", index=False)

    predictions_df = predictions_df.merge(test_df[["claim_id", "narrative"]], on="claim_id", how="left")
    predictions_df.to_csv(f"{results_label}_narr.csv", index=False)
    
    print(f"Narrrative Classification complete. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()