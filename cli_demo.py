import torch # MUST BE LINE 1
import pandas as pd
import hdbscan
import requests
import time
from sentence_transformers import SentenceTransformer
from mock_data import generate_switch_logs

def normalize_cbs_payload(legacy_payload):
    mapping_rules = {
        "txn_amt": "transactionAmount",
        "amt": "transactionAmount",
        "merch_id": "merchantIdentifier",
        "term_loc": "terminalLocation"
    }
    normalized = {}
    for key, value in legacy_payload.items():
        standard_key = mapping_rules.get(key, key)
        if standard_key == "transactionAmount":
            normalized[standard_key] = f"{float(value):.2f}"
        else:
            normalized[standard_key] = value
    return normalized

def generate_diagnosis(cluster_logs):
    runbook = """
    RUNBOOK FOR CODE 91 (Issuer/Switch Inoperative):
    1. Verify upstream VPN connectivity to Sponsor Bank.
    2. Check TCP socket health on port 8000.
    3. If timeouts exceed 10 per minute, automatically route to secondary switch.
    """
    
    prompt = f"""
    You are an L1 support engineer for a financial payment switch.
    Analyze these clustered error logs and the runbook to diagnose the issue. 
    Be concise. Do not hallucinate.

    Runbook Context: {runbook}
    
    Error Logs:
    {cluster_logs.to_string()}
    
    Provide a structured response:
    - Suspected Root Cause
    - Recommended Action
    """
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate", 
            json={"model": "qwen2.5:3b", "prompt": prompt, "stream": False}
        )
        return response.json().get("response", "Error generating diagnosis.")
    except Exception as e:
        return f"Failed to connect to local LLM: {str(e)}"

def run_demo():
    print("\n" + "="*60)
    print("🚀 STARTING FINTECH OPS COPILOT CLI DEMO")
    print("="*60 + "\n")

    # --- Phase 1: Ingestion ---
    print("[1/4] Ingesting synthetic ISO 8583 switch logs...")
    df = generate_switch_logs()
    time.sleep(1)
    print(f"      -> Successfully ingested {len(df)} logs.\n")

    # --- Phase 2: Embedding & Clustering ---
    print("[2/4] Loading local embedding model & clustering unstructured logs...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['error_msg'].tolist())
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
    df['cluster'] = clusterer.fit_predict(embeddings)
    
    anomalies = df[df['cluster'] != -1]
    time.sleep(1)
    
    if not anomalies.empty:
        print(f"      -> 🚨 SYSTEMIC ANOMALY DETECTED! ({len(anomalies)} related events grouped via HDBSCAN)")
        print("-" * 60)
        print(anomalies[['timestamp', 'bank_id', 'error_msg', 'response_code']].head(5).to_string(index=False))
        print("      ... (showing first 5 clustered events)")
        print("-" * 60 + "\n")
        
        # --- Phase 3: RAG & LLM Diagnosis ---
        print("[3/4] Passing clustered incident to deterministic RAG engine (Ollama)...")
        diagnosis = generate_diagnosis(anomalies)
        print("\n" + "*"*60)
        print("🤖 AI ROOT CAUSE & ACTION PLAN:")
        print("*"*60)
        print(diagnosis)
        print("*"*60 + "\n")
    else:
        print("      -> No systemic anomalies detected.\n")

    # --- Phase 4: Schema Normalization ---
    print("[4/4] Executing Legacy CBS Schema Normalization...")
    legacy_input = {
        "txn_amt": "1500",
        "merch_id": "M_9921",
        "term_loc": "PUNE_RURAL",
        "response_code": "00"
    }
    
    print(f"      -> RAW INPUT:  {legacy_input}")
    npc_payload = normalize_cbs_payload(legacy_input)
    time.sleep(1)
    print(f"      -> NPCI OUTPUT: {npc_payload}\n")
    
    print("="*60)
    print("✅ DEMO COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_demo()