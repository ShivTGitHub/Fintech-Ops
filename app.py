import streamlit as st
import pandas as pd
import hdbscan
import requests
from mock_data import generate_switch_logs

st.set_page_config(page_title="FinTech Ops Copilot", layout="wide")

# --- Deterministic Schema Mapper ---
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

# --- Ollama API Calls (No PyTorch Required) ---
def get_ollama_embeddings(texts):
    """Fetches vector embeddings directly from the local Ollama server."""
    try:
        response = requests.post(
            "http://localhost:11434/api/embed", 
            json={"model": "all-minilm", "input": texts}
        )
        return response.json().get("embeddings", [])
    except Exception as e:
        st.error(f"Failed to connect to Ollama Embeddings API: {e}")
        return []

def generate_diagnosis(cluster_logs):
    """Fetches the LLM diagnosis from the local Ollama server."""
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

# --- Main UI Dashboard ---
st.title("FinTech Ops Copilot: AI Observability & Integration")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("1. Transaction Observability (Incident Triage)")
    if st.button("Ingest & Analyze Recent Logs"):
        with st.spinner("Ingesting logs & fetching embeddings via Ollama..."):
            df = generate_switch_logs()
            st.write(f"Ingested {len(df)} logs. Applying semantic clustering...")
            
            # Use Ollama for embeddings instead of PyTorch
            embeddings = get_ollama_embeddings(df['error_msg'].tolist())
            
            if embeddings:
                # Cluster the embeddings using HDBSCAN
                clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
                df['cluster'] = clusterer.fit_predict(embeddings)
                
                anomalies = df[df['cluster'] != -1]
                
                if not anomalies.empty:
                    st.error(f"🚨 Systemic Anomaly Detected! ({len(anomalies)} related events)")
                    st.dataframe(anomalies[['timestamp', 'bank_id', 'error_msg', 'response_code']])
                    
                    st.markdown("**LLM Diagnostic Assessment:**")
                    with st.spinner("Generating AI diagnosis..."):
                        diagnosis = generate_diagnosis(anomalies)
                        st.info(diagnosis)
                else:
                    st.success("No systemic anomalies detected. Operational noise is within normal limits.")

with col2:
    st.subheader("2. Legacy CBS Integration")
    st.markdown("Automated Schema Normalization mapping non-standard CBS payloads to NPCI formats.")
    
    legacy_input = {
        "txn_amt": "1500",
        "merch_id": "M_9921",
        "term_loc": "PUNE_RURAL",
        "response_code": "00"
    }
    
    st.markdown("**Raw Legacy CBS Payload:**")
    st.json(legacy_input)
    
    if st.button("Normalize to NPCI Schema"):
        npc_payload = normalize_cbs_payload(legacy_input)
        st.markdown("**Validated NPCI Payload:**")
        st.json(npc_payload)
        st.success("Transformation completed deterministically via approved mapping rules.")