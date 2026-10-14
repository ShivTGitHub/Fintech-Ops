import streamlit as st
import pandas as pd
import hdbscan
import requests
from sklearn.preprocessing import normalize
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
    You are a strict L1 support engineer for a financial payment switch.
    Analyze these clustered error logs and the runbook to diagnose the issue. 
    
    CRITICAL INSTRUCTION: Your Recommended Action MUST be copied directly from the Runbook Context. Do not invent, guess, or provide generic advice outside of the Runbook.

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
        with st.spinner("Ingesting and correlating logs deterministically..."):
            df = generate_switch_logs()
            st.write(f"Ingested {len(df)} logs. Applying deterministic correlation...")
            
            # DETERMINISTIC CORRELATION: Group by Bank ID and Response Code
            # Find groups that have spiked (e.g., more than 10 errors in a short window)
            spikes = df.groupby(['bank_id', 'response_code']).size().reset_index(name='count')
            anomalies = spikes[spikes['count'] >= 10]
            
            if not anomalies.empty:
                # Isolate the specific logs belonging to the spike
                target_bank = anomalies.iloc[0]['bank_id']
                target_code = anomalies.iloc[0]['response_code']
                
                incident_logs = df[(df['bank_id'] == target_bank) & (df['response_code'] == target_code)]
                
                st.error(f"🚨 Systemic Anomaly Detected! ({len(incident_logs)} related events for {target_bank})")
                st.dataframe(incident_logs[['timestamp', 'bank_id', 'error_msg', 'response_code']])
                
                st.markdown("**LLM Diagnostic Assessment:**")
                with st.spinner("Generating AI diagnosis based on deterministic cluster..."):
                    diagnosis = generate_diagnosis(incident_logs)
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