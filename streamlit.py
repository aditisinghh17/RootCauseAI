import os
import re
import json
import streamlit as st
import PyPDF2
import nltk
import numpy as np
from rank_bm25 import BM25Okapi 
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file (if available)
load_dotenv()

# Ensure NLTK tokenizer is downloaded
nltk.download("punkt")

# Load Sentence Transformer for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize LLM (Groq API); use environment variable for the API key
client = Groq(api_key=os.getenv("GROQ_API_KEY") or "your_api_key_here")


# PDF Processing Functions


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text_content = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
    return "\n".join(text_content)

def split_into_meaningful_chunks(text):
    """Splits text into meaningful sections based on detected headings."""
    # Pattern to detect headings like "2 Overview", "3.1 Troubleshooting Steps"
    section_pattern = re.compile(r"^\d+(\.\d+)*\s+[A-Z][a-zA-Z\s]+", re.MULTILINE)
    matches = list(section_pattern.finditer(text))
    chunks = []
    for i in range(len(matches)):
        start_idx = matches[i].start()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start_idx:end_idx].strip()
        chunks.append(section_text)
    return chunks


# BM25 + Embeddings Retrieval Functions


def retrieve_relevant_chunks(symptoms, chunks):
    """Retrieve top relevant chunks using BM25 and embeddings.
       Returns a list of tuples: (chunk, confidence_score)
    """
    # Tokenize chunks for BM25
    tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    # Tokenize the symptom text
    symptom_tokens = word_tokenize(symptoms.lower())
    bm25_scores = bm25.get_scores(symptom_tokens)
    
    # Select top 10 candidate chunks based on BM25 score
    top_indices = np.argsort(bm25_scores)[-10:][::-1]
    top_chunks = [chunks[i] for i in top_indices]
    
    # Compute embeddings for the symptom and candidate chunks
    symptom_embedding = embedding_model.encode(symptoms, convert_to_tensor=True)
    chunk_embeddings = embedding_model.encode(top_chunks, convert_to_tensor=True)
    
    # Compute cosine similarity between the symptom and each candidate chunk
    cosine_scores = util.pytorch_cos_sim(symptom_embedding, chunk_embeddings)[0]
    sorted_indices = np.argsort(cosine_scores.cpu().numpy())[::-1]
    
    # Create a list of tuples (chunk, score) for the top 3 candidates
    sorted_candidates = [(top_chunks[i], cosine_scores.cpu().numpy()[i]) for i in sorted_indices]
    return sorted_candidates[:3]


# LLM Classification and JSON Extraction

def extract_json(text):
    """Extracts the first JSON object found in text using regex."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return None

def classify_with_llm(symptoms, chunk):
    """Use LLM to generate Cause Code and Resolution Code with a refined prompt."""
    prompt = f"""
You are an expert in troubleshooting.
Given the following symptom and the relevant troubleshooting manual section, determine:
- **Cause Code:** The root cause of the issue.
- **Resolution Code:** Clear, step-by-step instructions to resolve the issue.

Return your answer strictly as valid JSON with no additional commentary.
The JSON format must be exactly:
{{
  "Cause Code": "<brief, precise cause>",
  "Resolution Code": "<clear, step-by-step resolution>"
}}

### Data Provided:
- Symptom: "{symptoms}"
- Manual Section:
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    
    classification_raw = response.choices[0].message.content.strip()
    # Extract JSON from the response using regex
    json_str = extract_json(classification_raw)
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"error": "JSON decoding failed."}
    else:
        return {"error": "No valid JSON found in the LLM output."}


# Streamlit UI with Sidebar Layout

st.sidebar.title("Troubleshooting Assistant Setup")

# PDF Upload in the sidebar
pdf_file = st.sidebar.file_uploader("Upload Reference Manual (PDF)", type=["pdf"])

# Symptom input in the sidebar
symptoms = st.sidebar.text_area("Enter Customer Symptoms:")

# Button to trigger processing
process_button = st.sidebar.button("Find Relevant Solutions")

# Main Panel Title
st.title("Troubleshooting Assistant Results")

if pdf_file:
    with open("manual.pdf", "wb") as f:
        f.write(pdf_file.read())
    st.sidebar.success("PDF Uploaded Successfully!")

    # Process PDF and prepare manual chunks
    manual_text = extract_text_from_pdf("manual.pdf")
    manual_chunks = split_into_meaningful_chunks(manual_text)
    st.write(f"✅ Extracted {len(manual_chunks)} meaningful chunks from the manual.")

    if process_button:
        if symptoms:
            # Retrieve relevant chunks (with confidence scores)
            relevant_candidates = retrieve_relevant_chunks(symptoms, manual_chunks)
            
            st.subheader("🛠️ Classification Results:")
            # Iterate over each candidate chunk, perform LLM classification and display result with confidence score
            for i, (chunk, score) in enumerate(relevant_candidates):
                classification = classify_with_llm(symptoms, chunk)
                st.markdown(f"**Result from Candidate {i+1} (Confidence Score: {score:.2f}):**")
                st.write(f"**Confidence Score: {score:.2f}**") 
                st.json(classification)
        else:
            st.sidebar.warning("⚠️ Please enter symptoms before proceeding.")
else:
    st.sidebar.info("Please upload a reference manual PDF to begin.")
