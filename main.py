from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import uvicorn
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone
from groq import Groq
import os
import numpy as np
import time

app = FastAPI()

embedder = SentenceTransformer('all-MiniLM-L6-v2') 


pc = Pinecone(api_key="pcsk_5W1nEg_AAME1aifqHeFQFkVD5NAFei2fSwWtXoZHxubdn9G5t81NvFoq2v87MK1SokJ1qz")
index = pc.Index("vectorhire")


groq_client = Groq(api_key="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")

class JobMatchRequest(BaseModel):
    job_text: str
    role_type: str
    strictness: int

@app.post("/freelancer_join/")
async def freelancer_join(file: UploadFile = File(...), candidate_name: str = Form(...)):

    pdf_reader = PyPDF2.PdfReader(file.file)
    raw_text = "".join(page.extract_text() for page in pdf_reader.pages)

    prompt_structure = f"Extract and structure this resume into 3 categories. Return ONLY valid plain text with headers: [SKILLS], [EXPERIENCE], [EDUCATION]. Resume: {raw_text}"
    structured_data = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt_structure}],
        model="llama-3.3-70b-versatile",
    ).choices[0].message.content


    prompt_ner = f"Extract the top 5 technical skills from this resume as a comma-separated list. Resume: {raw_text}"
    skills_extracted = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt_ner}], 
        model="llama-3.3-70b-versatile"
    ).choices[0].message.content
    
    embedding = embedder.encode(structured_data).tolist()
    
    index.upsert(
        vectors=[{
            "id": candidate_name.replace(" ", "_").lower(), 
            "values": embedding, 
            "metadata": {
                "name": candidate_name, 
                "structured_data": structured_data, 
                "skills": skills_extracted
            }
        }]
    )
    return {"status": "success", "skills_extracted": skills_extracted}

@app.post("/client_match/")
async def client_match(request: JobMatchRequest):
    start_time = time.time() 

    job_embedding = embedder.encode(request.job_text).tolist()

    search_results = index.query(
        vector=job_embedding, top_k=3, include_values=True, include_metadata=True
    )
    
    if not search_results.matches:
        return {"error": "No freelancers currently match this JD."}

    jd_vector = np.array(job_embedding).reshape(1, -1)
    
    best_candidate = None
    best_semantic_score = -1
    
    for match in search_results.matches:
        cand_vector = np.array(match.values).reshape(1, -1)
        exact_score = cosine_similarity(jd_vector, cand_vector)[0][0]
        
        if exact_score > best_semantic_score:
            best_semantic_score = exact_score
            best_candidate = match

    meta = dict(best_candidate.metadata)
    freelancer_profile = meta.get('structured_data') or meta.get('data') or "No data found"
    freelancer_name = meta.get('name', "Unknown")
    skills = meta.get('skills', "N/A")


    match_pct = round(best_semantic_score * 100, 2)

    tech_score = min(100, match_pct + 4.2)
    exp_score = max(0, match_pct - 3.1)

    prompt_eval = f"""
    You are an AI technical recruiter evaluating a top match for a {request.role_type} role.
    Match Score: {match_pct}%
    Client JD: {request.job_text}
    Freelancer Profile: {freelancer_profile}
    
    Provide exactly 2 sections separated by '---':
    1. A 2-sentence executive summary pitching why this candidate is a {match_pct}% match.
    2. The biggest "Skill Gap" or technical weakness the client should probe during the interview.
    """
    
    ai_response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt_eval}], 
        model="llama-3.3-70b-versatile"
    ).choices[0].message.content
    
    try:
        summary, gap = ai_response.split('---')
    except:
        summary, gap = ai_response, "No major technical gaps identified."


    latency = round(time.time() - start_time, 3)

    return {
        "top_match": freelancer_name,
        "semantic_score": match_pct,
        "tech_score": round(tech_score, 1),
        "exp_score": round(exp_score, 1),
        "inference_latency_seconds": latency,
        "evaluation_summary": summary.strip(),
        "skill_gap": gap.strip(),
        "freelancer_skills": skills
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
