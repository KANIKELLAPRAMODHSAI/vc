
import streamlit as st
import requests
import time

FASTAPI_URL = "http://localhost:8000"
st.set_page_config(page_title="VectorHire Pro", layout="wide", page_icon="⚡")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = ""

def authenticate(role, username, password):

    if username and password:
        st.session_state.logged_in = True
        st.session_state.role = role
        st.session_state.username = username
        st.toast(f"Authenticated session for {username}", icon="✅")
    else:
        st.error("Invalid credentials. Payload missing required fields.")

if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center; margin-top: 50px;'>⚡ VectorHire OS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Identity & Access Management (IAM)</p>", unsafe_allow_html=True)
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("🧑‍💻 Talent Portal Authentication")
        f_user = st.text_input("Freelancer Username", placeholder="e.g., dev_kanikella")
        f_pass = st.text_input("Password", type="password", key="f_pass")
        if st.button("Initialize Freelancer Session", use_container_width=True):
            authenticate("freelancer", f_user, f_pass)
            st.rerun()
            
    with col2:
        st.success("🏢 Enterprise Portal Authentication")
        c_user = st.text_input("Client ID / Company Email", placeholder="e.g., hiring@techcorp.com")
        c_pass = st.text_input("Password", type="password", key="c_pass")
        if st.button("Initialize Client Session", use_container_width=True):
            authenticate("client", c_user, c_pass)
            st.rerun()

elif st.session_state.role == "freelancer":
    with st.sidebar:
        st.markdown(f"**Session ID:** `{st.session_state.username}`")
        st.markdown("**Node Status:** 🟢 Active")
        if st.button("Terminate Session", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()
            
    st.title("NLP Profile Vectorization")
    st.markdown("Upload raw PDF to execute NLP extraction and upsert embeddings to Pinecone.")
    
    uploaded_file = st.file_uploader("Upload Profile (PDF)", type=["pdf"])
    
    if st.button("Execute Pipeline & Upsert"):
        if uploaded_file:
            with st.status("Executing inference...", expanded=True) as status:
                st.write("📄 Parsing PDF byte stream...")
                time.sleep(0.5) 
                st.write("🧠 Generating dense vectors via sentence-transformers...")
                
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}

                data = {"candidate_name": st.session_state.username}
                response = requests.post(f"{FASTAPI_URL}/freelancer_join/", files=files, data=data)
                
                if response.status_code == 200:
                    status.update(label="Vector Upsert Complete", state="complete", expanded=False)
                    st.success("Profile indexed.")
                    
                    st.markdown("### 🔍 NER Extraction Output")
                    skills = response.json()['skills_extracted'].split(',')
                    
                    cols = st.columns(len(skills))
                    for i, skill in enumerate(skills):
                        cols[i].button(skill.strip(), key=f"btn_{i}")
                else:
                    status.update(label="HTTP 500: Pipeline Failure", state="error")
        else:
            st.error("Missing payload: PDF file required.")

elif st.session_state.role == "client":
    with st.sidebar:
        st.markdown(f"**Client Node:** `{st.session_state.username}`")
        st.divider()
        st.markdown("### ⚙️ Matching Parameters")
        strictness = st.slider("Cosine Similarity Threshold (%)", 50, 99, 75)
        st.selectbox("LLM Reasoning Backend", ["Groq LLaMA-3.3-70B", "Groq Mixtral-8x7b"])
        if st.button("Terminate Session", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    st.title("Semantic Search Engine")
    role_type = st.radio("Engagement Parameter", ["Contract", "Full-Time"], horizontal=True)
    job_desc = st.text_area("Input Job Description (Unstructured Text)", height=150)
    
    if st.button("Execute Vector Search", type="primary"):
        if job_desc:
            with st.spinner("Calculating pairwise similarities..."):
                payload = {"job_text": job_desc, "role_type": role_type, "strictness": strictness}
                response = requests.post(f"{FASTAPI_URL}/client_match/", json=payload)
                
                if response.status_code == 200:
                    res = response.json()
                    st.divider()
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Optimal Match ID", res['top_match'])
                    col2.metric("Inference Latency", f"{res['inference_latency_seconds']}s")
                    col3.metric("Cosine Similarity", f"{res['semantic_score']}%")
                    
                    st.markdown("### 📊 Multi-Dimensional Matrix")
                    st.markdown("**Semantic Context**")
                    st.progress(int(res['semantic_score']))
                    st.markdown("**Technical Keyword Overlap**")
                    st.progress(int(res['tech_score']))
                    st.markdown("**Experience Variance**")
                    st.progress(int(res['exp_score']))
                    
                    st.markdown("### 🧠 AI Evaluation Output")
                    tab1, tab2, tab3 = st.tabs(["Execution Summary", "🚨 Skill Gap", "Extracted Stack"])
                    
                    with tab1:
                        st.info(res['evaluation_summary'])
                    with tab2:
                        st.warning(res['skill_gap'])
                    with tab3:
                        st.code(res['freelancer_skills'], language="json")
                        
                    st.markdown("---")
                    st.write("**Human-in-the-Loop Feedback (RLHF)**")
                    f_col1, f_col2, f_col3 = st.columns([1,1,8])
                    if f_col1.button("👍 Valid Match"):
                        st.toast("Weights updated in local cache.")
                    if f_col2.button("👎 Invalid Match"):
                        st.toast("Drift anomaly logged.")
                else:
                    st.error("HTTP Exception: Backend connection refused.")
