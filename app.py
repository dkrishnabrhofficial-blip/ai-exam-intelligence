import streamlit as st
import re
from collections import Counter
import pandas as pd

# Page Setup
st.set_page_config(
    page_title="AI Exam Intelligence",
    page_icon="🎓",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header { font-size: 32px; font-weight: bold; color: #4A154B; }
    .sub-header { font-size: 18px; color: #555555; margin-bottom: 20px; }
    .metric-card { background-color: #F8F9FA; padding: 15px; border-radius: 10px; border-left: 5px solid #4A154B; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎓 AI Exam Intelligence & PYQ Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Offline, Frugal & Explainable Topic Prediction Engine</div>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar Configuration
st.sidebar.header("📁 Upload PYQ Dataset")
uploaded_files = st.sidebar.file_uploader(
    "Upload 10-Year Previous Year Question PDFs or Text Files",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ System Specs")
st.sidebar.text("Mode: 100% Offline (Edge)")
st.sidebar.text("Inference: CPU-Bound ML")
st.sidebar.text("Privacy: Zero Cloud Transmission")

# Core Helper Functions
def extract_text_from_file(uploaded_file):
    text = ""
    try:
        if uploaded_file.name.endswith(".pdf"):
            import pypdf
            reader = pypdf.PdfReader(uploaded_file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
        else:
            text = str(uploaded_file.read().decode("utf-8"))
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")
    return text

def analyze_patterns(raw_text):
    clean_text = re.sub(r'[^a-zA-Z\s]', '', raw_text).lower()
    words = clean_text.split()
    
    stopwords = set(["the", "a", "an", "in", "of", "and", "or", "to", "for", "with", "on", "at", "by", "from", "is", "what", "explain", "describe", "define", "state", "find", "calculate", "prove", "discuss", "show", "using"])
    filtered_words = [w for w in words if w not in stopwords and len(w) > 3]
    
    word_counts = Counter(filtered_words)
    total_count = sum(word_counts.values()) or 1
    
    results = []
    for word, count in word_counts.most_common(10):
        prob = round((count / total_count) * 100, 2)
        weight = "High" if prob > 5 else ("Medium" if prob > 2 else "Standard")
        xai_reason = f"Appeared {count} times across papers. Represents ~{prob}% term density in historical PYQ corpus."
        results.append({
            "Topic / Keyword": word.capitalize(),
            "Occurrence": count,
            "Predicted Probability (%)": prob,
            "Priority Level": weight,
            "XAI Reasoning": xai_reason
        })
    return pd.DataFrame(results)

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Topic Predictions & XAI", "📊 Frequency Matrix", "📑 Raw PYQ Corpus"])

if uploaded_files:
    all_text = ""
    with st.spinner("Extracting and processing PYQ documents..."):
        for f in uploaded_files:
            all_text += extract_text_from_file(f) + " "
    
    if all_text.strip():
        df_results = analyze_patterns(all_text)
        
        with tab1:
            st.header("Predicted High-Yield Topics")
            st.write("These topics have the highest statistical likelihood of appearing in upcoming examinations based on 10-year historical term density.")
            
            st.dataframe(df_results, use_container_width=True)
            
            st.markdown("### 🧠 Explainable AI (XAI) Insight")
            top_topic = df_results.iloc[0]["Topic / Keyword"]
            top_prob = df_results.iloc[0]["Predicted Probability (%)"]
            st.success(f"**Primary Focus Recommendation:** Focus heavily on **{top_topic}**. It holds a statistical density of **{top_prob}%** in the dataset.")

        with tab2:
            st.header("10-Year Frequency Distribution")
            chart_df = df_results[["Topic / Keyword", "Occurrence"]].set_index("Topic / Keyword")
            st.bar_chart(chart_df)

        with tab3:
            st.header("Raw PYQ Text Preview")
            st.write(f"Total Text Characters Extracted: **{len(all_text)}**")
            st.text_area("Corpus Content", all_text[:2000] + ("..." if len(all_text) > 2000 else ""), height=250)
    else:
        st.warning("Could not extract text from the uploaded files. Please check file formatting.")

else:
    with tab1:
        st.info("👈 Please upload your PYQ PDFs or TXT files from the sidebar to generate prediction models.")
        
        st.markdown("### 🔍 Sample System Output (Demo)")
        sample_df = pd.DataFrame([
            {"Topic / Keyword": "Dynamic Programming", "Occurrence": 42, "Predicted Probability (%)": 8.4, "Priority Level": "High", "XAI Reasoning": "Appeared 42 times across papers. High historical recurrence in core algorithms section."},
            {"Topic / Keyword": "Graph Traversal", "Occurrence": 31, "Predicted Probability (%)": 6.2, "Priority Level": "High", "XAI Reasoning": "Appeared 31 times across papers. Frequent requirement in short-answer section."},
            {"Topic / Keyword": "Eigenvalues", "Occurrence": 19, "Predicted Probability (%)": 3.8, "Priority Level": "Medium", "XAI Reasoning": "Appeared 19 times across papers. Steady presence in Linear Algebra syllabus."}
        ])
        st.dataframe(sample_df, use_container_width=True)

    with tab2:
        st.header("Sample Distribution Chart")
        sample_chart = pd.DataFrame({
            "Topic": ["Dynamic Programming", "Graph Traversal", "Eigenvalues"],
            "Occurrence": [42, 31, 19]
        }).set_index("Topic")
        st.bar_chart(sample_chart)

    with tab3:
        st.caption("No files uploaded yet.")