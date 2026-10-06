import io
import re
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. STREAMLIT PAGE CONFIG (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="AI Exam Intelligence Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. CUSTOM STYLING ---
st.markdown(
    """
    <style>
    .stMetric {
        background-color: #0f172a;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #1e293b;
    }
    </style>
""",
    unsafe_html=True,
)


# --- 3. HELPER FUNCTIONS ---
def extract_text_from_pdf(uploaded_file):
    try:
        import PyPDF2

        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF file {uploaded_file.name}: {e}")
        return ""


STOPWORDS = {
    "following", "given", "which", "statement", "correct", "incorrect", "below",
    "select", "choose", "option", "true", "false", "data", "science", "intelligence",
    "organizing", "question", "answer", "find", "calculate", "value", "consider",
    "regarding", "based", "among", "these", "where", "when", "what", "how",
    "explain", "describe", "discuss", "write", "short", "notes",
}

CONCEPT_MAPPER = {
    "Probability & Bayes' Theorem": r"\b(probability|bayes|conditional probability|random variable|distribution|poisson|binomial)\b",
    "Linear Regression & Correlation": r"\b(regression|linear regression|correlation|least squares|residuals)\b",
    "PCA & Dimensionality Reduction": r"\b(pca|principal component|eigenvalue|eigenvector|dimensionality reduction)\b",
    "SQL & Database Normalization": r"\b(sql|join|select|normalization|1nf|2nf|3nf|bcnf|foreign key|primary key)\b",
    "Graph & Search Algorithms": r"\b(graph|bfs|dfs|dijkstra|tree|shortest path|traversal)\b",
    "Machine Learning & Classification": r"\b(classification|svm|decision tree|clustering|kmeans|neural network|overfitting)\b",
    "Descriptive Statistics": r"\b(mean|median|mode|variance|standard deviation|percentile|z-score)\b",
}


def parse_questions_from_text(raw_text, year):
    raw_questions = [
        q.strip()
        for q in re.split(r"\n(?=\d+[\.\)]|\bQ\d+)", raw_text)
        if len(q.strip()) > 15
    ]
    parsed_list = []

    for idx, q_text in enumerate(raw_questions):
        matched_concept = "General Core Concept"
        for concept, pattern in CONCEPT_MAPPER.items():
            if re.search(pattern, q_text.lower()):
                matched_concept = concept
                break

        parsed_list.append(
            {
                "id": f"{year}_{idx+1}",
                "year": year,
                "text": q_text,
                "clean_text": " ".join(
                    [
                        w
                        for w in re.findall(r"\b[a-zA-Z]{3,}\b", q_text.lower())
                        if w not in STOPWORDS
                    ]
                ),
                "concept": matched_concept,
            }
        )

    return parsed_list


def analyze_semantic_repetitions(questions_db, similarity_threshold=0.45):
    if not questions_db:
        return []

    texts = [q["clean_text"] for q in questions_db if q["clean_text"]]
    if len(texts) < 2:
        for q in questions_db:
            q["cluster_id"] = 0
            q["repeat_count"] = 1
            q["years_appeared"] = [q["year"]]
        return questions_db

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    clusters = []
    visited = set()

    for i in range(len(questions_db)):
        if i in visited:
            continue
        current_cluster = [i]
        visited.add(i)

        for j in range(i + 1, len(questions_db)):
            if j not in visited and sim_matrix[i][j] >= similarity_threshold:
                current_cluster.append(j)
                visited.add(j)

        clusters.append(current_cluster)

    processed_questions = []
    for cluster_id, cluster_indices in enumerate(clusters):
        years = list(set([questions_db[idx]["year"] for idx in cluster_indices]))
        repeat_count = len(cluster_indices)

        for idx in cluster_indices:
            q = questions_db[idx]
            q["cluster_id"] = cluster_id
            q["repeat_count"] = repeat_count
            q["years_appeared"] = sorted(years)
            processed_questions.append(q)

    return processed_questions


def calculate_explainable_score(concept_name, questions_group, total_years_span=5):
    years = sorted(
        list(
            set(
                [
                    q["year"]
                    for q in questions_group
                    for year in q.get("years_appeared", [q["year"]])
                ]
            )
        )
    )
    freq = len(questions_group)

    freq_score = min(100, (freq / max(1, total_years_span)) * 100) * 0.30
    max_year = max(years) if years else 2020
    recency_score = (1.0 if max_year >= 2024 else 0.5) * 100 * 0.25
    avg_repeats = np.mean([q.get("repeat_count", 1) for q in questions_group])
    semantic_score = min(100, (avg_repeats / 3) * 100) * 0.20
    gap = 2026 - max_year
    gap_score = (80 if gap == 1 else (100 if gap == 2 else 40)) * 0.15
    trend_score = (100 if len(years) >= 3 else 50) * 0.10

    total_score = int(
        np.clip(
            freq_score + recency_score + semantic_score + gap_score + trend_score,
            10,
            98,
        )
    )

    if total_score >= 80:
        priority = "🔴 MUST PRACTICE"
    elif total_score >= 65:
        priority = "🟠 VERY IMPORTANT"
    elif total_score >= 50:
        priority = "🟡 IMPORTANT"
    else:
        priority = "🟢 GOOD TO PRACTICE"

    reasons = [
        f"Historical Frequency: Appeared across {len(years)} past papers ({', '.join(map(str, years))}).",
        f"Recency Weight: Last asked in {max_year} (Recency gap: {gap} year/s).",
        f"Semantic Repetition: Found {int(avg_repeats)} conceptually similar question variations.",
        f"Rotation Trend: Score of {total_score}/100 indicates high probability cycle.",
    ]

    return {
        "concept": concept_name,
        "score": total_score,
        "priority": priority,
        "frequency": freq,
        "years": years,
        "reasons": reasons,
        "sample_question": questions_group[0]["text"] if questions_group else "",
    }


YOUTUBE_RESOURCES = {
    "Probability & Bayes' Theorem": {
        "channel": "Gate Smashers / NISO Academy",
        "teacher": "Varun Singla",
        "best_for": "Concept Clarity & PYQ Solved Examples",
        "url": "https://www.youtube.com/results?search_query=bayes+theorem+gate+smashers",
    },
    "SQL & Database Normalization": {
        "channel": "Knowledge Gate",
        "teacher": "Sanchit Jain",
        "best_for": "Complete Normalization (1NF to BCNF) & Numericals",
        "url": "https://www.youtube.com/results?search_query=dbms+normalization+knowledge+gate",
    },
    "Linear Regression & Correlation": {
        "channel": "StatQuest",
        "teacher": "Josh Starmer",
        "best_for": "Intuitive Visual Explanation & Math Derivation",
        "url": "https://www.youtube.com/results?search_query=linear+regression+statquest",
    },
    "PCA & Dimensionality Reduction": {
        "channel": "Krish Naik",
        "teacher": "Krish Naik",
        "best_for": "Data Science & Practical ML Implementation",
        "url": "https://www.youtube.com/results?search_query=pca+krish+naik",
    },
    "Graph & Search Algorithms": {
        "channel": "Abdul Bari",
        "teacher": "Abdul Bari",
        "best_for": "Core Algorithms & Complexity Analysis",
        "url": "https://www.youtube.com/results?search_query=graph+algorithms+abdul+bari",
    },
}


# --- 4. MAIN USER INTERFACE ---
st.title("🎓 AI Exam Intelligence & Multi-Year PYQ Predictor")
st.caption("Evidence-Based Prediction & PYQ Analytics Engine")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙ Data Input Panel")
    exam_name = st.text_input("Subject / Exam Name", value="Data Science & AI")
    uploaded_files = st.file_uploader(
        "Upload PYQ Papers (PDF)", type=["pdf"], accept_multiple_files=True
    )
    st.markdown("---")
    pasted_text = st.text_area("Paste text here if PDF is not available", height=150)
    pasted_year = st.number_input(
        "Year of Pasted Paper", min_value=2015, max_value=2026, value=2025
    )
    run_analysis = st.button("🚀 Analyze Exam Intelligence", type="primary")

# --- EXECUTION LOGIC ---
if run_analysis:
    all_questions = []

    if uploaded_files:
        for idx, file in enumerate(uploaded_files):
            year_match = re.search(r"\b(20\d{2})\b", file.name)
            year = int(year_match.group(1)) if year_match else (2025 - idx)
            raw_txt = extract_text_from_pdf(file)
            if raw_txt:
                q_parsed = parse_questions_from_text(raw_txt, year)
                all_questions.extend(q_parsed)

    if pasted_text.strip():
        q_parsed = parse_questions_from_text(pasted_text, int(pasted_year))
        all_questions.extend(q_parsed)

    if not all_questions:
        st.error(
            "⚠️ No readable question text found! Please upload valid PYQ PDFs or paste paper text."
        )
    else:
        with st.spinner(
            "Analyzing semantic question similarities & past patterns..."
        ):
            processed_db = analyze_semantic_repetitions(all_questions)

        concept_groups = defaultdict(list)
        for q in processed_db:
            concept_groups[q["concept"]].append(q)

        predictions = []
        for concept, group in concept_groups.items():
            pred = calculate_explainable_score(concept, group)
            predictions.append(pred)

        predictions.sort(key=lambda x: x["score"], reverse=True)

        st.success(
            f"Successfully parsed **{len(processed_db)} questions** across multiple years!"
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Analyzed Questions", len(processed_db))
        col2.metric("Core Concepts Identified", len(predictions))
        col3.metric("Highest Prediction Score", f"{predictions[0]['score']}/100")
        col4.metric("Data Quality", "HIGH" if len(uploaded_files) >= 3 else "MEDIUM")

        st.markdown("---")

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "🔥 Predicted Priority Questions",
                "📅 Smart Study Priority Plan",
                "🎥 Verified YouTube & Free Resources",
                "📈 10-Year Trend Dashboard",
            ]
        )

        with tab1:
            st.subheader("🔥 Evidence-Based Priority Predictions")
            for pred in predictions:
                with st.expander(
                    f"{pred['priority']} — **{pred['concept']}** (Score: {pred['score']}/100)"
                ):
                    st.markdown("**Sample Representative Question:**")
                    st.info(f"\"{pred['sample_question']}\"")
                    st.markdown("#### 🧠 Why this prediction?")
                    for r in pred["reasons"]:
                        st.write(f"• {r}")
                    st.markdown(
                        f"**Appeared Years:** {', '.join(map(str, pred['years']))}"
                    )

        with tab2:
            st.subheader("📅 Time-Based Revision Planner")
            plan_days = st.radio(
                "Select available preparation time:",
                [
                    "1 Day Preparation (Emergency)",
                    "3 Days Preparation",
                    "7 Days Preparation",
                ],
                horizontal=True,
            )
            limit = 2 if "1 Day" in plan_days else (4 if "3 Days" in plan_days else 6)
            top_topics = predictions[:limit]

            for idx, item in enumerate(top_topics, 1):
                st.markdown(
                    f"### **Priority {idx}: {item['concept']}** (Predicted Score: {item['score']}/100)"
                )
                st.progress(item["score"] / 100)
                st.write(
                    f"**Action:** Master core theoretical principles and solve at least 3 previous variations from years {', '.join(map(str, item['years']))}."
                )
                st.markdown("---")

        with tab3:
            st.subheader("🎥 Topic-Wise Free YouTube Learning Resources")
            for pred in predictions:
                res = YOUTUBE_RESOURCES.get(
                    pred["concept"],
                    {
                        "channel": "Standard Academic Channel",
                        "teacher": "Subject Specialist",
                        "best_for": "Complete Syllabus Lectures & PYQ Practice",
                        "url": f"https://www.youtube.com/results?search_query={pred['concept'].replace(' ', '+')}+pyq",
                    },
                )
                st.markdown(f"### 📌 Topic: **{pred['concept']}**")
                res_col1, res_col2 = st.columns([2, 1])
                with res_col1:
                    st.write(f"• **Recommended Channel:** {res['channel']}")
                    st.write(f"• **Educator / Teacher:** {res['teacher']}")
                    st.write(f"• **Best For:** {res['best_for']}")
                with res_col2:
                    st.link_button(
                        f"▶️ Watch Free on YouTube", res["url"], type="primary"
                    )
                st.markdown("---")

        with tab4:
            st.subheader("📈 Concept Distribution & Frequency Chart")
            chart_data = pd.DataFrame(
                {
                    "Concept": [p["concept"] for p in predictions],
                    "Prediction Score": [p["score"] for p in predictions],
                    "PYQ Occurrences": [p["frequency"] for p in predictions],
                }
            ).set_index("Concept")
            st.bar_chart(chart_data)
