import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from resume_parser import extract_text_from_pdf

st.set_page_config(page_title="Job SkillMap", layout="wide")

# --- Load Dataset ---
try:
    df = pd.read_csv("data/Cleaned_DS_Jobs.csv")
    df['Clean_Description'] = df['Job Description'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)).lower())
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# --- Load Skills ---
try:
    with open("skills.txt", "r") as f:
        skill_list = [line.strip().lower() for line in f if line.strip()]
except:
    st.error("skills.txt file not found or empty.")
    st.stop()

# --- Skill Matching ---
def match_skills(text):
    matched = [s for s in skill_list if s in text.lower()]
    missing = [s for s in skill_list if s not in text.lower()]
    return matched, missing

# --- Clustering ---
def cluster_jobs(df, k=4):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=10)
    X = vectorizer.fit_transform(df['Clean_Description'])
    model = KMeans(n_clusters=k, random_state=42)
    df['cluster'] = model.fit_predict(X)
    centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    cluster_keywords = {i: [terms[ind] for ind in centroids[i, :5]] for i in range(k)}
    return df, cluster_keywords

# --- Skill Presence Columns ---
for skill in skill_list:
    col = f"has_{skill.replace(' ', '_')}"
    if col not in df.columns:
        df[col] = df['Clean_Description'].apply(lambda x: 1 if skill in x else 0)

# --- Streamlit UI ---
st.title("🧠 Job SkillMap: Resume Skill Matcher & Job Cluster Explorer")

# Sidebar for Upload
with st.sidebar:
    uploaded_file = st.file_uploader("📄 Upload your resume (PDF)", type=["pdf"])

# --- Resume Matching ---
if uploaded_file:
    try:
        resume_text = extract_text_from_pdf(uploaded_file)
        matched, missing = match_skills(resume_text)

        st.success(f"✅ Skills Found: {', '.join(matched) if matched else 'None'}")
        st.warning(f"❌ Skills Missing: {', '.join(missing) if missing else 'None'}")

        def qualifies(row):
            return all(row.get(f"has_{s.replace(' ', '_')}", 0) == 1 for s in matched)

        df['match_score'] = df.apply(qualifies, axis=1)
        matched_jobs = df[df['match_score'] == True]

        st.markdown(f"🎯 You qualify for **{len(matched_jobs)} / {len(df)}** jobs.")
        st.dataframe(matched_jobs[['Job Title', 'Company Name', 'Location']].head(10))

    except Exception as e:
        st.error(f"Error reading resume: {e}")

# --- Skill Frequency ---
st.subheader("📊 Most In-Demand Skills in Dataset")
skill_cols = [f"has_{s.replace(' ', '_')}" for s in skill_list if f"has_{s.replace(' ', '_')}" in df.columns]

if skill_cols:
    skill_counts = df[skill_cols].sum().sort_values(ascending=False)
    if skill_counts.sum() > 0:
        fig, ax = plt.subplots()
        skill_counts.head(10).plot(kind='bar', color='skyblue', ax=ax)
        ax.set_ylabel("Number of Jobs")
        ax.set_title("Top Skills in Job Listings")
        st.pyplot(fig)
    else:
        st.warning("Skill data is present but contains all zeros.")
else:
    st.warning("No matching skill columns found in dataset.")

# --- Clustering ---
st.subheader("🧩 Job Clusters Based on Descriptions")
df, cluster_keywords = cluster_jobs(df)

cluster_desc = pd.DataFrame.from_dict(cluster_keywords, orient='index', columns=[f"Top {i+1}" for i in range(5)])
st.markdown("### 🔑 Cluster Keywords")
st.dataframe(cluster_desc)

selected_cluster = st.selectbox("🔍 Explore Jobs by Cluster", sorted(df['cluster'].unique()))
clustered_jobs = df[df['cluster'] == selected_cluster][['Job Title', 'Company Name', 'Location', 'Clean_Description']]
st.dataframe(clustered_jobs.head(10))
