# =========================================================
# AI CUSTOMER SEGMENTATION DASHBOARD
# PREMIUM MODERN STREAMLIT UI
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Customer Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PREMIUM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"]{
    background: linear-gradient(135deg,#0f172a,#020617);
    color: white;
    font-family: 'Poppins', sans-serif;
}

[data-testid="stSidebar"]{
    background: rgba(15,23,42,0.92);
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] *{
    color: white !important;
}

.block-container{
    padding-top: 2rem !important;
    max-width: 1450px;
}

.hero{
    background:
             linear-gradient(135deg,#0f0c29,#302b63,#24243e);

    border-radius: 30px;

    padding: 45px;

    margin-bottom: 30px;

    border: 1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(16px);

    box-shadow:
        0 10px 40px rgba(0,0,0,0.35);
}

.hero h1{
    font-size: 3.4rem !important;
    font-weight: 800 !important;
    color: white;
}

.hero p{
    color: #cbd5e1;
    font-size: 1.1rem;
}

.metric-card{
    background: rgba(255,255,255,0.05);

    border-radius: 24px;

    padding: 25px;

    border: 1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(12px);

    transition: 0.3s ease;
}

.metric-card:hover{
    transform: translateY(-5px);
}

.metric-title{
    color: #94a3b8;
    font-size: 0.95rem;
}

.metric-value{
    color: white;
    font-size: 2.1rem;
    font-weight: 700;
}

.stButton > button{

    background:
        linear-gradient(
            135deg,
            #6366f1,
            #3b82f6
        );

    border:none;

    color:white;

    border-radius:14px;

    padding:0.75rem 1rem;

    font-weight:700;

    width:100%;
}

.stDownloadButton > button{

    background:
        linear-gradient(
            135deg,
            #10b981,
            #059669
        );

    border:none;

    color:white;

    border-radius:14px;

    padding:0.75rem 1rem;

    font-weight:700;

    width:100%;
}

div[data-testid="metric-container"]{

    background: rgba(255,255,255,0.04);

    border:1px solid rgba(255,255,255,0.08);

    padding:20px;

    border-radius:22px;
}

table{
    border-collapse: collapse !important;
}

thead tr{
    background:#1e293b !important;
}

tbody tr{
    background: rgba(15,23,42,0.75);
}

tbody tr:nth-child(even){
    background: rgba(30,41,59,0.75);
}

thead th{
    color:#60a5fa !important;
}

tbody td{
    color:#e2e8f0 !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

def page_header(title, subtitle=""):

    html_code = f"""
    <div class="hero">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """

    st.markdown(
        html_code,
        unsafe_allow_html=True
    )

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data(uploaded=None):

    if uploaded is not None:
        return pd.read_csv(uploaded)

    if os.path.exists("mall_customers.csv"):
        return pd.read_csv("mall_customers.csv")

    st.error("CSV file not found")
    st.stop()

# =========================================================
# PREPROCESS
# =========================================================

def preprocess(df, features):

    X = df[features]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled

# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(X_scaled, n_clusters):

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    return model, labels

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# 🚀 AI Customer Analytics")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Dataset",
            "Elbow Method",
            "Train Model",
            "Cluster Analysis",
            "Insights"
        ]
    )

    st.markdown("---")

    n_clusters = st.slider(
        "Number of Clusters",
        2,
        8,
        5
    )

    features = st.multiselect(
        "Select Features",
        [
            "Annual Income (k$)",
            "Spending Score (1-100)",
            "Age"
        ],
        default=[
            "Annual Income (k$)",
            "Spending Score (1-100)",
            "Age"
        ]
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

# =========================================================
# LOAD DATA
# =========================================================

df = load_data(uploaded_file)

# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "Overview":

    page_header(
        "🚀 AI Customer Segmentation Dashboard",
        "Machine Learning powered customer analytics using K-Means Clustering"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Customers</div>
            <div class="metric-value">{len(df)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Average Income</div>
            <div class="metric-value">${df['Annual Income (k$)'].mean():.0f}k</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Spending Score</div>
            <div class="metric-value">{df['Spending Score (1-100)'].mean():.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Average Age</div>
            <div class="metric-value">{df['Age'].mean():.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            df,
            x="Annual Income (k$)",
            y="Spending Score (1-100)",
            color="Age",
            size="Age",
            template="plotly_dark",
            title="Customer Spending Behaviour",
            color_continuous_scale="Turbo"
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        pie = px.pie(
            df,
            names="Gender",
            template="plotly_dark",
            hole=0.6,
            title="Gender Distribution"
        )

        pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(
            pie,
            use_container_width=True
        )

# =========================================================
# DATASET PAGE
# =========================================================

elif page == "Dataset":

    page_header(
        "📊 Dataset",
        "Mall Customer Dataset"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.markdown("## 📈 Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

# =========================================================
# ELBOW METHOD
# =========================================================

elif page == "Elbow Method":

    page_header(
        "📉 Elbow Method",
        "Find Optimal Number of Clusters"
    )

    X_scaled = preprocess(df, features)

    inertias = []

    K = range(2,11)

    for k in K:

        km = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        km.fit(X_scaled)

        inertias.append(km.inertia_)

    fig = px.line(
        x=list(K),
        y=inertias,
        markers=True,
        template="plotly_dark"
    )

    fig.update_layout(
        title="Elbow Method",
        xaxis_title="Clusters",
        yaxis_title="WCSS",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# TRAIN MODEL
# =========================================================

elif page == "Train Model":

    page_header(
        "🤖 Train K-Means Model",
        f"Training with {n_clusters} clusters"
    )

    if st.button("🚀 Train Model"):

        X_scaled = preprocess(df, features)

        model, labels = train_model(
            X_scaled,
            n_clusters
        )

        silhouette = silhouette_score(
            X_scaled,
            labels
        )

        df_clustered = df.copy()

        df_clustered["Cluster"] = labels + 1

        st.session_state["df_clustered"] = df_clustered
        st.session_state["labels"] = labels
        st.session_state["n_clusters"] = n_clusters

        st.success("Model Trained Successfully")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Clusters",
            n_clusters
        )

        c2.metric(
            "Silhouette Score",
            f"{silhouette:.3f}"
        )

        c3.metric(
            "Inertia",
            f"{model.inertia_:.0f}"
        )

        model_bytes = pickle.dumps(model)

        st.download_button(
            "⬇ Download Model",
            model_bytes,
            "kmeans_model.pkl"
        )

# =========================================================
# CLUSTER ANALYSIS
# =========================================================

elif page == "Cluster Analysis":

    page_header(
        "📌 Cluster Analysis",
        "Customer Segmentation Visualization"
    )

    if "df_clustered" not in st.session_state:

        st.warning("Train the model first")

    else:

        df_clustered = st.session_state["df_clustered"]

        fig = px.scatter(
            df_clustered,
            x="Annual Income (k$)",
            y="Spending Score (1-100)",
            color=df_clustered["Cluster"].astype(str),
            size="Age",
            template="plotly_dark",
            title="Customer Clusters"
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            df_clustered,
            use_container_width=True
        )

# =========================================================
# INSIGHTS
# =========================================================

elif page == "Insights":

    page_header(
        "💡 Business Insights",
        "Customer Behaviour Analysis"
    )

    if "df_clustered" not in st.session_state:

        st.warning("Train the model first")

    else:

        df_clustered = st.session_state["df_clustered"]

        cluster_stats = df_clustered.groupby("Cluster")[
            [
                "Age",
                "Annual Income (k$)",
                "Spending Score (1-100)"
            ]
        ].mean()

        st.dataframe(
            cluster_stats,
            use_container_width=True
        )

        st.markdown("## 🎯 Recommendations")

        st.success("Premium customers can be targeted with luxury products.")
        st.info("Low spending customers can receive discounts and offers.")
        st.warning("Create loyalty programs for medium-spending customers.")
        st.error("Personalized marketing improves customer retention.")