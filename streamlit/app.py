import streamlit as st
import redis
import json
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Vial MCP Dashboard", layout="wide")

# Redis connection
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=int(os.getenv("REDIS_PORT", 6379)), decode_responses=True)

def get_metrics():
    try:
        metrics = redis_client.lrange("api_metrics", 0, -1)
        return [json.loads(m) for m in metrics]
    except Exception as e:
        st.error(f"Failed to retrieve metrics: {str(e)}")
        with open("db/errorlog.md", "a") as f:
            f.write(f"- **[{datetime.datetime.utcnow().isoformat()}]** Streamlit metrics error: {str(e)}\n")
        return []

st.title("Vial MCP Controller Dashboard")

# API Metrics Visualization
st.header("API Metrics")
metrics = get_metrics()
if metrics:
    df = pd.DataFrame(metrics)
    st.subheader("Request Latency")
    fig = px.line(df, x="timestamp", y="latency", color="endpoint", title="API Request Latency Over Time")
    st.plotly_chart(fig)

    st.subheader("Status Code Distribution")
    status_counts = df["status_code"].value_counts()
    fig = px.pie(values=status_counts.values, names=status_counts.index, title="API Status Codes")
    st.plotly_chart(fig)

# Data Retrieval Results
st.header("Data Retrieval Results")
source = st.selectbox("Select Data Source", ["postgres", "milvus", "weaviate", "pgvector", "faiss"])
query = st.text_input("Enter Query")
if st.button("Retrieve"):
    try:
        response = requests.post(
            "http://unified_server:8000/v1/api/retrieve",
            json={"user_id": "user123", "query": query, "source": source, "wallet": {}, "format": "json"},
            headers={"Authorization": f"Bearer {os.getenv('API_KEY')}"}
        )
        response.raise_for_status()
        st.json(response.json())
    except Exception as e:
        st.error(f"Retrieval failed: {str(e)}")
        with open("db/errorlog.md", "a") as f:
            f.write(f"- **[{datetime.datetime.utcnow().isoformat()}]** Streamlit retrieval error: {str(e)}\n")
