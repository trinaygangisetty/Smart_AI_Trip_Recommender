import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google.cloud import bigquery
import openai
from openai import OpenAI
import ast
import os
import json
import tempfile
from google.oauth2 import service_account

with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as temp_file:
    json.dump(dict(st.secrets["gcp_service_account"]), temp_file)
    service_account_path = temp_file.name
    
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Loading embeddings and indexes
@st.cache_resource
def load_destination_data():
    df = pd.read_pickle("destination_embeddings/destination_embeddings.pkl")
    index = faiss.read_index("destination_embeddings/faiss_destinations.index")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, index, model

@st.cache_resource
def load_trip_history_data():
    df = pd.read_pickle("trip_member_embeddings/trip_history_embeddings.pkl")
    index = faiss.read_index("trip_member_embeddings/faiss_trip_history.index")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, index, model

@st.cache_resource
def load_weather_aware_destinations():
    df = pd.read_pickle("destination_weather_embeddings/destination_weather_embeddings.pkl")
    index = faiss.read_index("destination_weather_embeddings/faiss_destinations_weather.index")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, index, model

# side menu Bar
st.title("🌍 AI Trip Recommender Dashboard")
section = st.sidebar.radio("Choose Feature", [
    "📊 BigQuery Recommendations",
    "🔍 RAG - Destination Tags",
    "👥 RAG - Trip History Match",
    "🌦️ Weather-Aware Destination Insights",
    "💬 Gemini Chat Assistant"
])

# ---------------- BIGQUERY RECOMMENDATIONS ------------------
if section == "📊 BigQuery Recommendations":
    st.subheader("Personalized Recommendations from BigQuery")

    st.markdown("""
    ### 🧠 How This Works:
    - 🎯 **Top Picks**: Best matches for your travel style, within your budget.
    - 💎 **Hidden Gems**: High seasonal rating, surprisingly affordable.
    - 🎲 **Wildcard Picks**: Unusual picks outside your norm, but well-rated.
    - 💸 **Best Value**: Most experience for the price.
    - 🌤️ **Peak Season Picks**: Must-visits in the selected month.
    """)

    with st.form("bigquery_form"):
        st.markdown("#### ✍️ Tell us a bit about yourself:")

        age = st.number_input(
            "👤 Age",
            min_value=18,
            max_value=80,
            value=30,
            help="Please enter an age between 18 and 80 years"
        )

        budget = st.number_input(
            "💰 Travel Budget ($)",
            min_value=50000,
            max_value=1000000,
            value=300000,
            step=10000,
            help="Enter your total travel budget (between 80,000 and 700,000)"
        )

        preference = st.selectbox(
            "🧭 Travel Style",
            ["adventure", "luxury", "beach", "cultural", "nature", "romantic"],
            help="What kind of travel experience do you enjoy most?"
        )

        month = st.selectbox(
            "📅 Travel Month",
            pd.date_range("2023-01-01", periods=12, freq="M").strftime("%B"),
            help="When do you plan to travel?"
        )

        submitted = st.form_submit_button("Get Recommendations")


    if submitted:
        credentials = service_account.Credentials.from_service_account_file(service_account_path)
        client = bigquery.Client(credentials=credentials, project=credentials.project_id)

        params = [
            bigquery.ScalarQueryParameter("budget", "INT64", budget),
            bigquery.ScalarQueryParameter("preference", "STRING", preference.lower()),
            bigquery.ScalarQueryParameter("month", "STRING", month.title())
        ]

        def run_query(sql):
            job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
            return job.to_dataframe()

        queries = {
            "🎯 Top Picks": (
                """SELECT d.destination, d.tags, d.avg_cost_usd, w.weather, w.seasonal_rating
                FROM `trip-recommendation-project.travel_data.destinations` d
                JOIN `trip-recommendation-project.travel_data.weather` w ON d.destination = w.destination
                WHERE d.avg_cost_usd <= @budget AND LOWER(d.tags) LIKE CONCAT('%', @preference, '%') AND w.month = @month
                ORDER BY w.seasonal_rating DESC LIMIT 3""",
                "💡 This destination aligns with your travel style and fits your budget."
            ),
            "💎 Hidden Gems": (
                """SELECT d.destination, d.tags, d.avg_cost_usd, w.weather, w.seasonal_rating,
                        ROUND(SAFE_DIVIDE(w.seasonal_rating, d.avg_cost_usd / 1000), 2) AS hidden_gem_score
                FROM `trip-recommendation-project.travel_data.destinations` d
                JOIN `trip-recommendation-project.travel_data.weather` w ON d.destination = w.destination
                WHERE d.avg_cost_usd <= @budget AND w.month = @month
                ORDER BY hidden_gem_score DESC LIMIT 3""",
                "💡 Highly rated this month and surprisingly affordable — a hidden gem!"
            ),
            "🎲 Wildcard Picks": (
                """SELECT d.destination, d.tags, d.avg_cost_usd, w.weather, w.seasonal_rating
                FROM `trip-recommendation-project.travel_data.destinations` d
                JOIN `trip-recommendation-project.travel_data.weather` w ON d.destination = w.destination
                WHERE d.avg_cost_usd <= @budget AND LOWER(d.tags) NOT LIKE CONCAT('%', @preference, '%')
                    AND w.month = @month AND w.seasonal_rating >= 3.5
                ORDER BY w.seasonal_rating DESC LIMIT 3""",
                "💡 Not your usual pick, but great weather and strong reviews."
            ),
            "💸 Best Value": (
                """SELECT d.destination, d.tags, d.avg_cost_usd, w.weather, w.seasonal_rating,
                        ROUND(d.avg_cost_usd / w.seasonal_rating, 0) AS value_score
                FROM `trip-recommendation-project.travel_data.destinations` d
                JOIN `trip-recommendation-project.travel_data.weather` w ON d.destination = w.destination
                WHERE d.avg_cost_usd <= @budget AND w.month = @month AND w.seasonal_rating >= 3.5
                ORDER BY value_score ASC LIMIT 3""",
                "💡 You get the most experience for your dollar here."
            ),
            "🌤️ Peak Season Picks": (
                """SELECT d.destination, d.tags, d.avg_cost_usd, w.weather, w.seasonal_rating
                FROM `trip-recommendation-project.travel_data.destinations` d
                JOIN `trip-recommendation-project.travel_data.weather` w ON d.destination = w.destination
                WHERE d.avg_cost_usd <= @budget AND w.month = @month AND w.seasonal_rating >= 4.5
                ORDER BY w.seasonal_rating DESC LIMIT 3""",
                "💡 This destination is at its best this month!"
            )
        }

        for label, (sql, insight) in queries.items():
            df = run_query(sql)
            st.markdown(f"### {label}")
            if not df.empty:
                st.dataframe(df)
                st.markdown(insight)
            else:
                st.info("No results found for your criteria.")
                
            # --- Combining all results for export ---
        all_results = []
        for label, (sql, _) in queries.items():
            df = run_query(sql)
            if not df.empty:
                df["type"] = label.encode('ascii', 'ignore').decode()
                all_results.append(df)

        if all_results:  # Only show export if there's at least one result
            combined_df = pd.concat(all_results, ignore_index=True)

            # CSV Export
            csv = combined_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download All Recommendations as CSV",
                data=csv,
                file_name="trip_recommendations.csv",
                mime="text/csv"
            )

            # PDF Export
            from fpdf import FPDF
            import io

            def generate_pdf(dataframe):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)

                pdf.cell(200, 10, txt="Trip Recommendations", ln=True, align='C')
                pdf.ln(5)
                
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 8, txt="Traveler Info:", ln=True)
                pdf.set_font("Arial", size=11)
                pdf.cell(200, 8, txt=f"Age: {age}", ln=True)
                pdf.cell(200, 8, txt=f"Budget: ${budget:,}", ln=True)
                pdf.cell(200, 8, txt=f"Travel Style: {preference}", ln=True)
                pdf.cell(200, 8, txt=f"Month: {month}", ln=True)

                pdf.ln(5)
                pdf.line(10, pdf.get_y(), 200, pdf.get_y())  # separator line
                pdf.ln(5)

                for i, row in dataframe.iterrows():
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(200, 8, txt=f"{row['type']} - {row['destination']}", ln=True)
                    pdf.set_font("Arial", size=11)
                    for col in ["tags", "avg_cost_usd", "weather", "seasonal_rating"]:
                        if col in row:
                            value = f"{col.replace('_', ' ').capitalize()}: {row[col]}"
                            pdf.cell(200, 6, txt=value, ln=True)
                    pdf.ln(4)

                return pdf.output(dest='S').encode('latin1')

            pdf_data = generate_pdf(combined_df)
            st.download_button(
                label="⬇️ Download All Recommendations as PDF",
                data=pdf_data,
                file_name="trip_recommendations.pdf",
                mime="application/pdf"
            )


# ------------------- RAG - DESTINATION TAGS ------------------
elif section == "🔍 RAG - Destination Tags":
    st.subheader("RAG-based Recommendations using Destination Tags")

    df, index, model = load_destination_data()
    df["tags"] = df["tags"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    with st.form("rag_form"):
        st.markdown("#### ✍️ Enter your travel preferences:")

        age = st.number_input(
            "👤 Age",
            min_value=18,
            max_value=80,
            value=30,
            help="Please enter an age between 18 and 80 years"
        )

        budget = st.number_input(
            "💰 Budget ($)",
            min_value=50000,
            max_value=1000000,
            value=300000,
            step=10000,
            help="Enter your total travel budget (between 80,000 and 700,000)"
        )

        # Flatten and clean all tags
        all_tags = sorted(set(tag for tags in df["tags"] for tag in tags))
        preference = st.selectbox(
            "🧭 Travel Preference",
            options=all_tags,
            help="Select one primary travel style you'd like to explore"
        )

        month = st.selectbox(
            "📅 Travel Month",
            pd.date_range("2023-01-01", periods=12, freq="M").strftime("%B"),
            help="When do you plan to travel?"
        )

        submitted = st.form_submit_button("Find Matches")

    if submitted:
        st.markdown(f"""
        ### Your Query:
        Suggest {preference} destinations for a {age}-year-old traveler with a budget of ${budget:,} in {month}. Focus on safety and quality experience.
        """)

        query_text = (
            f"Suggest {preference} destinations for a {age}-year-old traveler "
            f"with a budget of ${budget} in {month}. Focus on safety and quality experience."
        )

        filtered_df = df[
            (df["avg_cost_usd"] <= budget) &
            (df["tags"].apply(lambda tags: preference.lower() in [t.lower() for t in tags]))
        ]

        if filtered_df.empty:
            st.warning("No matching destinations found after filtering. Try changing your budget or preference.")
        else:
            # Step 2: Search only within filtered data
            query_vector = model.encode([query_text])[0].astype("float32")
            filtered_vectors = np.array(filtered_df["embedding"].tolist()).astype("float32")

            sub_index = faiss.IndexFlatL2(query_vector.shape[0])
            sub_index.add(filtered_vectors)

            _, sub_indices = sub_index.search(np.array([query_vector]), k=min(5, len(filtered_df)))

            matched_rows = []
            for idx in sub_indices[0]:
                row = filtered_df.iloc[idx].copy()
                row["score"] = float(np.dot(query_vector, row["embedding"]) / (np.linalg.norm(query_vector) * np.linalg.norm(row["embedding"])))
                matched_rows.append(row)

            results = pd.DataFrame(matched_rows)

        st.markdown("### ✨ Recommended Destinations")
        for _, row in results.iterrows():
            st.markdown(f"""
            **{row['destination']} ({row['country']})**  
            Tags: {row['tags']}  
            Cost: ${row['avg_cost_usd']:,}  
            Safety: {row['safety_rating']}  
            Relevance Score: {row['score']:.2f}  
            _Why this was picked?_  
            This destination aligns with your preference for **{preference}** experiences and was semantically matched using AI.
            """)


# ------------------- RAG - TRIP HISTORY ---------------------
elif section == "👥 RAG - Trip History Match":
    st.subheader("Find Similar Travelers Based on Trip History")

    df, index, model = load_trip_history_data()

    with st.form("trip_history"):
        st.markdown("#### ✍️ Describe your traveler profile:")

        age = st.number_input(
            "👤 Age",
            min_value=18,
            max_value=80,
            value=35,
            help="Please enter your age between 18 and 80"
        )

        budget = st.number_input(
            "💰 Budget ($)",
            min_value=50000,
            max_value=1000000,
            value=300000,
            step=10000,
            help="Enter the total budget for travel (between $50,000 and $1,000,000)"
        )

        interest = st.selectbox(
            "🧭 Travel Preference",
            ["adventure", "luxury", "beach", "cultural", "nature", "romantic"],
            help="Choose the type of travel experience you enjoy"
        )

        submitted = st.form_submit_button("Find Matches")

    if submitted:
        query_text = f"A {age}-year-old traveler interested in {interest} with a budget of ${budget}."
        query_vector = model.encode([query_text]).astype("float32")
        distances, indices = index.search(query_vector, 5)

        results = df.iloc[indices[0]].copy()
        results["score"] = distances[0]

        st.markdown(f"""
        ### 🧾 Your Profile Summary
        A {age}-year-old traveler interested in **{interest}** with a budget of ${budget:,}.
        """)

        st.markdown("### 👥 Similar Trip Profiles")
        for _, row in results.iterrows():
            st.markdown(f"""
            **Member ID: {row['member_id']}**

            {row['summary']}

            🔗 **Similarity Score:** {row['score']:.2f}
            """)


# ------------------- WEATHER-AWARE INSIGHTS -------------------
elif section == "🌦️ Weather-Aware Destination Insights":
    st.subheader("AI-Powered Weather-Aware Destination Insights")

    df, index, model = load_weather_aware_destinations()
    df["tags"] = df["tags"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    with st.form("weather_rag_form"):
        st.markdown("#### ✍️ Describe your ideal travel conditions:")

        age = st.number_input("👤 Age", min_value=18, max_value=80, value=30)
        budget = st.number_input("💰 Budget ($)", min_value=50000, max_value=1000000, value=300000, step=10000)

        all_tags = sorted(set(tag for tags in df["tags"] for tag in tags))
        preference = st.selectbox("🧭 Travel Style", options=all_tags)

        month = st.selectbox(
            "📅 Travel Month",
            pd.date_range("2023-01-01", periods=12, freq="M").strftime("%B")
        )

        submitted = st.form_submit_button("Get Smart Recommendations")

    if submitted:
        st.markdown(f"""
        ### 📝 Your Query:
        Find {preference} destinations for a {age}-year-old traveler with a budget of ${budget:,} in {month}, considering safety, cost, and weather.
        """)

        query = (
            f"{preference} destinations for a {age}-year-old traveler "
            f"with a budget of ${budget} in {month}. Focus on safety and weather."
        )
        query_vector = model.encode([query])[0].astype("float32")

        filtered = df[
            (df["avg_cost_usd"] <= budget) &
            (df["month"] == month) &
            (df["tags"].apply(lambda tags: preference.lower() in [t.lower() for t in tags]))
        ]

        if filtered.empty:
            st.warning("No destinations match your filters. Try changing budget or preference.")
        else:
            filtered_vectors = np.array(filtered["embedding"].tolist()).astype("float32")
            sub_index = faiss.IndexFlatL2(query_vector.shape[0])
            sub_index.add(filtered_vectors)
            _, sub_indices = sub_index.search(np.array([query_vector]), k=min(5, len(filtered)))

            matched = []
            for idx in sub_indices[0]:
                row = filtered.iloc[idx].copy()
                row["score"] = float(np.dot(query_vector, row["embedding"]) /
                                     (np.linalg.norm(query_vector) * np.linalg.norm(row["embedding"])))
                matched.append(row)

            results = pd.DataFrame(matched)

            st.markdown("### ✨ Smart Destination Recommendations")

            for _, row in results.iterrows():
                st.markdown(f"""
                **{row['destination']} ({row['country']})**  
                Tags: {row['tags']}  
                Month: {row['month']}  
                Weather: {row['weather']}  
                Cost: ${row['avg_cost_usd']:,}  
                Safety Rating: {row['safety_rating']}  
                Seasonal Rating: {row['seasonal_rating']}  
                Relevance Score: {row['score']:.2f}
                """)

                # Smart insights
                if row["seasonal_rating"] >= 4.5:
                    st.markdown("☀️ **Perfect weather this month — peak season!**")
                elif row["seasonal_rating"] < 3:
                    st.markdown("⚠️ **Weather may not be ideal this month.**")

                if row["safety_rating"] < 3:
                    st.markdown("🚫 **Low safety rating — consider with caution.**")

                if row["avg_cost_usd"] < budget * 0.5:
                    st.markdown("💸 **Great value for your budget!**")

                st.markdown("_Why this was picked?_  \n"
                            f"This destination aligns with your **{preference}** preference and performed well in semantic matching, weather, and cost.")
                st.markdown("---")

# ---------------------- GEMINI CHAT --------------------------
elif section == "💬 Gemini Chat Assistant":
    st.subheader("Ask Gemini Questions Based on Your CSV Files")

    uploaded_files = st.file_uploader("Upload CSV file(s)", type=["csv"], accept_multiple_files=True)
    query = st.text_area("Ask a question about the data...")

    if st.button("Get Answer") and uploaded_files and query:
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np

            # Load model
            model = SentenceTransformer("all-MiniLM-L6-v2")

            # Step 1: Load and combine text chunks
            combined_texts = []
            chunk_to_text_map = []

            for uploaded_file in uploaded_files:
                df = pd.read_csv(uploaded_file)
                for i, row in df.iterrows():
                    text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
                    combined_texts.append(text)
                    chunk_to_text_map.append(text)

            # Step 2: Embed the chunks
            embeddings = model.encode(combined_texts).astype("float32")

            # Step 3: Search for relevant chunks
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)

            query_vec = model.encode([query]).astype("float32")
            D, I = index.search(query_vec, k=5)

            # Step 4: Combine top chunks
            context = "\n\n".join([chunk_to_text_map[i] for i in I[0]])

            # Step 5: Send to Gemini
            prompt_text = f"""You are a helpful assistant that answers questions using data.

DATA SNIPPETS:
{context}

QUESTION: {query}
ANSWER:"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user",
                    "content": prompt_text
                }]
            )

            st.markdown("### 💬 GPT's Response")
            st.write(response.choices[0].message.content)


        except Exception as e:
            st.error(f"Something went wrong: {e}")


