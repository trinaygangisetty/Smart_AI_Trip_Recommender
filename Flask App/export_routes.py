from flask import Blueprint, send_file, request
from google.cloud import bigquery
import pandas as pd
import io
from fpdf import FPDF

export_bp = Blueprint('export', __name__)

def fetch_all_recommendations(budget, preference, month):
    client = bigquery.Client()

    params = [
        bigquery.ScalarQueryParameter("budget", "INT64", budget),
        bigquery.ScalarQueryParameter("preference", "STRING", preference.lower()),
        bigquery.ScalarQueryParameter("month", "STRING", month.title())
    ]

    queries = {
        "Top Picks": """
            SELECT 'Top Picks' AS type, d.destination, d.tags, d.avg_cost_usd, w.weather, w.seasonal_rating
            FROM `trip-recommendation-project.travel_data.destinations` d
            JOIN `trip-recommendation-project.travel_data.weather` w ON d.destination = w.destination
            WHERE d.avg_cost_usd <= @budget AND LOWER(d.tags) LIKE CONCAT('%', @preference, '%') AND w.month = @month
        """,
        "Hidden Gems": """
            SELECT 'Hidden Gems' AS type, d.destination, d.tags, d.avg_cost_usd, w.weather, w.seasonal_rating,
                   ROUND(SAFE_DIVIDE(w.seasonal_rating, d.avg_cost_usd / 1000), 2) AS hidden_gem_score
            FROM `trip-recommendation-project.travel_data.destinations` d
            JOIN `trip-recommendation-project.travel_data.weather` w ON d.destination = w.destination
            WHERE d.avg_cost_usd <= @budget AND w.month = @month
            ORDER BY hidden_gem_score DESC
        """,
        "Wildcard Picks": """
            SELECT 'Wildcard Picks' AS type, d.destination, d.tags, d.avg_cost_usd, w.weather, w.seasonal_rating
            FROM `trip-recommendation-project.travel_data.destinations` d
            JOIN `trip-recommendation-project.travel_data.weather` w ON d.destination = w.destination
            WHERE d.avg_cost_usd <= @budget AND LOWER(d.tags) NOT LIKE CONCAT('%', @preference, '%')
              AND w.month = @month AND w.seasonal_rating >= 3.5
        """,
        "Best Value": """
            SELECT 'Best Value' AS type, d.destination, d.tags, d.avg_cost_usd, w.weather, w.seasonal_rating,
                   ROUND(d.avg_cost_usd / w.seasonal_rating, 0) AS value_score
            FROM `trip-recommendation-project.travel_data.destinations` d
            JOIN `trip-recommendation-project.travel_data.weather` w ON d.destination = w.destination
            WHERE d.avg_cost_usd <= @budget AND w.month = @month AND w.seasonal_rating >= 3.5
        """,
        "Peak Season Picks": """
            SELECT 'Peak Season Picks' AS type, d.destination, d.tags, d.avg_cost_usd, w.weather, w.seasonal_rating
            FROM `trip-recommendation-project.travel_data.destinations` d
            JOIN `trip-recommendation-project.travel_data.weather` w ON d.destination = w.destination
            WHERE d.avg_cost_usd <= @budget AND w.month = @month AND w.seasonal_rating >= 4.5
        """
    }

    dfs = []
    for label, sql in queries.items():
        job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
        dfs.append(job.to_dataframe())

    return pd.concat(dfs, ignore_index=True)


@export_bp.route('/export/csv')
def export_csv():
    budget = int(request.args.get("budget", 300000))
    preference = request.args.get("preference", "adventure")
    month = request.args.get("month", "July")
    df = fetch_all_recommendations(budget, preference, month)

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return send_file(
        io.BytesIO(buffer.read().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name="recommendations.csv"
    )


@export_bp.route('/export/pdf')
def export_pdf():
    budget = int(request.args.get("budget", 300000))
    preference = request.args.get("preference", "adventure")
    month = request.args.get("month", "July")
    df = fetch_all_recommendations(budget, preference, month)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)
    pdf.set_fill_color(230, 230, 250)

    # Header
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Trip Recommendations", ln=True, align='C')
    pdf.ln(5)

    # User input summary
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 8, txt=f"Age: {request.args.get('age', 'N/A')}", ln=True)
    pdf.cell(200, 8, txt=f"Budget: ${budget:,}", ln=True)
    pdf.cell(200, 8, txt=f"Travel Style: {preference}", ln=True)
    pdf.cell(200, 8, txt=f"Month: {month}", ln=True)
    pdf.ln(5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    for i, row in df.iterrows():
        pdf.ln(5)
        for col in df.columns:
            pdf.cell(200, 8, txt=f"{col}: {row[col]}", ln=True)
        pdf.ln(3)
    
    pdf_bytes = pdf.output(dest='S').encode('latin1')

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name="recommendations.pdf"
    )

