from flask import Flask, render_template, request
from google.cloud import bigquery
from export_routes import export_bp
from rag_routes import rag_bp
from rag_history_routes import rag_history_bp
from rag_chat_routes import rag_chat_bp


app = Flask(__name__)
app.register_blueprint(export_bp)
app.register_blueprint(rag_bp)
app.register_blueprint(rag_history_bp)
app.register_blueprint(rag_chat_bp)


@app.route('/', methods=['GET', 'POST'])
def index():
    top_picks = hidden_gems = wildcards = best_value = peak_season = None
    
    if request.method == 'POST':
        age = int(request.form['age'])
        budget = int(request.form['budget'])
        preference = request.form['preference']
        month = request.form['month']
        
        result = get_recommendations(age, budget, preference, month)
        
        return render_template(
            'index.html', 
            top_picks=result["top_picks"],
            hidden_gems=result["hidden_gems"],
            wildcards=result["wildcards"],
            best_value=result["best_value"],
            peak_season=result["peak_season"],
            budget=budget,
            preference=preference,
            month=month
        )
        
    return render_template("index.html")

def get_recommendations(age, budget, preference, month):
    client = bigquery.Client()
    
    params = [
        bigquery.ScalarQueryParameter("budget", "INT64", budget),
        bigquery.ScalarQueryParameter("preference", "STRING", preference.lower()),
        bigquery.ScalarQueryParameter("month", "STRING", month.title())
    ]
    
    top_query = """
    SELECT d.destination, d.tags, d.avg_cost_usd, w.weather, w.seasonal_rating
    FROM `trip-recommendation-project.travel_data.destinations` d
    JOIN `trip-recommendation-project.travel_data.weather` w
      ON d.destination = w.destination
    WHERE d.avg_cost_usd <= @budget
      AND LOWER(d.tags) LIKE CONCAT('%', @preference, '%')
      AND w.month = @month
    ORDER BY w.seasonal_rating DESC
    LIMIT 3
    """
    
    gems_query = """
    SELECT d.destination, d.tags, d.avg_cost_usd, w.weather, w.seasonal_rating,
           ROUND(SAFE_DIVIDE(w.seasonal_rating, d.avg_cost_usd / 1000), 2) AS hidden_gem_score
    FROM `trip-recommendation-project.travel_data.destinations` d
    JOIN `trip-recommendation-project.travel_data.weather` w
      ON d.destination = w.destination
    WHERE d.avg_cost_usd <= @budget
      AND w.month = @month
    ORDER BY hidden_gem_score DESC
    LIMIT 3
    """
    
    wild_query = """
    SELECT d.destination, d.tags, d.avg_cost_usd, w.weather, w.seasonal_rating
    FROM `trip-recommendation-project.travel_data.destinations` d
    JOIN `trip-recommendation-project.travel_data.weather` w
      ON d.destination = w.destination
    WHERE d.avg_cost_usd <= @budget
      AND LOWER(d.tags) NOT LIKE CONCAT('%', @preference, '%')
      AND w.month = @month
      AND w.seasonal_rating >= 3.5
    ORDER BY w.seasonal_rating DESC
    LIMIT 3
    """
    
    value_query = """
    SELECT d.destination, d.tags, d.avg_cost_usd, w.weather, w.seasonal_rating,
       ROUND(d.avg_cost_usd / w.seasonal_rating, 0) AS value_score
    FROM `trip-recommendation-project.travel_data.destinations` d
    JOIN `trip-recommendation-project.travel_data.weather` w
       ON d.destination = w.destination
    WHERE d.avg_cost_usd <= @budget
       AND w.month = @month
       AND w.seasonal_rating >= 3.5
    ORDER BY value_score ASC
    LIMIT 3
    """
    
    peak_query = """
    SELECT d.destination, d.tags, d.avg_cost_usd, w.weather, w.seasonal_rating
    FROM `trip-recommendation-project.travel_data.destinations` d
    JOIN `trip-recommendation-project.travel_data.weather` w
        ON d.destination = w.destination
    WHERE d.avg_cost_usd <= @budget
        AND w.month = @month
        AND w.seasonal_rating >= 4.5
    ORDER BY w.seasonal_rating DESC
    LIMIT 3
    """
    
    
    top = client.query(top_query, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
    gems = client.query(gems_query, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
    wcards = client.query(wild_query, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
    value = client.query(value_query, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
    peak = client.query(peak_query, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
    
    
    return {
        "top_picks": [dict(row) for row in top],
        "hidden_gems": [dict(row) for row in gems],
        "wildcards": [dict(row) for row in wcards],
        "best_value": [dict(row) for row in value],
        "peak_season": [dict(row) for row in peak]
    }


if __name__ == '__main__':
    app.run(debug=True)