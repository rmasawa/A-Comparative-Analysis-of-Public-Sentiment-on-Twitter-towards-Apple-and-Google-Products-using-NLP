import joblib
from dash import Dash, html, dcc, Input, Output

# Load trained model and vectorizer
model = joblib.load('xgb_sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

sentiment_labels = ['Negative', 'Neutral', 'Positive'] 

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Group Sentiment Analysis"),
    html.Label("Enter text for sentiment prediction:"),
    dcc.Textarea(id='input_text', style={'width': '60%', 'height': 100}),
    html.Br(),
    html.Button('Predict', id='predict_btn', n_clicks=0),
    html.H2(id='prediction_result')
])

@app.callback(
    Output('prediction_result', 'children'),
    Input('predict_btn', 'n_clicks'),
    Input('input_text', 'value')
)
def predict_sentiment(n_clicks, input_text):
    if n_clicks > 0 and input_text and input_text.strip():
        X_input = vectorizer.transform([input_text])
        pred = model.predict(X_input)[0]
        label = sentiment_labels[pred] if pred < len(sentiment_labels) else str(pred)
        return f"Prediction: {label}"
    return ""

if __name__ == '__main__':
    app.run(debug=True)
