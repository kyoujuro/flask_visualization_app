from flask import Flask, redirect, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly
import json
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy import stats
import plotly.graph_objects as go
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        df = pd.read_csv(file)
        df_json = df.to_json(orient='split')
        columns = df.columns.tolist()
        return render_template('visualizations.html', df_json=df_json, columns=columns)

@app.route('/plot', methods=['POST'])
def plot():
    data = json.loads(request.form['data'])
    df = pd.DataFrame(data['data'], columns=data['columns'])
    plot_type = request.form['plot_type']
    x_col = request.form['x_col']
    y_col = request.form['y_col']

    if plot_type == 'line':
        fig = px.line(df, x=x_col, y=y_col, title=f'Waveform of {y_col}')
    elif plot_type == 'pie':
        fig = px.pie(df, names=x_col, values=y_col, title=f'Pie chart of {y_col}')
    elif plot_type == 'bar':
        fig = px.bar(df, x=x_col, y=y_col, title=f'Bar chart of {y_col}')
    else:
        return jsonify({'error': 'Invalid plot type'})

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/ml_results', methods=['POST'])
def ml_results():
    data = json.loads(request.form['data'])
    df = pd.DataFrame(data['data'], columns=data['columns'])
    x_cols = request.form.getlist('x_cols')

    if len(x_cols) == 0:
        return jsonify({'error': 'At least one feature column must be selected for ML'})

    X = df[x_cols].values
    y = df.iloc[:, 0].values  # Assuming the first column as target for simplicity
    model = KMeans(n_clusters=3)
    model.fit(X)

    fig = px.scatter_matrix(df, dimensions=x_cols, color=model.labels_, title='KMeans Clustering')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON
@app.route('/regression', methods=['POST'])
def regression():
    data = json.loads(request.form['data'])
    df = pd.DataFrame(data['data'], columns=data['columns'])
    x_cols = request.form.getlist('x_cols')
    y_col = request.form['y_col']

    if len(x_cols) == 0 or not y_col:
        return jsonify({'error': 'At least one X column and one Y column must be selected for regression'})

    X = df[x_cols].values
    y = df[y_col].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Add prediction to dataframe
    df['Predicted'] = y_pred

    # Create plot with actual vs predicted values
    fig = px.scatter_matrix(df, dimensions=x_cols, color=y_col, title=f'Regression of {y_col} on {", ".join(x_cols)}')
    fig.add_traces(px.line(x=df[x_cols[0]], y=y_pred, labels={'x': x_cols[0], 'y': 'Predicted'}).data)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON
@app.route('/timeseries', methods=['POST'])
def timeseries():
    data = json.loads(request.form['data'])
    df = pd.DataFrame(data['data'], columns=data['columns'])
    date_col = request.form['date_col']
    
    value_col = request.form['value_col']

    forecast_period = int(request.form['forecast_period'])

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    # SARIMAXモデルで予測
    model = SARIMAX(df[value_col], order=(1, 1, 1))
    results = model.fit()

    forecast = results.get_forecast(steps=forecast_period)
    forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_period, freq='D')
    forecast_mean = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()

    # プロット
    forecast = results.get_forecast(steps=forecast_period)
    forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_period, freq='D')
    forecast_mean = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()

    # Convert forecast index and confidence interval to Series
    forecast_index_series = pd.Series(forecast_index)
    lower_conf_int = pd.Series(forecast_conf_int['lower ' + value_col].values, index=forecast_index)
    upper_conf_int = pd.Series(forecast_conf_int['upper ' + value_col].values, index=forecast_index)

    # プロット
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df[value_col], mode='lines', name='Historical Data'))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast_mean, mode='lines', name='Forecast', line=dict(color='red')))
    fig.add_trace(go.Scatter(
        x=list(forecast_index) + list(forecast_index[::-1]),
        y=lower_conf_int + upper_conf_int[::-1],
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255, 0, 0, 0)'),
        name='95% Confidence Interval'
    ))

    fig.update_layout(title='Time Series Forecast', xaxis_title='Date', yaxis_title='Value')
   
    plot_data = fig.to_dict()
    for trace in plot_data['data']:
        if isinstance(trace['x'], np.ndarray):
            trace['x'] = trace['x'].tolist()
        if isinstance(trace['y'], np.ndarray):
            trace['y'] = trace['y'].tolist()

    return jsonify(plot_data)
@app.route('/ab_test', methods=['POST'])
def ab_test():
    data = json.loads(request.form['data'])
    df = pd.DataFrame(data['data'], columns=data['columns'])
    group_col = request.form['group_col']
    value_col = request.form['value_col']

    groups = df.groupby(group_col)[value_col].apply(list)
    if len(groups) != 2:
        return jsonify({'error': 'A/B test requires exactly two groups'})

    group_a, group_b = groups
    t_stat, p_value = stats.ttest_ind(group_a, group_b)

    result = {
        't_stat': t_stat,
        'p_value': p_value,
        'is_significant': p_value < 0.05
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
