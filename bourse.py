import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from sklearn.preprocessing import MinMaxScaler
import torch
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel

# Configuration de la page
st.set_page_config(page_title='Analyse des Marchés Financiers', layout='wide')

# Titre de l'application
st.title('Analyse des Marchés Financiers')

# Sélection de la valeur
st.sidebar.header('Paramètres de sélection')
ticker = st.sidebar.text_input('Entrez le symbole boursier (ex: AAPL, MSFT, GOOGL)', 'DSY.PA')

# Sélection des dates
start_date = st.sidebar.date_input('Date de début', value=pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('Date de fin', value=pd.to_datetime('2023-01-01'))

# Fonction pour télécharger les données
@st.cache_data
def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data.copy()  # Copier les données pour éviter la mutation

# Téléchargement des données
data = download_data(ticker, start_date, end_date)

# Calcul des moyennes mobiles
data['SMA50'] = data['Close'].rolling(window=50).mean()
data['SMA200'] = data['Close'].rolling(window=200).mean()

# Sélection des analyses à afficher
st.sidebar.header('Options d\'analyse')
show_sma50 = st.sidebar.checkbox('Afficher SMA50', value=True)
show_sma200 = st.sidebar.checkbox('Afficher SMA200', value=True)
show_signals = st.sidebar.checkbox('Afficher Signaux d\'achat/vente', value=True)
show_arima = st.sidebar.checkbox('Afficher Prédiction ARIMA', value=False)
show_lstm = st.sidebar.checkbox('Afficher Prédiction LSTM', value=False)
show_gru = st.sidebar.checkbox('Afficher Prédiction GRU', value=False)
show_transformer = st.sidebar.checkbox('Afficher Prédiction Transformer', value=False)

# Détection des signaux
data['Signal'] = 0
data['Signal'][50:] = np.where(data['SMA50'][50:] > data['SMA200'][50:], 1, 0)

# Fonction pour prédire avec ARIMA
def predict_arima(data):
    model = ARIMA(data['Close'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    return forecast

# Fonction pour préparer les données pour LSTM et GRU
def prepare_data(data, scaler):
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    train_data_len = int(np.ceil(len(scaled_data) * 0.95))
    train_data = scaled_data[0:int(train_data_len), :]

    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = scaled_data[train_data_len - 60:, :]
    x_test = []
    y_test = data['Close'][train_data_len:].values
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test

# Fonction pour prédire avec LSTM
def predict_lstm(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train, y_train, x_test, y_test = prepare_data(data, scaler)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Fonction pour prédire avec GRU
def predict_gru(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train, y_train, x_test, y_test = prepare_data(data, scaler)

    model = Sequential()
    model.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(GRU(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Fonction pour prédire avec Transformer
def predict_transformer(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    input_sequence_length = 60
    output_sequence_length = 30

    train_data_len = int(np.ceil(len(scaled_data) * 0.95))
    train_data = scaled_data[0:int(train_data_len)]

    x_train = []
    y_train = []
    for i in range(input_sequence_length, len(train_data) - output_sequence_length):
        x_train.append(train_data[i - input_sequence_length:i])
        y_train.append(train_data[i:i + output_sequence_length])

    x_train, y_train = np.array(x_train), np.array(y_train)

    config = TimeSeriesTransformerConfig(
        input_size=1,
        prediction_length=output_sequence_length,
        context_length=input_sequence_length,
        lags_sequence=[1],
    )

    model = TimeSeriesTransformerModel(config)
    model.train()

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(1):
        optimizer.zero_grad()
        output = model(x_train).last_hidden_state
        loss = loss_fn(output, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        x_test = scaled_data[train_data_len - input_sequence_length:]
        x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(0)
        predictions = model(x_test).last_hidden_state.squeeze().numpy()

    predictions = scaler.inverse_transform(predictions)
    return predictions

# Visualisation des données avec Plotly
fig = go.Figure()

# Ajout des prix de clôture
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Prix de clôture', line=dict(color='blue')))

# Ajout des moyennes mobiles en fonction des cases cochées
if show_sma50:
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], mode='lines', name='SMA50', line=dict(color='red')))
if show_sma200:
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA200'], mode='lines', name='SMA200', line=dict(color='green')))

# Ajout des signaux d'achat/vente
if show_signals:
    buy_signals = data[data['Signal'] == 1]
    sell_signals = data[data['Signal'] == 0]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', marker=dict(color='green', size=10), name='Acheter'))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', marker=dict(color='red', size=10), name='Vendre'))

# Ajout de la prédiction ARIMA
if show_arima:
    forecast = predict_arima(data)
    forecast_index = pd.date_range(start=data.index[-1], periods=len(forecast) + 1, closed='right')
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast, mode='lines', name='Prédiction ARIMA', line=dict(color='purple')))

# Ajout de la prédiction LSTM
if show_lstm:
    lstm_predictions = predict_lstm(data)
    lstm_index = data.index[-len(lstm_predictions):]
    fig.add_trace(go.Scatter(x=lstm_index, y=lstm_predictions, mode='lines', name='Prédiction LSTM', line=dict(color='orange')))

# Ajout de la prédiction GRU
if show_gru:
    gru_predictions = predict_gru(data)
    gru_index = data.index[-len(gru_predictions):]
    fig.add_trace(go.Scatter(x=gru_index, y=gru_predictions, mode='lines', name='Prédiction GRU', line=dict(color='brown')))

# Ajout de la prédiction Transformer
if show_transformer:
    transformer_predictions = predict_transformer(data)
    transformer_index = pd.date_range(start=data.index[-1], periods=len(transformer_predictions) + 1, closed='right')
    fig.add_trace(go.Scatter(x=transformer_index, y=transformer_predictions, mode='lines', name='Prédiction Transformer', line=dict(color='pink')))

# Mise à jour de la disposition du graphique
fig.update_layout(title=f'Analyse des Marchés Financiers pour {ticker}', xaxis_title='Date', yaxis_title='Prix', template='plotly_white')

# Affichage du graphique
st.plotly_chart(fig)

# Affichage des données
st.subheader(f'Données historiques pour {ticker}')
st.write(data.tail())
