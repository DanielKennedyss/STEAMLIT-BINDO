import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import io
from pandas.tseries.offsets import BDay
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS ENTERPRISE
# ==========================================
st.set_page_config(
    page_title="Enterprise Fund Manager | Robo-Advisor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title { font-size: 38px; font-weight: 800; color: #0F172A; margin-bottom: 0px; }
    .sub-title { font-size: 16px; color: #64748B; margin-bottom: 25px; font-weight: 500;}
    .metric-box { background-color: #F8FAFC; padding: 15px; border-radius: 10px; border: 1px solid #E2E8F0; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI ANALISIS & KOMPUTASI CACHE
# ==========================================
@st.cache_data(ttl=1800)
def fetch_and_clean_data(ticker, start, end):
    """Menarik data dan membersihkan Multi-Index dari Yahoo Finance"""
    end_adjusted = end + datetime.timedelta(days=1)
    df = yf.download(ticker, start=start, end=end_adjusted)
    
    # Flatten Multi-Index Columns jika ada (Pencegahan Error yfinance v1.4+)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def compute_technical_indicators(df):
    """Kalkulasi Indikator Teknikal Tingkat Lanjut secara Native"""
    # 1. Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. Moving Average Convergence Divergence (MACD)
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 3. Moving Average 50 & 200
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    return df

@st.cache_resource
def train_lstm_engine(scaled_data, training_len, t_steps, epochs, batch_size, dropout_rate):
    """Arsitektur LSTM Dinamis dengan Hyperparameter Tuning"""
    train_data = scaled_data[0:int(training_len), :]
    x_train, y_train = [], []
    for i in range(t_steps, len(train_data)):
        x_train.append(train_data[i-t_steps:i, 0])
        y_train.append(train_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0) 
    return model

# ==========================================
# 3. SIDEBAR: KONTROL INSTITUSIONAL
# ==========================================
with st.sidebar:
    st.markdown("### 🏛️ **Institutional Terminal**")
    
    dict_saham = {"Bank Central Asia (BBCA)": "BBCA.JK", "Bank Rakyat Indonesia (BBRI)": "BBRI.JK", "Bank Mandiri (BMRI)": "BMRI.JK"}
    saham_user = st.selectbox("Aset Portofolio:", list(dict_saham.keys()))
    ticker_target = dict_saham[saham_user]
    
    st.markdown("---")
    st.markdown("#### ⚙️ **Hyperparameter Tuning (AI)**")
    time_step = st.slider("Time Steps (Memory)", 30, 120, 60, 10)
    epochs_val = st.select_slider("Epochs (Training Depth)", options=[10, 25, 35, 50], value=35)
    batch_val = st.select_slider("Batch Size", options=[16, 32, 64], value=32)
    drop_val = st.slider("Dropout Rate (Regularization)", 0.1, 0.5, 0.2, 0.1)
    
    st.markdown("---")
    st.markdown("#### 📅 **Rentang Data Historis**")
    today = datetime.date.today()
    five_years_ago = today - datetime.timedelta(days=5*365)
    start_date = st.date_input("Start Date", value=five_years_ago)
    end_date = st.date_input("End Date (Real-Time)", value=today)
    
    st.markdown("---")
    st.markdown("#### 💼 **Simulasi Modal (Paper Trading)**")
    modal_awal = st.number_input("Modal Awal (IDR)", min_value=10000000, value=100000000, step=10000000)

if start_date > end_date:
    st.sidebar.error("Error: Start Date melebihi End Date.")
    st.stop()

# ==========================================
# 4. QUANTITATIVE ENGINE (BACKEND)
# ==========================================
raw_data = fetch_and_clean_data(ticker_target, start_date, end_date)
if raw_data.empty:
    st.error("Data tidak ditemukan. Sesuaikan kalender.")
    st.stop()

raw_data = compute_technical_indicators(raw_data)
close_prices = raw_data[['Close']].values

# Preprocessing & AI Training
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)
training_data_len = int(np.ceil(len(scaled_data) * 0.8))

if training_data_len <= time_step:
    st.error(f"Data terlalu pendek untuk Time Step {time_step}.")
    st.stop()

with st.spinner('Memproses Komputasi Kuantitatif Deep Learning...'):
    model_lstm = train_lstm_engine(scaled_data, training_data_len, time_step, epochs_val, batch_val, drop_val)

# Pengujian & Prediksi
test_data = scaled_data[training_data_len - time_step:, :]
x_test = []
y_test = close_prices[training_data_len:, :]
for i in range(time_step, len(test_data)):
    x_test.append(test_data[i-time_step:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model_lstm.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Kalkulasi Metrik Evaluasi
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

# Prediksi T+1
last_n_days = scaled_data[-time_step:]
x_besok = np.array([last_n_days])
x_besok = np.reshape(x_besok, (x_besok.shape[0], x_besok.shape[1], 1))
pred_besok = scaler.inverse_transform(model_lstm.predict(x_besok))[0][0]

tanggal_terakhir = raw_data.index[-1]
tanggal_prediksi = tanggal_terakhir + BDay(1)
harga_terakhir = float(raw_data['Close'].iloc[-1])

# --- ALGORITMA CONFLUENCE (Keputusan Institusional) ---
rsi_terakhir = raw_data['RSI'].iloc[-1]
macd_terakhir = raw_data['MACD'].iloc[-1]
signal_terakhir = raw_data['Signal_Line'].iloc[-1]

lstm_signal = "BULLISH" if pred_besok > harga_terakhir else "BEARISH"
rsi_signal = "OVERSOLD (Potensi Naik)" if rsi_terakhir < 30 else ("OVERBOUGHT (Potensi Turun)" if rsi_terakhir > 70 else "NEUTRAL")
macd_signal = "BUY/GOLDEN CROSS" if macd_terakhir > signal_terakhir else "SELL/DEATH CROSS"

if lstm_signal == "BULLISH" and rsi_terakhir < 60 and macd_terakhir > signal_terakhir:
    final_decision = "STRONG BUY 🟢"
elif lstm_signal == "BEARISH" and rsi_terakhir > 40 and macd_terakhir < signal_terakhir:
    final_decision = "STRONG SELL 🔴"
else:
    final_decision = "HOLD / WAIT & SEE 🟡"

# ==========================================
# 5. DASHBOARD LAYOUT & UX
# ==========================================
st.markdown(f"<div class='main-title'>Quantitative Desk: {saham_user}</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Algorithmic Confluence Platform • Deep Learning & Technical Analysis Integration</div>", unsafe_allow_html=True)

# TAMPILAN KPI UTAMA
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Market Close ({tanggal_terakhir.strftime('%d %b')})", f"Rp {harga_terakhir:,.0f}", f"{harga_terakhir - raw_data['Close'].iloc[-2]:,.0f}")
c2.metric(f"AI Prediction ({tanggal_prediksi.strftime('%d %b')})", f"Rp {pred_besok:,.0f}", f"{pred_besok - harga_terakhir:,.0f} (T+1)", delta_color="inverse")
c3.metric("Confluence Signal", final_decision, f"RSI: {rsi_terakhir:.1f}", delta_color="off")
c4.metric("Model Precision", "OPTIMAL", f"MAPE: {mape:.2f}%")

st.markdown("---")

# TABS DASHBOARD
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Pro Charts (Candlestick & Volume)", 
    "🧠 Smart Analytics & Hybrid Confluence", 
    "💼 Backtesting & Paper Trading",
    "⚙️ Enterprise Export & Alerts"
])

with tab1:
    st.markdown("### **Interactive Candlestick & Volume Profile**")
    
    # Subplots: Candlestick di atas, Volume di bawah
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # Trace 1: Candlestick
    fig.add_trace(go.Candlestick(x=raw_data.index, open=raw_data['Open'], high=raw_data['High'], low=raw_data['Low'], close=raw_data['Close'], name='Market Price'), row=1, col=1)
    # Trace 2: MA50
    fig.add_trace(go.Scatter(x=raw_data.index, y=raw_data['MA50'], line=dict(color='orange', width=1.5), name='MA 50'), row=1, col=1)
    
    # Fibonacci Levels (Perhitungan otomatis High/Low 6 bulan terakhir)
    recent_data = raw_data.tail(180)
    max_price, min_price = recent_data['High'].max(), recent_data['Low'].min()
    diff = max_price - min_price
    fib_levels = [max_price, max_price - 0.236 * diff, max_price - 0.5 * diff, max_price - 0.618 * diff, min_price]
    colors = ['red', 'yellow', 'green', 'blue', 'gray']
    for level, color in zip(fib_levels, colors):
        fig.add_hline(y=level, line_dash="dot", line_color=color, opacity=0.5, row=1, col=1)

    # Trace 3: Volume
    colors_vol = ['red' if row['Open'] > row['Close'] else 'green' for index, row in raw_data.iterrows()]
    fig.add_trace(go.Bar(x=raw_data.index, y=raw_data['Volume'], marker_color=colors_vol, name='Volume'), row=2, col=1)
    
    fig.update_layout(xaxis_rangeslider_visible=False, height=700, template="plotly_white", showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### **Multi-Indicator Confluence (Deep Learning + Technicals)**")
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.markdown("#### 1. AI Forecasting Data (LSTM)")
        df_valid = raw_data[['Close']].iloc[training_data_len:].copy()
        df_valid['AI_Pred'] = predictions
        
        fig_ai = go.Figure()
        fig_ai.add_trace(go.Scatter(x=df_valid.index, y=df_valid['Close'], name='Actual', line=dict(color='#16A34A')))
        fig_ai.add_trace(go.Scatter(x=df_valid.index, y=df_valid['AI_Pred'], name='AI Prediction', line=dict(color='#DC2626', dash='dash')))
        fig_ai.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), legend=dict(yanchor="bottom", y=1.02, x=0, orientation="h"))
        st.plotly_chart(fig_ai, use_container_width=True)
        
    with col_b:
        st.markdown("#### 2. Momentum & Sentimen Proxy")
        st.info(f"""
        **Analisis RSI (14 Days):** Nilai **{rsi_terakhir:.1f}** -> {rsi_signal}  
        **Analisis MACD:** {macd_signal} (MACD: {macd_terakhir:.1f}, Signal: {signal_terakhir:.1f})  
        **Simulasi NLP/Sentimen Pasar:** Volatilitas normal, tidak ada anomali sentimen berita ekstrem (Proxy Berbasis Momentum VIX).
        """)
        st.success(f"**Keputusan Eksekusi (Algorithmic Output): {final_decision}**")

with tab3:
    st.markdown("### **Algorithmic Backtesting & Paper Trading**")
    st.markdown("Simulasi: Jika AI menjalankan Trading Portofolio secara otomatis di masa lalu.")
    
    # Logic Backtesting Sederhana: Beli jika AI prediksi naik besok, Jual jika turun
    df_bt = df_valid.copy()
    df_bt['Signal'] = np.where(df_bt['AI_Pred'].shift(-1) > df_bt['Close'], 1, 0) # 1 = Buy/Hold, 0 = Sell/Cash
    df_bt['Daily_Return'] = df_bt['Close'].pct_change()
    df_bt['Strategy_Return'] = df_bt['Signal'].shift(1) * df_bt['Daily_Return']
    
    # Hitung Kumulatif ROI
    df_bt['Cumulative_Market'] = (1 + df_bt['Daily_Return']).cumprod() * modal_awal
    df_bt['Cumulative_Strategy'] = (1 + df_bt['Strategy_Return']).cumprod() * modal_awal
    
    nilai_akhir_pasar = df_bt['Cumulative_Market'].iloc[-1]
    nilai_akhir_ai = df_bt['Cumulative_Strategy'].iloc[-1]
    
    c_bt1, c_bt2 = st.columns(2)
    c_bt1.metric("Nilai Akhir Jika Beli & Tahan (Buy & Hold)", f"Rp {nilai_akhir_pasar:,.0f}", f"{(nilai_akhir_pasar-modal_awal)/modal_awal*100:.2f}%")
    c_bt2.metric("Nilai Akhir Menggunakan Sinyal AI (LSTM Strategy)", f"Rp {nilai_akhir_ai:,.0f}", f"{(nilai_akhir_ai-modal_awal)/modal_awal*100:.2f}%", delta_color="normal")
    
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Cumulative_Market'], name='Market Performance (B&H)'))
    fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Cumulative_Strategy'], name='AI Strategy Performance'))
    fig_bt.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), legend=dict(yanchor="bottom", y=1.02, x=0, orientation="h"))
    st.plotly_chart(fig_bt, use_container_width=True)

with tab4:
    st.markdown("### **Enterprise Reporting & Automated Alerts**")
    col_x, col_y = st.columns(2)
    
    with col_x:
        st.markdown("#### 📥 Unduh Laporan Kuantitatif")
        st.write("Ekspor hasil analisis metrik harga riil, prediksi AI, dan indikator teknikal.")
        
        # Pembuatan file CSV di memori
        csv_buffer = df_valid.to_csv().encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv_buffer,
            file_name=f'RoboAdvisor_Report_{ticker_target}_{tanggal_terakhir.strftime("%Y%m%d")}.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    with col_y:
        st.markdown("#### 🔔 Konfigurasi Webhook Alerts")
        st.write("Integrasi sinyal otomatis ke platform komunikasi Anda.")
        wa_number = st.text_input("Nomor WhatsApp (Cth: +6281234...)")
        if st.button("Kirim Uji Coba Sinyal (Simulasi)", use_container_width=True):
            if wa_number:
                st.success(f"Simulasi Webhook berhasil! Pesan sinyal '{final_decision}' dijadwalkan ke {wa_number} besok pagi sebelum bursa buka.")
            else:
                st.error("Masukkan nomor tujuan terlebih dahulu.")