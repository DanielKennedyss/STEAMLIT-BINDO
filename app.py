import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from pandas.tseries.offsets import BDay
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Robo-Advisor Perbankan Blue Chip",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title { font-size: 38px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { font-size: 16px; color: #4B5563; margin-bottom: 25px; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI CACHING COMPUTATION (ADVANCE)
# ==========================================
@st.cache_data(ttl=1800)  # Refresh cache setiap 30 menit untuk data real-time
def fetch_historical_data(ticker, start, end):
    """Menarik data historis dinamis dari Yahoo Finance API"""
    # Menambahkan 1 hari pada end_date agar data hari ini ikut terambil penuh
    end_adjusted = end + datetime.timedelta(days=1)
    data = yf.download(ticker, start=start, end=end_adjusted)
    return data

@st.cache_resource
def train_lstm_engine(scaled_data, training_len, t_steps):
    """Melatih arsitektur LSTM dengan Time Steps Dinamis"""
    train_data = scaled_data[0:int(training_len), :]
    x_train, y_train = [], []
    
    for i in range(t_steps, len(train_data)):
        x_train.append(train_data[i-t_steps:i, 0])
        y_train.append(train_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=35, verbose=0) 
    return model

# ==========================================
# 3. SIDEBAR KONTROL & PARAMETER DINAMIS
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2621/2621430.png", width=80)
    st.markdown("### **Panel Kontrol Analisis**")
    
    dict_saham = {
        "Bank Central Asia (BBCA)": "BBCA.JK",
        "Bank Rakyat Indonesia (BBRI)": "BBRI.JK",
        "Bank Mandiri (BMRI)": "BMRI.JK"
    }
    saham_user = st.selectbox("Pilih Emiten Perbankan:", list(dict_saham.keys()))
    ticker_target = dict_saham[saham_user]
    
    st.markdown("---")
    st.markdown("#### **Pengaturan Waktu (Real-Time)**")
    
    # Default rentang waktu 5 tahun ke belakang hingga hari ini
    today = datetime.date.today()
    five_years_ago = today - datetime.timedelta(days=5*365)
    
    start_date = st.date_input("Tanggal Mulai Observasi", value=five_years_ago)
    end_date = st.date_input("Tanggal Akhir (Hingga Hari Ini)", value=today)
    
    st.markdown("---")
    st.markdown("#### **Kalibrasi Hyperparameter AI**")
    time_step = st.slider("Memori Jangka Pendek LSTM (Hari)", min_value=30, max_value=120, value=60, step=10, 
                          help="Menentukan berapa banyak hari ke belakang yang digunakan AI untuk memprediksi hari esok.")
    
    st.markdown("---")
    st.caption("AFL 3 Proyek Sistem Informasi Bisnis © 2026")

# Pastikan tanggal logika tidak terbalik
if start_date > end_date:
    st.sidebar.error("Error: Tanggal Mulai tidak boleh melewati Tanggal Akhir.")
    st.stop()

# ==========================================
# 4. ENGINE PEMROSESAN DATA (CORE BACKEND)
# ==========================================
raw_data = fetch_historical_data(ticker_target, start_date, end_date)

if raw_data.empty:
    st.error("Data tidak ditemukan untuk rentang waktu tersebut. Silakan sesuaikan tanggal.")
    st.stop()

close_prices = raw_data[['Close']].values

# Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)
training_data_len = int(np.ceil(len(scaled_data) * 0.8))

# Mencegah error jika data terlalu sedikit untuk time_step yang dipilih
if training_data_len <= time_step:
    st.error(f"Rentang waktu terlalu pendek untuk Time Step {time_step} hari. Perlebar rentang tanggal observasi.")
    st.stop()

# Eksekusi Pembelajaran Mesin
model_lstm = train_lstm_engine(scaled_data, training_data_len, time_step)

# Penyiapan Data Uji
test_data = scaled_data[training_data_len - time_step:, :]
x_test = []
y_test = close_prices[training_data_len:, :]
for i in range(time_step, len(test_data)):
    x_test.append(test_data[i-time_step:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Proses Prediksi dan Inversi
predictions = model_lstm.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Kalkulasi Metrik Evaluasi Akhir
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

# Prediksi untuk Hari Esok (T+1 Real-Time)
last_n_days = scaled_data[-time_step:]
x_besok = np.array([last_n_days])
x_besok = np.reshape(x_besok, (x_besok.shape[0], x_besok.shape[1], 1))
pred_besok_scaled = model_lstm.predict(x_besok)
pred_besok = scaler.inverse_transform(pred_besok_scaled)[0][0]

# Kalkulasi Tanggal Prediksi (Mengabaikan Weekend)
tanggal_terakhir = raw_data.index[-1]
tanggal_prediksi = tanggal_terakhir + BDay(1)

# ==========================================
# 5. STRUKTUR LAYOUT DASHBOARD UTAMA
# ==========================================
st.markdown(f"<div class='main-title'>Dashboard Analisis Finansial {saham_user}</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Sistem Informasi Manajemen Portofolio Berbasis Algoritma Rekuren Jaringan Saraf Tiruan (LSTM)</div>", unsafe_allow_html=True)

# ROW 1: TAMPILAN KPI METRIK UTAMA
col1, col2, col3, col4 = st.columns(4)
with col1:
    harga_terakhir = float(raw_data['Close'].values.flatten()[-1])
    harga_sebelumnya = float(raw_data['Close'].values.flatten()[-2])
    perubahan = harga_terakhir - harga_sebelumnya
    st.metric(label=f"Penutupan ({tanggal_terakhir.strftime('%d %b %Y')})", value=f"Rp {harga_terakhir:,.2f}", delta=f"{perubahan:,.2f}")

with col2:
    st.metric(label=f"Estimasi T+1 ({tanggal_prediksi.strftime('%d %b %Y')})", value=f"Rp {pred_besok:,.2f}", delta=f"{pred_besok - harga_terakhir:,.2f} (Prediksi)", delta_color="inverse")

with col3:
    st.metric(label="Rata-rata Tingkat Error (MAPE)", value=f"{mape:.2f}%", help="Persentase deviasi rata-rata model AI dari harga riil pasar.")

with col4:
    st.metric(label="Status Model Komputasi", value="OPTIMAL", delta=f"Memori: {time_step} Hari")

st.markdown("---")

# ROW 2: STRUKTUR TAB HALAMAN INTERAKTIF
tab1, tab2, tab3 = st.tabs(["📊 Grafik Interaktif Peramalan", "📋 Data Transaksi Real-Time", "🔬 Analisis Implikasi Bisnis"])

with tab1:
    st.markdown("### **Visualisasi Tren Pergerakan Saham Aktual vs Hasil Estimasi LSTM**")
    
    df_train = raw_data[:training_data_len].copy()
    df_valid = raw_data[training_data_len:].copy()
    df_valid['Predictions'] = predictions
    
    # Menambahkan fitur Moving Average (MA) untuk visualisasi yang lebih advance
    df_valid['MA50'] = df_valid['Close'].rolling(window=50, min_periods=1).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train.index, y=df_train['Close'].values.flatten(), name='Data Latih (Aktual)', line=dict(color='#2563EB')))
    fig.add_trace(go.Scatter(x=df_valid.index, y=df_valid['Close'].values.flatten(), name='Data Uji (Aktual Riil)', line=dict(color='#16A34A', width=2)))
    fig.add_trace(go.Scatter(x=df_valid.index, y=df_valid['Predictions'].values.flatten(), name='Estimasi Model AI (LSTM)', line=dict(color='#DC2626', dash='dash', width=2)))
    fig.add_trace(go.Scatter(x=df_valid.index, y=df_valid['MA50'].values.flatten(), name='Indikator MA50', line=dict(color='#F59E0B', width=1.5, dash='dot')))
    
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Periode Transaksi (Waktu)",
        yaxis_title="Nilai Penutupan Saham (IDR)",
        hovermode="x unified",
        legend=dict(orientation="h", ylink=1, y=1.02, x=0, xanchor="left"),
        margin=dict(l=20, r=20, t=30, b=20),
        height=550
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### **Arsip Data Historis Terintegrasi**")
    col_t1, col_t2 = st.columns([2, 1])
    
    with col_t1:
        st.markdown(f"#### **Matriks OHLCV 10 Hari Terakhir (Hingga {today.strftime('%d %b %Y')})**")
        st.dataframe(raw_data.tail(10).style.format("{:,.2f}"), use_container_width=True)
    
    with col_t2:
        st.markdown("#### **Parameter Statistik Kesalahan Model**")
        st.json({
            "Metrik Evaluasi": "Nilai Kuantitatif",
            "Root Mean Squared Error (RMSE)": f"{rmse:.2f}",
            "Mean Absolute Error (MAE)": f"{mae:.2f}",
            "Mean Absolute Percentage Error (MAPE)": f"{mape:.2f}%"
        })

with tab3:
    st.markdown("### **Rekomendasi Strategis & Implikasi Pengambilan Keputusan**")
    st.info(f"""
    **Analisis Teknis Sistem Informasi:** Berdasarkan pengujian komputasi pada rentang waktu yang dipilih, model LSTM mendeteksi nilai deviasi mutlak harian (MAE) rata-rata sebesar **Rp {mae:.2f}** dari harga aktual. 
    Dengan tingkat presisi keandalan mencapai **{100-mape:.2f}%**, platform merekomendasikan batas toleransi margin pengaman portofolio Anda di kisaran nominal tersebut.
    """)
    
    st.markdown(f"""
    #### **Sinyal Keputusan T+1 ({tanggal_prediksi.strftime('%d %b %Y')}):**
    * Harga Penutupan Terakhir: **Rp {harga_terakhir:,.2f}**
    * Estimasi AI Besok: **Rp {pred_besok:,.2f}**
    * Tren Jangka Pendek: **{'Meningkat 📈' if pred_besok > harga_terakhir else 'Menurun 📉'}**
    
    #### **Panduan Manajemen Risiko:**
    1. **Eksekusi Strategi Entry Point:** Investor ritel disarankan memanfaatkan hasil kalkulasi arah tren prediksi harian untuk menghindari bias impulsif (*FOMO*).
    2. **Mitigasi Kerugian (Cut-Loss):** Batas MAE sistem dapat dijadikan sebagai jangkar kuantitatif otomatis untuk menentukan penempatan sabuk pengaman (*stop-loss*) secara algoritmik di pasar sekunder.
    """)
