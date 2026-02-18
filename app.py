import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_data():
    crops = ['Tomato', 'Onion', 'Potato']
    mandis = ['Azadpur (Delhi)', 'Pune (Maharashtra)', 'Kolar (Karnataka)', 'Koyambedu (Tamil Nadu)', 'Vashi (Mumbai)']
    
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
    data = []
    
    for crop in crops:
        base = 1200 if crop=='Tomato' else 1500 if crop=='Onion' else 1000
        for mandi in mandis:
            for i, date in enumerate(dates):
                price = base + i*2 + np.random.randint(-50, 50)
                data.append({'date': date, 'crop': crop, 'mandi': mandi, 'price': price})
    
    return pd.DataFrame(data)

st.set_page_config(page_title="MandiGPT", page_icon="🌾")
st.title("🌾 MandiGPT")
st.write("AI Market Advisor for Farmers")

df = load_data()

crop = st.selectbox("Select Crop", df['crop'].unique())
mandi = st.selectbox("Select Mandi", df['mandi'].unique())

if st.button("Get Prediction"):
    today = df[(df.crop==crop) & (df.mandi==mandi)].iloc[-1].price
    tomorrow = today + np.random.randint(-50, 100)
    
    st.metric("Today", f"₹{today}")
    st.metric("Tomorrow", f"₹{tomorrow}")
    
    if tomorrow > today:
        st.success("✅ WAIT - Prices going up")
    else:
        st.error("⚠️ SELL NOW - Prices going down")