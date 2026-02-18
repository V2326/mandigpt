import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="MandiGPT Pro",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
    }
    .feature-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    .profit {
        color: green;
        font-weight: bold;
    }
    .loss {
        color: red;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA GENERATION
# ============================================

@st.cache_data
def load_sample_data():
    """Creates sample data for multiple crops and mandis"""
    
    crops = ['Tomato', 'Onion', 'Potato', 'Wheat', 'Rice', 'Soybean', 'Cotton', 'Sugarcane']
    mandis = {
        'Azadpur (Delhi)': {'state': 'Delhi', 'lat': 28.7, 'lon': 77.2, 'distance': 0},
        'Pune (Maharashtra)': {'state': 'Maharashtra', 'lat': 18.5, 'lon': 73.9, 'distance': 150},
        'Kolar (Karnataka)': {'state': 'Karnataka', 'lat': 13.1, 'lon': 78.1, 'distance': 250},
        'Koyambedu (Tamil Nadu)': {'state': 'Tamil Nadu', 'lat': 13.0, 'lon': 80.2, 'distance': 350},
        'Vashi (Mumbai)': {'state': 'Maharashtra', 'lat': 19.0, 'lon': 73.0, 'distance': 120},
        'Agra (UP)': {'state': 'Uttar Pradesh', 'lat': 27.2, 'lon': 78.0, 'distance': 280},
        'Indore (MP)': {'state': 'Madhya Pradesh', 'lat': 22.7, 'lon': 75.9, 'distance': 180},
        'Patna (Bihar)': {'state': 'Bihar', 'lat': 25.6, 'lon': 85.1, 'distance': 320},
    }
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # MSP Data (Minimum Support Price)
    msp_data = {
        'Wheat': 2275, 'Rice': 2200, 'Soybean': 4600, 
        'Cotton': 6620, 'Sugarcane': 340, 'Tomato': 0,
        'Onion': 0, 'Potato': 0
    }
    
    data = []
    weather_data = []
    np.random.seed(42)
    
    for crop in crops:
        # Base prices
        if crop == 'Tomato':
            base = 1200
        elif crop == 'Onion':
            base = 1500
        elif crop == 'Potato':
            base = 1000
        elif crop == 'Wheat':
            base = 2275
        elif crop == 'Rice':
            base = 2200
        elif crop == 'Soybean':
            base = 4600
        elif crop == 'Cotton':
            base = 6620
        else:  # Sugarcane
            base = 340
            
        for mandi, mandi_info in mandis.items():
            mandi_factor = np.random.uniform(0.9, 1.1)
            
            for i, date in enumerate(dates):
                # Price patterns
                trend = i * 1.5
                weekly = np.sin(i * 2 * np.pi / 7) * 40
                monthly = np.sin(i * 2 * np.pi / 30) * 60
                noise = np.random.normal(0, 25)
                
                price = (base + trend + weekly + monthly + noise) * mandi_factor
                
                data.append({
                    'date': date,
                    'crop': crop,
                    'mandi': mandi,
                    'state': mandi_info['state'],
                    'distance': mandi_info['distance'],
                    'modal_price': int(max(100, price)),
                    'arrivals': int(np.random.uniform(50, 300)),
                    'msp': msp_data.get(crop, 0)
                })
                
                # Weather data
                if i % 7 == 0:  # Weekly weather
                    weather_data.append({
                        'date': date,
                        'mandi': mandi,
                        'temperature': np.random.uniform(22, 38),
                        'rainfall': np.random.exponential(5),
                        'humidity': np.random.uniform(40, 90)
                    })
    
    return pd.DataFrame(data), pd.DataFrame(weather_data)

# ============================================
# PRICE PREDICTION
# ============================================

def predict_prices(df, crop, mandi, days=5):
    """Predict prices for next days"""
    filtered = df[(df['crop'] == crop) & (df['mandi'] == mandi)].copy()
    
    if len(filtered) < 7:
        predictions = []
        for i in range(1, days+1):
            predictions.append({
                'day': i,
                'price': 1200 + i * 10,
                'change': 1.0
            })
        return predictions
    
    filtered = filtered.sort_values('date')
    prices = filtered['modal_price'].values
    dates = filtered['date'].values
    
    # Calculate trend
    last_7 = np.mean(prices[-7:])
    last_30 = np.mean(prices[-30:]) if len(prices) >= 30 else last_7
    trend = last_7 / last_30 if last_30 > 0 else 1.0
    
    predictions = []
    for i in range(1, days+1):
        predicted = last_7 * trend + (i * 8) + np.random.normal(0, 15)
        predictions.append({
            'day': i,
            'price': int(predicted),
            'change': trend
        })
    
    return predictions

# ============================================
# WEATHER IMPACT
# ============================================

def get_weather_impact(weather_df, mandi, days=3):
    """Analyze weather impact on prices"""
    mandi_weather = weather_df[weather_df['mandi'] == mandi]
    if len(mandi_weather) == 0:
        return "Normal", 1.0
    
    latest = mandi_weather.iloc[-1]
    impact = 1.0
    reason = "Normal weather conditions"
    
    if latest['temperature'] > 35:
        impact = 1.15
        reason = f"High heat ({latest['temperature']:.1f}°C) may reduce supply, prices expected to rise"
    elif latest['temperature'] < 15:
        impact = 0.9
        reason = f"Cold weather ({latest['temperature']:.1f}°C) may reduce demand, prices may drop"
    
    if latest['rainfall'] > 20:
        impact *= 1.1
        reason += ". Heavy rain may damage crops, prices likely to increase"
    
    return reason, impact

# ============================================
# TRANSPORT CALCULATOR
# ============================================

def calculate_transport_profit(price, distance, crop_quantity=100):
    """Calculate profit after transport cost"""
    # Transport cost: ₹10 per km per ton
    transport_cost = distance * 10 * (crop_quantity / 100)
    total_value = price * crop_quantity
    profit = total_value - transport_cost
    return profit, transport_cost

# ============================================
# MAIN APP
# ============================================

def main():
    # Header
    st.markdown("<h1 class='main-header'>🌾 MandiGPT Pro</h1>", unsafe_allow_html=True)
    st.markdown("#### *Complete Market Intelligence for Indian Farmers*")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading market data..."):
        df, weather_df = load_sample_data()
    
    # Sidebar for navigation
    st.sidebar.image("https://via.placeholder.com/150x150.png?text=🌾+MandiGPT", width=150)
    st.sidebar.title("Navigation")
    option = st.sidebar.radio(
        "Choose Feature",
        ["📊 Price Prediction", 
         "📍 Best Mandi Finder",
         "🚚 Transport Calculator",
         "🌤️ Weather Impact",
         "📈 Price Trends",
         "💰 MSP Information",
         "🌱 Crop Comparison",
         "📱 SMS Alerts"]
    )
    
    # Main content area
    if option == "📊 Price Prediction":
        show_price_prediction(df)
    elif option == "📍 Best Mandi Finder":
        show_best_mandi(df)
    elif option == "🚚 Transport Calculator":
        show_transport_calculator(df)
    elif option == "🌤️ Weather Impact":
        show_weather_impact(df, weather_df)
    elif option == "📈 Price Trends":
        show_price_trends(df)
    elif option == "💰 MSP Information":
        show_msp_info(df)
    elif option == "🌱 Crop Comparison":
        show_crop_comparison(df)
    elif option == "📱 SMS Alerts":
        show_sms_alerts()

# ============================================
# FEATURE 1: PRICE PREDICTION (A)
# ============================================

def show_price_prediction(df):
    st.header("📊 AI Price Prediction")
    st.markdown("Get tomorrow's predicted prices for your crop")
    
    col1, col2 = st.columns(2)
    with col1:
        crop = st.selectbox("Select Crop", df['crop'].unique(), key="pred_crop")
    with col2:
        mandi = st.selectbox("Select Mandi", df[df['crop']==crop]['mandi'].unique(), key="pred_mandi")
    
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        # Get today's price
        today_data = df[(df['crop'] == crop) & (df['mandi'] == mandi)]
        if len(today_data) > 0:
            today_price = today_data.sort_values('date', ascending=False).iloc[0]['modal_price']
        else:
            today_price = 1000
        
        # Get predictions
        predictions = predict_prices(df, crop, mandi, days=5)
        tomorrow_price = predictions[0]['price']
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Today's Price", f"₹{today_price:,.0f}/qtl")
        with col2:
            change = ((tomorrow_price - today_price) / today_price) * 100
            st.metric("Tomorrow's Price", f"₹{tomorrow_price:,.0f}/qtl", f"{change:.1f}%")
        with col3:
            if change > 2:
                st.success("✅ RECOMMENDATION: WAIT")
                st.caption(f"Prices expected to rise by {change:.1f}%")
            elif change < -2:
                st.error("⚠️ RECOMMENDATION: SELL NOW")
                st.caption(f"Prices expected to drop by {abs(change):.1f}%")
            else:
                st.info("ℹ️ RECOMMENDATION: HOLD")
                st.caption("Market is stable")
        
        # Show 5-day forecast
        st.subheader("📅 5-Day Price Forecast")
        cols = st.columns(5)
        for i, pred in enumerate(predictions):
            with cols[i]:
                st.metric(f"Day {pred['day']}", f"₹{pred['price']:,.0f}")

# ============================================
# FEATURE 2: BEST MANDI FINDER (B)
# ============================================

def show_best_mandi(df):
    st.header("📍 Best Mandi Finder")
    st.markdown("Find which mandi gives the highest price for your crop")
    
    crop = st.selectbox("Select your crop", df['crop'].unique(), key="best_crop")
    farmer_location = st.text_input("Your village/district", "Enter your location")
    
    if st.button("Find Best Mandi", type="primary"):
        # Get latest prices for all mandis for this crop
        latest_prices = []
        for mandi in df[df['crop']==crop]['mandi'].unique():
            mandi_data = df[(df['crop']==crop) & (df['mandi']==mandi)]
            if len(mandi_data) > 0:
                latest = mandi_data.sort_values('date', ascending=False).iloc[0]
                latest_prices.append({
                    'mandi': mandi,
                    'price': latest['modal_price'],
                    'distance': latest['distance']
                })
        
        # Sort by price
        latest_prices.sort(key=lambda x: x['price'], reverse=True)
        
        st.subheader("🏆 Top Mandis by Price")
        for i, m in enumerate(latest_prices[:5]):
            profit, transport = calculate_transport_profit(m['price'], m['distance'])
            col1, col2, col3 = st.columns([3,1,2])
            with col1:
                st.write(f"**{i+1}. {m['mandi']}**")
            with col2:
                st.write(f"₹{m['price']:,.0f}/qtl")
            with col3:
                if i == 0:
                    st.markdown(f"<span class='profit'>✨ BEST PRICE</span>", unsafe_allow_html=True)
        
        # Recommendation
        best = latest_prices[0]
        st.success(f"✅ **Recommendation:** Sell at {best['mandi']} for ₹{best['price']:,.0f}/qtl")

# ============================================
# FEATURE 3: TRANSPORT CALCULATOR (C)
# ============================================

def show_transport_calculator(df):
    st.header("🚚 Transport Cost Calculator")
    st.markdown("Calculate your profit after transport costs")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        crop = st.selectbox("Crop", df['crop'].unique(), key="trans_crop")
    with col2:
        mandi = st.selectbox("Destination Mandi", df[df['crop']==crop]['mandi'].unique(), key="trans_mandi")
    with col3:
        quantity = st.number_input("Quantity (quintals)", min_value=1, max_value=1000, value=100)
    
    # Get price
    price_data = df[(df['crop']==crop) & (df['mandi']==mandi)]
    if len(price_data) > 0:
        price = price_data.sort_values('date', ascending=False).iloc[0]['modal_price']
        distance = price_data.iloc[0]['distance']
        
        # Calculate
        total_value = price * quantity
        transport_cost = distance * 10 * quantity
        profit = total_value - transport_cost
        profit_per_qtl = profit / quantity
        
        # Display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"💰 Total Value: ₹{total_value:,.0f}")
        with col2:
            st.warning(f"🚚 Transport: ₹{transport_cost:,.0f}")
        with col3:
            if profit > total_value * 0.7:
                st.success(f"✅ Net Profit: ₹{profit:,.0f}")
            else:
                st.error(f"⚠️ Net Profit: ₹{profit:,.0f}")
        
        st.metric("Profit per quintal", f"₹{profit_per_qtl:,.0f}")
        
        if profit_per_qtl < price * 0.5:
            st.warning("⚠️ Transport cost is high! Consider closer mandi.")

# ============================================
# FEATURE 4: WEATHER IMPACT (D)
# ============================================

def show_weather_impact(df, weather_df):
    st.header("🌤️ Weather Impact on Prices")
    st.markdown("See how weather affects market prices")
    
    mandi = st.selectbox("Select Mandi", df['mandi'].unique(), key="weather_mandi")
    
    reason, impact = get_weather_impact(weather_df, mandi)
    
    st.info(f"📊 Weather Analysis: {reason}")
    
    # Show impact on prices
    crop = st.selectbox("Select your crop", df['crop'].unique(), key="weather_crop")
    
    base_price = df[(df['crop']==crop) & (df['mandi']==mandi)]['modal_price'].iloc[-1]
    adjusted_price = base_price * impact
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Normal Price", f"₹{base_price:,.0f}")
    with col2:
        change = (impact - 1) * 100
        st.metric("Weather Adjusted Price", f"₹{adjusted_price:,.0f}", f"{change:.1f}%")

# ============================================
# FEATURE 5: PRICE TRENDS (E)
# ============================================

def show_price_trends(df):
    st.header("📈 Price Trends Analysis")
    st.markdown("View historical price trends with interactive charts")
    
    col1, col2 = st.columns(2)
    with col1:
        crop = st.selectbox("Select Crop", df['crop'].unique(), key="trend_crop")
    with col2:
        mandi = st.selectbox("Select Mandi", df[df['crop']==crop]['mandi'].unique(), key="trend_mandi")
    
    # Filter data
    trend_data = df[(df['crop']==crop) & (df['mandi']==mandi)].sort_values('date')
    
    if len(trend_data) > 0:
        # Create chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend_data['date'],
            y=trend_data['modal_price'],
            mode='lines+markers',
            name='Price',
            line=dict(color='green', width=3)
        ))
        
        fig.update_layout(
            title=f"{crop} Price Trend - {mandi}",
            xaxis_title="Date",
            yaxis_title="Price (₹/quintal)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Highest Price", f"₹{trend_data['modal_price'].max():,.0f}")
        with col2:
            st.metric("Lowest Price", f"₹{trend_data['modal_price'].min():,.0f}")
        with col3:
            st.metric("Average Price", f"₹{trend_data['modal_price'].mean():,.0f}")

# ============================================
# FEATURE 6: MSP INFORMATION (G)
# ============================================

def show_msp_info(df):
    st.header("💰 MSP Information")
    st.markdown("Government Minimum Support Price for crops")
    
    # Get MSP data
    msp_data = df[['crop', 'msp']].drop_duplicates()
    msp_data = msp_data[msp_data['msp'] > 0]
    
    for _, row in msp_data.iterrows():
        crop = row['crop']
        msp = row['msp']
        
        # Get current market price
        current = df[df['crop']==crop]['modal_price'].iloc[-1]
        
        col1, col2, col3 = st.columns([2,1,2])
        with col1:
            st.write(f"**{crop}**")
        with col2:
            st.write(f"MSP: ₹{msp:,.0f}")
        with col3:
            diff = current - msp
            if diff > 0:
                st.markdown(f"<span class='profit'>Market: ₹{current:,.0f} (+{diff})</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='loss'>Market: ₹{current:,.0f} ({diff})</span>", unsafe_allow_html=True)
    
    st.info("ℹ️ If market price falls below MSP, government will buy at MSP")

# ============================================
# FEATURE 7: CROP COMPARISON (H)
# ============================================

def show_crop_comparison(df):
    st.header("🌱 Crop Comparison")
    st.markdown("Compare which crop gives best price today")
    
    mandi = st.selectbox("Select Mandi", df['mandi'].unique(), key="comp_mandi")
    
    # Get latest prices for all crops in this mandi
    crop_prices = []
    for crop in df['crop'].unique():
        crop_data = df[(df['crop']==crop) & (df['mandi']==mandi)]
        if len(crop_data) > 0:
            price = crop_data.sort_values('date', ascending=False).iloc[0]['modal_price']
            crop_prices.append({
                'crop': crop,
                'price': price
            })
    
    # Sort by price
    crop_prices.sort(key=lambda x: x['price'], reverse=True)
    
    st.subheader("🏆 Best Crops to Sell Today")
    for i, cp in enumerate(crop_prices):
        col1, col2, col3 = st.columns([2,1,3])
        with col1:
            st.write(f"**{i+1}. {cp['crop']}**")
        with col2:
            st.write(f"₹{cp['price']:,.0f}")
        with col3:
            if i == 0:
                st.markdown("<span class='profit'>✨ HIGHEST PRICE</span>", unsafe_allow_html=True)
            elif i == 1:
                st.markdown("<span class='profit'>👍 Good Choice</span>", unsafe_allow_html=True)
    
    # Recommendation
    best = crop_prices[0]
    st.success(f"✅ **Best Crop:** {best['crop']} at ₹{best['price']:,.0f}/qtl")

# ============================================
# FEATURE 8: SMS ALERTS (Demo)
# ============================================

def show_sms_alerts():
    st.header("📱 SMS Price Alerts")
    st.markdown("Get price alerts directly on your phone")
    
    with st.form("sms_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Your Name")
        with col2:
            phone = st.text_input("Phone Number", "+91")
        
        crop = st.selectbox("Crop to track", ['Tomato', 'Onion', 'Potato', 'Wheat', 'Rice'])
        mandi = st.selectbox("Mandi", ['Azadpur (Delhi)', 'Pune', 'Kolar', 'Vashi'])
        
        target_price = st.number_input("Alert me when price crosses (₹/qtl)", min_value=500, value=1500)
        
        submitted = st.form_submit_button("🔔 Set Alert", type="primary")
        
        if submitted:
            st.success(f"✅ Alert set! You'll get SMS at {phone} when {crop} price crosses ₹{target_price} at {mandi}")
            st.info("📱 Demo: In real app, you'd receive SMS like: 'Price alert: Tomato at Azadpur is now ₹1600/qtl'")

# ============================================
# FOOTER
# ============================================

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "🌾 **MandiGPT Pro**\n\n"
    "AI-powered market intelligence for Indian farmers\n\n"
    "Features:\n"
    "• Price Prediction\n"
    "• Best Mandi Finder\n"
    "• Transport Calculator\n"
    "• Weather Impact\n"
    "• Price Trends\n"
    "• MSP Info\n"
    "• Crop Comparison\n"
    "• SMS Alerts\n\n"
    "Built for Agrithon 2024"
)

if __name__ == "__main__":
    main()
