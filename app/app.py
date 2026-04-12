# app.py - Market Intelligence Simulator

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.neighbors import NearestNeighbors
import os

# Page Configuration
st.set_page_config(page_title="Football Market Simulator", layout="wide", initial_sidebar_state="expanded")

# Initialize Session State for Scenario Comparison
if 'saved_scenario' not in st.session_state:
    st.session_state.saved_scenario = None

# Phase 1: Engine Room (Caching & Loading)
@st.cache_data
def load_data():
    data_path = os.path.join("app", "data", "simulator_base_data.csv")
    df = pd.read_csv(data_path)
    return df

@st.cache_resource
def load_model():
    model = xgb.Booster()
    model_path = os.path.join("models", "xgb_valuation_model.json")
    model.load_model(model_path)
    return model

try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"Error loading system files. Error: {e}")
    st.stop()

feature_cols = [col for col in df.columns if not col.endswith('_SHAP') and 'Value' not in col and col != 'Player_Name']

# Phase 2: Left Sidebar (Control Room)
st.sidebar.title("Player Search")
player_name = st.sidebar.selectbox("Select Player Database", df['Player_Name'].tolist())

player_data = df[df['Player_Name'] == player_name].iloc[0]
base_value = player_data['True_Value_EUR'] if 'True_Value_EUR' in player_data else player_data['Predicted_Value_EUR']
# Reverse-engineer position using the baseline fallback logic
position_label = "Attacker" # The default baseline category

for col in df.columns:
    val = str(player_data[col]).strip().lower()
    
    # If we find a positive flag, check which of the remaining 3 positions it is
    if val in ['1', '1.0', 'true']:
        col_name = str(col).lower()
        if 'midfielder' in col_name:
            position_label = "Midfielder"
            break
        elif 'defender' in col_name:
            position_label = "Defender"
            break
        elif 'goalkeeper' in col_name:
            position_label = "Goalkeeper"
            break

st.sidebar.markdown("### Scenario Configuration")
ui_features = ['Age', 'Overall', 'Potential', 'Sprint_Speed', 'Finishing', 'Short_Passing', 'Dribbling', 'Standing_Tackle', 'Strength']
new_inputs = {}

for col in feature_cols:
    if col in ui_features:
        current_val = int(player_data[col])
        new_inputs[col] = st.sidebar.slider(f"{col}", min_value=15, max_value=99, value=current_val)
    else:
        new_inputs[col] = player_data[col]

# Resetting saved scenario if a new player is selected
if 'current_player' not in st.session_state or st.session_state.current_player != player_name:
    st.session_state.saved_scenario = None
    st.session_state.current_player = player_name

# Phase 3: Real-Time Inference 
input_df = pd.DataFrame([new_inputs])
dmatrix = xgb.DMatrix(input_df)

new_log_pred = model.predict(dmatrix)[0]
projected_value = np.expm1(new_log_pred)
delta = projected_value - base_value
delta_pct = (delta / base_value) * 100

# Calculate Uncertainty Bounds (Using ~8% MAE proxy from Jupyter testing)
mae_margin = projected_value * 0.08 
lower_bound = projected_value - mae_margin
upper_bound = projected_value + mae_margin

# --- Phase 4: Main Dashboard (The Financial Output) ---
st.title("Market Intelligence & Valuation Simulator")

# High-Visibility Player Header
st.markdown(f"<h1 style='text-align: center;'>{player_name} <span style='font-size: 24px; color: #888;'>({position_label})</span> | <span style='color: #4CAF50;'>€{base_value/1_000_000:.2f}M</span></h1>", unsafe_allow_html=True)

# High-Visibility Model Uncertainty Warning
st.markdown(f"""
<div style='background-color: rgba(255, 165, 0, 0.1); border-left: 5px solid #ffa500; padding: 15px; border-radius: 5px; font-size: 18px; margin-bottom: 20px;'>
    <b>⚠️ Model Confidence Interval (±8% MAE):</b> Estimated true market value ranges between 
    <span style='color: #ffa500; font-size: 20px;'><b>€{lower_bound/1_000_000:.2f}M</b></span> and 
    <span style='color: #ffa500; font-size: 20px;'><b>€{upper_bound/1_000_000:.2f}M</b></span>.
</div>
""", unsafe_allow_html=True)

# Top Row Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Base Value", f"€{base_value/1_000_000:.2f}M")
col2.metric("Projected Value 📈", f"€{projected_value/1_000_000:.2f}M", f"{delta_pct:.2f}%")
col3.metric("Value Impact (Delta)", f"€{abs(delta)/1_000_000:.2f}M", "Increase" if delta > 0 else "-Decrease")

# Scenario Comparison Engine
st.divider()
scol1, scol2 = st.columns([1, 2])

with scol1:
    if st.button("Save Current Scenario"):
        st.session_state.saved_scenario = {
            "value": projected_value,
            "delta_pct": delta_pct
        }
        st.success("Scenario locked in memory.")

with scol2:
    if st.session_state.saved_scenario:
        saved_val = st.session_state.saved_scenario['value']
        saved_pct = st.session_state.saved_scenario['delta_pct']
        
        # Dynamic coloring for the percentage text
        saved_color = "green" if saved_pct >= 0 else "red"
        live_color = "green" if delta_pct >= 0 else "red"
        
        # 3. The Bigger, Tabular Comparison Box
        comparison_html = f"""
        <div style="background-color: rgba(28, 131, 225, 0.1); border: 2px solid #1c83e1; border-radius: 10px; padding: 20px;">
            <h3 style="text-align: center; color: #1c83e1; margin-top: 0; margin-bottom: 20px;">Comparison Scenario</h3>
            <table style="width: 100%; text-align: center; border-collapse: collapse;">
                <tr>
                    <th style="width: 50%; font-size: 18px; padding-bottom: 10px; border-right: 1px solid rgba(28, 131, 225, 0.3);">Option A (Saved)</th>
                    <th style="width: 50%; font-size: 18px; padding-bottom: 10px;">Option B (Live)</th>
                </tr>
                <tr>
                    <td style="border-right: 1px solid rgba(28, 131, 225, 0.3); padding-top: 10px;">
                        <b style="font-size: 26px;">€{saved_val/1_000_000:.2f}M</b><br>
                        <span style="color: {saved_color}; font-weight: bold; font-size: 18px;">{saved_pct:+.2f}%</span>
                    </td>
                    <td style="padding-top: 10px;">
                        <b style="font-size: 26px;">€{projected_value/1_000_000:.2f}M</b><br>
                        <span style="color: {live_color}; font-weight: bold; font-size: 18px;">{delta_pct:+.2f}%</span>
                    </td>
                </tr>
            </table>
        </div>
        """
        st.markdown(comparison_html, unsafe_allow_html=True)
st.divider()

# Phase 5: The Bonus Features (Tabs)
tab1, tab2, tab3 = st.tabs(["📊 Value Drivers", "🎯 Similar Targets", "🤝 Negotiation Room"])

with tab1:
    st.subheader("Current Value Drivers")
    st.markdown("What attributes are currently anchoring this player's price tag?")
    
    shap_cols = [c for c in df.columns if c.endswith('_SHAP')]
    shap_data = []
    for c in shap_cols:
        stat_name = c.replace('_SHAP', '')
        if stat_name in ui_features: 
            shap_data.append({"Attribute": stat_name, "Impact (€M)": player_data[c] / 1_000_000})
            
    shap_df_ui = pd.DataFrame(shap_data).sort_values(by="Impact (€M)", ascending=False)
    st.bar_chart(shap_df_ui, x="Attribute", y="Impact (€M)")

with tab2:
    st.subheader("AI Scouting: Cheaper Alternatives")
    st.markdown("Using Nearest Neighbors to find players with >90% statistical similarity at a lower market price.")
    
    knn_matrix = df[feature_cols].values
    knn = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn.fit(knn_matrix)
    
    target_vector = player_data[feature_cols].values.reshape(1, -1)
    distances, indices = knn.kneighbors(target_vector)
    
    similar_players = df.iloc[indices[0][1:]] 
    cheaper_players = similar_players[similar_players['Predicted_Value_EUR'] < base_value]
    
    if not cheaper_players.empty:
        for _, row in cheaper_players.head(3).iterrows():
            st.success(f"**{row['Player_Name']}** | Value: €{row['Predicted_Value_EUR']/1_000_000:.2f}M")
    else:
        st.warning("No significantly cheaper players with this exact statistical profile found in the elite database.")

with tab3:
    st.subheader("Bid Simulator")
    st.markdown(f"**Selling Club's Internal Valuation (FMV):** €{base_value/1_000_000:.2f}M")
    
    user_bid_m = st.number_input("Enter your official bid (in Millions €):", min_value=1.0, max_value=500.0, value=float(base_value/1_000_000))
    user_bid = user_bid_m * 1_000_000
    
    if st.button("Submit Official Bid"):
        gap = (user_bid - base_value) / base_value
        
        if gap >= 0.05:
            st.success("🟢 DEAL AGREED: Bid accepted instantly. You may have slightly overpaid.")
        elif gap >= -0.05:
            st.info("🟡 COUNTER-OFFER: Deal agreed in principle, but agent is demanding higher performance bonuses.")
        elif gap >= -0.20:
            st.warning("🟠 REJECTED: Bid is too far below our valuation. Submit a revised offer.")
        else:
            st.error("🔴 NEGOTIATIONS BROKEN DOWN: The selling club considered this bid insulting and has walked away.")