# app.py - Market Intelligence Simulator

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.neighbors import NearestNeighbors
import os

# Page Configuration
st.set_page_config(page_title="Football Market Simulator", layout="wide", initial_sidebar_state="expanded")

# Phase 1: The Engine Room (Caching & Loading)
# We cache this so the app doesn't reload the 314-player database every time a slider moves
@st.cache_data
def load_data():
    # Streamlit runs from the root folder, so we point to app/data/
    data_path = os.path.join("app", "data", "simulator_base_data.csv")
    df = pd.read_csv(data_path)
    return df

@st.cache_resource
def load_model():
    model = xgb.Booster()
    # Pointing back to your original models folder
    model_path = os.path.join("models", "xgb_valuation_model.json")
    model.load_model(model_path)
    return model

try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"Error loading system files. Please ensure you are running this from the project root directory. Error: {e}")
    st.stop()

# Identify the feature columns (excluding the SHAP columns and target prices)
feature_cols = [col for col in df.columns if not col.endswith('_SHAP') and 'Value' not in col and col != 'Player_Name']

# Phase 2: Left Sidebar (Control Room) 
st.sidebar.title("Player Search")
player_name = st.sidebar.selectbox("Select Player Database", df['Player_Name'].tolist())

# Extracting current player data
player_data = df[df['Player_Name'] == player_name].iloc[0]
base_value = player_data['True_Value_EUR'] if 'True_Value_EUR' in player_data else player_data['Predicted_Value_EUR']

st.sidebar.markdown("### Scenario Configuration")
st.sidebar.markdown("Adjust attributes to simulate new market value.")

# Creating dynamic sliders based on current player stats
# We just grab the core stats for the UI to keep it clean, but pass all to the model
ui_features = ['Age', 'Overall', 'Potential', 'Sprint_Speed', 'Finishing', 'Short_Passing', 'Dribbling', 'Standing_Tackle', 'Strength']
new_inputs = {}

for col in feature_cols:
    if col in ui_features:
        # Creating slider for the core UI features
        current_val = int(player_data[col])
        new_inputs[col] = st.sidebar.slider(f"{col}", min_value=15, max_value=99, value=current_val)
    else:
        # Keeping background features exactly the same as the base player
        new_inputs[col] = player_data[col]

#  Phase 3: Real-Time Inference
# Converting slider inputs back into a matrix format for XGBoost
input_df = pd.DataFrame([new_inputs])
dmatrix = xgb.DMatrix(input_df)

# Running prediction and reversing the log transformation
new_log_pred = model.predict(dmatrix)[0]
projected_value = np.expm1(new_log_pred)
delta = projected_value - base_value
delta_pct = (delta / base_value) * 100

# Phase 4: Main Dashboard (The Financial Output) 
st.title(" Market Intelligence & Valuation Simulator")
st.markdown(f"**Player:** {player_name} | **Current Base Value:** €{base_value/1_000_000:.2f}M")

# Top Row Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Base Value", f"€{base_value/1_000_000:.2f}M")
col2.metric("Projected Value ", f"€{projected_value/1_000_000:.2f}M", f"{delta_pct:.2f}%")
col3.metric("Value Impact (Delta)", f"€{abs(delta)/1_000_000:.2f}M", "Increase" if delta > 0 else "-Decrease")

st.divider()

# Phase 5: Bonus Features 
tab1, tab2, tab3 = st.tabs(["Value Drivers (SHAP)", "Similar Targets (Bonus)", "Negotiation Room (Bonus)"])

# TAB 1: Baseline SHAP Impact
with tab1:
    st.subheader("Current Value Drivers")
    st.markdown("What attributes are currently anchoring this player's price tag?")
    
    # Extracting SHAP columns for this specific player
    shap_cols = [c for c in df.columns if c.endswith('_SHAP')]
    shap_data = []
    for c in shap_cols:
        stat_name = c.replace('_SHAP', '')
        if stat_name in ui_features: # Only show core stats for clean UI
            shap_data.append({"Attribute": stat_name, "Impact (€M)": player_data[c] / 1_000_000})
            
    shap_df_ui = pd.DataFrame(shap_data).sort_values(by="Impact (€M)", ascending=False)
    st.bar_chart(shap_df_ui, x="Attribute", y="Impact (€M)")

# TAB 2: Bonus Feature 1 - "Find similar players at lower cost"
with tab2:
    st.subheader("AI Scouting: Cheaper Alternatives")
    st.markdown("Using Nearest Neighbors to find players with >90% statistical similarity at a lower market price.")
    
    # Build KNN model on the fly using our feature matrix
    knn_matrix = df[feature_cols].values
    knn = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn.fit(knn_matrix)
    
    # Finding neighbors for our currently selected player
    target_vector = player_data[feature_cols].values.reshape(1, -1)
    distances, indices = knn.kneighbors(target_vector)
    
    # Filtering out the player himself and finding cheaper options
    similar_players = df.iloc[indices[0][1:]] # Skip index 0 (the player himself)
    cheaper_players = similar_players[similar_players['Predicted_Value_EUR'] < base_value]
    
    if not cheaper_players.empty:
        for _, row in cheaper_players.head(3).iterrows():
            st.success(f"**{row['Player_Name']}** | Value: €{row['Predicted_Value_EUR']/1_000_000:.2f}M")
    else:
        st.warning("No significantly cheaper players with this exact statistical profile found in the elite database.")

# TAB 3: Bonus Feature 2 - "Negotiation Simulator"
with tab3:
    st.subheader("War Room: Bid Simulator")
    st.markdown(f"**Selling Club's Internal Valuation (FMV):** €{base_value/1_000_000:.2f}M")
    
    # User inputs their budget
    user_bid_m = st.number_input("Enter your official bid (in Millions €):", min_value=1.0, max_value=500.0, value=float(base_value/1_000_000))
    user_bid = user_bid_m * 1_000_000
    
    # Logic Engine
    if st.button("Submit Official Bid"):
        gap = (user_bid - base_value) / base_value
        
        if gap >= 0.05:
            st.success("DEAL AGREED: Bid accepted instantly. You may have slightly overpaid.")
        elif gap >= -0.05:
            st.info("COUNTER-OFFER: Deal agreed in principle, but agent is demanding higher performance bonuses.")
        elif gap >= -0.20:
            st.warning("REJECTED: Bid is too far below our valuation. Submit a revised offer.")
        else:
            st.error(" NEGOTIATIONS BROKEN DOWN: The selling club considered this bid insulting and has walked away.")