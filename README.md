# What If Simulator: Market Intelligence & Valuation 

## OVERVIEW
This repository contains a full-stack, end-to-end Market Intelligence Simulator developed for the SoccerSolver Technical Challenge. Designed for Sporting Directors and recruitment teams, the platform empowers decision-makers to conduct real-time, data-driven "what-if" analyses on elite football players. By combining web scraping, fuzzy string matching, machine learning, and a highly responsive Streamlit frontend, this application evaluates how discrete changes in behavioral and physical attributes mathematically impact a player's true transfer market value.

## Repo Structure
```text
whatif_simulator/
│
├── app/
│   ├── app.py                      # Main Streamlit application script
│   └── data/
│       └── simulator_base_data.csv # Processed feature matrix for UI│
├── data/
│   ├── sofifa_cleaned_behavioral.csv   #Cleaned data from SoFifa               
│   ├── sofifa_raw_data_COMPLETE.csv    #Raw data from SoFifa
|   ├── transfermarkt_raw_COMPLETE.csv  #Transfermarkt data
|   ├── UNIFIED_ML_DATASET.csv
│
├── models/
|   ├── Position Stratified Evaluation Report.md  
│   └── xgb_valuation_model.json    # Trained XGBoost engine
│
├── .gitignore
├── LICENSE
├── main.ipynb                      # Jupyter Notebook (Data pipeline & ML Training)
├── requirements.txt                # Pinned Python dependencies
└── README.md                       # Project documentation

```

## Planning and Roadmap

The development of this MVP was executed in five distinct phases:

1. **Data Acquisition**: Bypassing bot-protection to scrape real-world performance metrics and financial valuations.
2. **Data Engineering**: Cleaning, normalizing, and fuzzy-matching disparate datasets into a unified architecture.
3. **Machine Learning**: Training and hyperparameter-tuning a predictive valuation engine.
4. **UI/UX Development**: Building a zero-latency, interactive dashboard for non-technical users.
5. **Bonus Implementation**: Developing advanced heuristics for similar player targeting and contract negotiations.

## Key Features:
- **Real-Time What-If Engine**: Adjust a player's physical and technical attributes via sliders to instantly calculate their new projected market value.
- **Value Drivers (SHAP Integration)**: Visual breakdown of the exact financial leverage (in millions of Euros) each attribute contributes to a player's total valuation.
- **AI Scouting (Nearest Neighbors)**: Automatically identifies alternative players with >90% statistical similarity who possess a lower market valuation.
- **Negotiation Simulator**: A logic-based decision tree that tests user bids against the model's Fair Market Value (FMV) to simulate real-world negotiation outcomes.
- **Scenario Comparison**: Ability to lock in an initial "Option A" valuation and compare it side-by-side against a live "Option B" projection.
- **Model Uncertainty Bounds**: Explicitly displays the model's Confidence Interval (±8% MAE) to ensure transparent, realistic pricing expectations.

## Tech Stack

- **Frontend**: Streamlit
- **Backend & Machine Learning**: Python, Pandas, NumPy, Scikit-Learn, XGBoost
- **Explainability**: SHAP (Shapley Additive exPlanations)
- **Data Extraction & Engineering**: Playwright (Async), nest_asyncio, RapidFuzz

## System Architecture
The pipeline is divided into an offline training environment and a live production environment.

1. **Offline Training** (`main.ipynb`): `Playwright` asynchronously extracts data from SoFIFA and Transfermarkt. `RapidFuzz` merges the datasets despite spelling variations. The data is preprocessed, one-hot encoded, and fed into an `XGBoost` regressor. The trained model and final feature matrix are exported to disk.
2. **Live Production** (`app.py`): The Streamlit server reads the cached `xgb_valuation_model.json` and `simulator_base_data.csv`. User interactions with the UI sliders instantly rebuild a 1D prediction matrix, passing it to the XGBoost instance to calculate and render the new price in under 200ms.

## Installation and Setup
To ensure perfect reproducibility, all external dependencies have been pinned.

**Prerequisites**: Ensure you have Python 3.9+ installed.

1. **Clone the repository:**
    ```
    git clone https://github.com/Ashwynkumar16/whatif_simulator.git
    cd whatif_simulator
    ```
2. **Install Dependencies:**
    ```
    pip install -r requirements.txt
    ```
3. **Install browser binaries (Required for the Playwright web scraper):**
    (Note: Using python -m ensures the command executes cleanly regardless of system PATH configurations).
    ```
    python -m playwright install
    ```
4. **Run the Simulator:**
    ```
    python -m streamlit run app/app.py
    ```
5. **Stopping the Program**:
    Press Ctrl + C or Cmd + C in the terminal

The application will automatically launch in your default web browser at http://localhost:8501.

## Key Decisions

- **Choosing XGBoost**: Football market data is inherently non-linear (e.g., the value difference between 80 and 85 Pace is much higher than between 60 and 65). XGBoost effectively captures these complex, non-linear relationships better than standard linear regression.
- **Log-Transforming the Target**: Transfer values are heavily right-skewed (a few players are worth €150M+, most are under €20M). The target variable was log-transformed (`np.log1p`) during training to normalize the distribution and exponentially reversed (`np.expm1`) during inference.
- **Playwright over BeautifulSoup**: Because modern football databases utilize heavily JavaScript-rendered tables and bot-protection, standard HTML parsers fail. Playwright was implemented to simulate genuine browser interactions and successfully extract the data.

## Challenges and Limitations

- **By-passing antibot measures while Scraping** To bypass the anti-botscraping measures on both SoFifa and TransferMarkt, we introduced delays alongside `Headless=False` and this process consumed close to 10 hours
- **The String Matching Problem**: Merging Transfermarkt ("Vinicius Junior") with SoFIFA ("Vinícius Jr.") natively resulted in **massive** data loss. This was solved by implementing the RapidFuzz library to calculate the Levenshtein distance between names, ensuring high-fidelity data merges.
- **Data Scoping (Missing Features)**: During Phase 1, a deliberate scoping decision was made to exclude Injury History and Contract Expiry. Standard accessible datasets do not track historical medical records natively. Attempting to scrape deeply nested, historical medical tables for hundreds of players introduced an extreme risk of IP-blocking. Consequently, the MVP was constrained to serve strictly as a "Pure Attribute Valuation" baseline.

---

**Author**: Ashwyn Kumar
