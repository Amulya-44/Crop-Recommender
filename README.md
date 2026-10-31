# 🌾 Crop Recommendation System

A machine-learning project that recommends the most suitable crop from soil measurements, with an interactive **Streamlit** web app.

**[Live demo](https://your-app-name.streamlit.app)** · *(replace with your deployed link)*

## Overview
Given a soil sample's **Nitrogen (N), Phosphorus (P), Potassium (K)** and **pH**, the model predicts which of **22 crops** (rice, maize, banana, coffee, apple, …) fits best, and shows the top-3 candidates with confidence scores.

## Results
| Algorithm | CV acc. | Validation acc. |
|---|---|---|
| **Gradient Boosting** (selected) | 77.9% | 77.3% |
| Random Forest | 80.5% | 75.9% |
| SVM | 76.9% | 75.0% |
| Extra Trees | 78.2% | 73.2% |
| KNN | 75.3% | 72.3% |
| Logistic Regression | 68.6% | 66.4% |

**Final test accuracy: ~77.5%** on a held-out 20% split (22 balanced classes, so random guessing is ~4.5%).

## Approach
- **Data:** 2,200 samples, 100 per crop, no missing values (`data/soil_measures.csv`).
- **Feature engineering:** N/P, K/P and N/K ratios plus high-potassium and high-phosphorus flags.
- **Split:** 70% train / 10% validation / 20% test, stratified.
- **Model selection:** 6 algorithms tuned with 3-fold `GridSearchCV`; best chosen on validation accuracy, test set used once.

## Project structure
```
├── app.py                 # Streamlit app
├── src/
│   ├── features.py        # feature engineering (shared by train + app)
│   └── train.py           # training / model comparison script
├── data/soil_measures.csv # dataset
├── models/                # saved model + metrics.json
├── notebooks/             # original exploratory analysis
├── requirements.txt
└── LICENSE
```

## Run locally
```bash
git clone https://github.com/Amulya-44/crop-recommender.git
cd crop-recommender
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
To retrain the model: `python -m src.train`

## Limitations
- Only four soil features are used. Climate variables (rainfall, temperature, humidity) would likely improve accuracy, especially for look-alike crops such as lentil, pigeonpeas and apple.
- The dataset is small and not tied to a specific region; treat predictions as guidance, not agronomic advice.

## Tech stack
Python · pandas · scikit-learn · Streamlit · Plotly

## License
MIT - see [LICENSE](LICENSE).
