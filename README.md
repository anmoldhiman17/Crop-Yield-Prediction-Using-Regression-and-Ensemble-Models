📌 Overview

This project predicts crop yield using advanced Machine Learning models.
It analyzes real agricultural & climatic parameters such as:

🌧 Rainfall

🌡 Average Temperature

🧪 Pesticide Usage

🌍 Geographical Area

🌾 Crop Type

📅 Year

The goal is to provide farmers, researchers, and policymakers with accurate, data-driven predictions that support smarter agricultural planning.

🎯 Project Highlights

✨ Data preprocessing + feature engineering
✨ Comparison of 3 ML models
✨ Visualization of trends & correlations
✨ Model performance metrics (MAE, RMSE, R²)
✨ Best model with 98.7% accuracy
✨ Fully deployed using Streamlit

🤖 Machine Learning Models Used
Model	R² Score	MAE	RMSE
Linear Regression	0.6448	31791.65	50757.59
Gradient Boosting	0.8754	19448.66	30052.15
Random Forest (Winner)	⭐ 0.9876	⭐ 3464.94	⭐ 9482.22

🔥 Random Forest delivered the best accuracy and is used for final prediction.

🧠 Workflow
Dataset → Cleaning → Feature Engineering → Model Training 
        → Evaluation → Export .pkl Model → Streamlit Deployment

📊 Features & Visualizations

The project includes insights such as:

Rainfall vs Yield

Temperature vs Yield

Pesticide usage trends

Crop distribution

Model comparison charts

Visual graphs make the data easier to understand for stakeholders.

🌐 Streamlit Web App

An interactive prediction interface where users can input:

Rainfall

Temperature

Pesticides

Year

Crop

Area

→ And instantly get the predicted yield.

🛠 Tech Stack
Category	Tools
Language	Python
ML	Scikit-Learn
Data Handling	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Deployment	Streamlit
Model Saving	Joblib
📁 Project Structure
📦 Crop-Yield-Prediction
│
├── 📄 app.py                  # Streamlit app
├── 📄 model.pkl               # Trained Random Forest model
├── 📄 requirements.txt        # Dependencies
├── 📄 README.md               # Documentation
│
└── 📂 dataset/                # Rainfall, Temperature, Yield, Pesticides etc.

🚀 How to Run Locally
1️⃣ Clone the repository
git clone https://github.com/your-username/Crop-Yield-Prediction.git
cd Crop-Yield-Prediction

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit App
streamlit run app.py

🌱 Future Enhancements

Integration of satellite imagery

Incorporating soil nutrient data

Real-time weather forecasting API

Deep learning models (LSTM, CNN)

Mobile application

🤝 Contributors

👤 Anmol Dhiman
👤 Tanish Sonker
👤 Lucky Sonker
👤 Nishant Chauhan

📚 References

FAO Crop Production Statistics

Research papers on ML in Agriculture

Random Forest Prediction Models

Gradient Boosting Applications

⭐ Support the Project

If you like this work, give the repo a ⭐ on GitHub — it motivates us to build more awesome projects!
