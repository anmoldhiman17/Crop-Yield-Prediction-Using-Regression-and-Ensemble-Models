🌾 Crop Yield Prediction Using Regression & Ensemble Models
A Machine Learning approach to revolutionize agricultural productivity.
<p align="center">
  <img src="https://i.imgur.com/Gx1w5xP.png" width="100%">
</p>

🚀 Overview

Agriculture is highly sensitive to climatic conditions, making crop yield prediction a crucial challenge.
This project leverages Machine Learning to accurately predict crop yield using real-world features like:

🌧 Rainfall

🌡 Temperature

🧪 Pesticide Usage

🗺 Area (Country/Region)

🌾 Crop Type

📅 Year

Using advanced regression and ensemble algorithms, the project identifies patterns in agricultural data and predicts yield with up to 98.7% accuracy.

⭐ Key Features

✔ Cleaned, preprocessed & feature-engineered dataset
✔ ML Model Training with 3 algorithms
✔ Detailed model comparison
✔ Performance metrics (MAE, RMSE, R² Score)
✔ Visualizations for deeper insight
✔ Final model exported as .pkl
✔ Fully interactive Streamlit Web App for real-time prediction

🤖 Machine Learning Models Used
Model	R² Score	MAE	RMSE
Linear Regression	0.6448	31,791.65	50,757.59
Gradient Boosting Regressor	0.8754	19,448.66	30,052.15
⭐ Random Forest Regressor	⭐ 0.9876	⭐ 3,464.94	⭐ 9,482.22

➡️ Random Forest is selected as the final model (Best Performance).

📊 Project Workflow
1️⃣ Dataset Collection
2️⃣ Data Cleaning & Preprocessing
3️⃣ Feature Engineering
4️⃣ Model Training (Regression + Ensemble Models)
5️⃣ Model Evaluation (MAE, RMSE, R²)
6️⃣ Model Comparison
7️⃣ Saving Best Model (.pkl)
8️⃣ Streamlit Deployment

🧠 Tech Stack
Category	Tools Used
Language	Python
Libraries	Pandas, NumPy, Scikit-Learn
Visualization	Matplotlib, Seaborn
Deployment	Streamlit
Model Saving	Joblib
🌐 Streamlit Web Application

The project includes a clean & interactive UI made with Streamlit.
Users can input:

Year

Rainfall

Average Temperature

Pesticide Usage

Crop Type

Area

And instantly get the predicted crop yield.

Run the app locally:

streamlit run app.py

📁 Project Structure
📦 Crop-Yield-Prediction
│
├── app.py                    # Streamlit Web App
├── model.pkl                 # Trained Random Forest Model
├── requirements.txt          # Project Dependencies
├── README.md                 # Documentation
│
└── dataset/
     ├── yield.csv
     ├── rainfall.csv
     ├── pesticides.csv
     ├── temp.csv
     └── yield_df.csv

🔧 How to Run the Project Locally
1️⃣ Clone this repository
git clone https://github.com/your-username/Crop-Yield-Prediction.git
cd Crop-Yield-Prediction

2️⃣ Install the dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit app
streamlit run app.py

🌱 Future Enhancements

✨ Integration of satellite imagery (NDVI, crop health index)
✨ Incorporating soil properties (pH, nitrogen, phosphorus)
✨ Real-time weather API integration
✨ Deep learning: LSTM / CNN models for time-series prediction
✨ Mobile application version

👥 Contributors
Name	Role
Anmol Dhiman	Lead Developer & ML Engineer
