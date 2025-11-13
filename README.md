# Water Pollution Detection Using AI 🌊🤖

This project predicts whether water is Safe, Moderate, or Contaminated using machine learning models trained on water quality parameters. It was developed as part of the Green AI Training Program (Skills4Future – Edunet Foundation).

## 🔍 Features
- Input water parameters: pH, DO, Conductivity, BOD, Nitrate, Temperature, Fecal Coliform, Total Coliform  
- ML models used: Decision Tree, Random Forest, Logistic Regression, KNN  
- Built in Google Colab  
- Deployed using a Streamlit web app  
- Model saved as .joblib pipeline + label encoder

## 🧠 Tech Stack
- Python  
- Scikit-Learn  
- Pandas, NumPy  
- Streamlit  
- Matplotlib, Seaborn

## 📁 Project Structure
```
water-pollution-detection-ai/
│
├── streamlit_app.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── dt_model_pipeline.joblib
│   ├── label_encoder.joblib
│
├── data/
│   ├── water_data.csv
│
├── notebooks/
│   ├── project_WaterPollution.ipynb.ipynb
```
