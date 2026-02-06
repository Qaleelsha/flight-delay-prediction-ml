# ✈️ Flight Delay Prediction – End-to-End Machine Learning Project

## 📌 Overview
Flight delays cause major disruptions for airlines and passengers, leading to operational losses and poor customer experience.  
This project builds a **production-ready machine learning system** to predict whether a flight will be delayed by **15 minutes or more**, using real-world flight data.

The project covers the **complete ML lifecycle** — from data analysis and feature engineering to model deployment as a live web application.
---
Live app: https://flight-delay-prediction-ml.streamlit.app/
---

## 🎯 Problem Statement
Given flight details such as airline, origin, destination, departure time, and date, predict whether the flight will be delayed by **15+ minutes**.

This is framed as a **binary classification problem**:
- `0` → Flight on time  
- `1` → Flight delayed  

---

## 📊 Dataset
- **Source:** Public US flight data (2021–2023)
- **Why this dataset?**
  - Recent and realistic
  - Large-scale, real-world aviation data
  - Suitable for production-style ML problems

### Key Features Used
- Airline code
- Origin & destination airport
- Flight distance
- Departure hour
- Day of week
- Month

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights discovered:
- Evening flights have a higher chance of delays
- Certain airports and airlines are consistently more delay-prone
- Seasonal patterns affect delay probability
- Flight delays are **class-imbalanced**, requiring careful modeling decisions

EDA was used to guide feature engineering and model selection.

---

## 🧠 Feature Engineering & Preprocessing
- Extracted time-based features (hour, weekday, month)
- Encoded categorical variables using One-Hot Encoding
- Handled class imbalance using evaluation-driven model selection
- Prevented data leakage by separating training and inference logic

---

## 🤖 Model Development
Trained and evaluated multiple models:
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting (final model)

### Model Selection Criteria
- ROC-AUC
- Recall (important for catching delayed flights)
- Business impact over raw accuracy

### Key Decision
Instead of using the default 0.5 threshold, the classification threshold was **tuned to improve recall**, making predictions more realistic for operational use.

---

## 📈 Model Performance
- Balanced precision and recall for delayed flights
- Robust performance across unseen data
- Stable predictions during live inference

---

## 🔍 Model Interpretability
- Analyzed feature importance to understand delay drivers
- Identified departure time, airport congestion, and seasonality as key contributors
- Ensured predictions are explainable and interview-ready

---

## 🌐 Web Application
A user-friendly **Streamlit web app** was built to:
- Input flight details
- Predict delay status
- Display delay probability for transparency

### Live App:
👉 **https://flight-delay-prediction-ml.streamlit.app/**

---

## 🚀 Deployment
- Deployed on **Streamlit Cloud**
- Model artifacts versioned for reproducibility
- Clean dependency management using `requirements.txt`
- Same preprocessing pipeline used for training and inference

---

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Joblib
- **Visualization:** Matplotlib, Seaborn
- **Web App:** Streamlit
- **Tools:** Git, GitHub

---

## 📁 Project Structure
flight-delay-prediction-ml/
│
├── app/ # Streamlit application
├── src/ # Reusable ML pipeline code
├── notebooks/ # EDA and experimentation
├── data/ # Data structure (ignored from git)
├── models/ # Trained model artifacts
├── requirements.txt
└── README.md


---

## 🎓 What This Project Demonstrates
- End-to-end ownership of an ML system
- Strong data analysis and feature engineering skills
- Practical model evaluation beyond accuracy
- Deployment and real-world inference understanding
- Clean, maintainable ML code structure

---

👨‍💻 Author
Qaleel Sha Backer B.Tech CSE (Data Science & Machine Learning)

🔗 LinkedIn: https://www.linkedin.com/in/qaleel-sha-backer 🔗 GitHub: https://github.com/Qaleelsha

---
## 📬 Looking for Opportunities
I am actively seeking **Data Analyst / Machine Learning Intern** roles where I can apply data-driven problem solving to real-world systems.

📩 Feel free to connect or reach out!
---

⭐ If you like this project, consider starring the repository!