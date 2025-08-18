# 💼 Developer Salary Prediction and Data Exploration

## 📌 Project Overview
This project focuses on predicting **annual developer salaries** using the **Stack Overflow Developer Survey 2024** dataset.  
A **Random Forest Regressor** was trained with **hyperparameter tuning (RandomizedSearchCV)** to optimize performance.  
The goal was not only accurate prediction but also meaningful insights into how demographic, educational, and professional factors influence salaries.  
The final model was deployed as an **interactive Streamlit web app** for real-time predictions and salary trend exploration.

---

## ⚙️ Tools & Technologies
- **Programming & Libraries**: Python, Pandas, NumPy, Scikit-learn  
- **Modeling**: Random Forest Regressor with hyperparameter tuning (RandomizedSearchCV)  
- **Deployment**: Streamlit, Pickle  

---

## 📊 Model Performance
- **Train Set** → RMSE: `16,861.78`, R²: `0.871`  
- **Test Set** → RMSE: `20,866.17`, R²: `0.803`  

✔️ The model explains **~87% of salary variance on training data** and **~80% on unseen data**,  
with an average prediction error of about **$20k** in salary estimation.

---

## 🚀 Key Outcomes
- Built a **robust, high-accuracy salary prediction system** (R² ≈ 0.85).  
- Outperformed baseline models like **Decision Trees** and **Linear Regression**.  
- Deployed as a **Streamlit web application** for interactive salary predictions.  
- Enables **real-time insights** into salary trends across demographics, education, and professional attributes.  

---

## 📂 Project Structure
.
├── data/ # Dataset (Stack Overflow Developer Survey 2024)
├── notebooks/ # Jupyter notebooks for exploration & model training
├── models/ # Saved model artifacts (.pkl files)
├── app/ # Streamlit application
└── README.md # Project documentation
