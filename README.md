**🎓 Student GPA Predictor (Machine Learning Web App)**

This project is an end-to-end Machine Learning application that predicts a student's GPA based on behavioral and academic factors such as study hours, attendance, and sleep patterns.

The goal of this project is to demonstrate the full ML workflow:

* Data simulation & preprocessing
* Model training using scikit-learn
* Model evaluation
* Deployment as an interactive web app using Streamlit

---

**🚀 Live Demo**
https://gpa-predictor-app.streamlit.app/

---

**🧠 Project Motivation**

Educational institutions often need early indicators of academic performance to identify students who may require support.

This project simulates how machine learning can be used to estimate student outcomes and categorize academic risk levels.

---

**⚙️ Tech Stack**

* Python
* Pandas & NumPy
* Scikit-learn (Linear Regression)
* Streamlit (Web App)
* Joblib (Model persistence)

---

**📊 Features**

* Interactive GPA prediction using sliders
* Real-time model inference
* Risk categorization (Low / Medium / High)
* Simple, interpretable regression model

---

**🏗️ Project Structure**

```
app.py               # Streamlit web app
train_model.py       # Model training script
gpa_model.joblib     # Saved trained model
requirements.txt     # Dependencies
```

---

**▶️ How to Run Locally**

Clone the repository:

```
git clone https://github.com/AbdelrahmanEhssan/gpa-predictor-streamlit.git
cd gpa-predictor-streamlit
```

Install dependencies:

```
pip install -r requirements.txt
```

Train the model:

```
python train_model.py
```

Run the app:

```
streamlit run app.py
```

---

**📌 Future Improvements**

* Use real educational datasets instead of simulated data
* Add more behavioral and demographic features
* Try advanced models (Random Forest, Gradient Boosting)
* Add visual analytics dashboard

---

**👤 Author**

**Abdelrahman Ehssan**
Computer Science Senior | Aspiring AI & Data Science Engineer

LinkedIn: https://www.linkedin.com/in/abdelrahman-ehssan/

---

⭐ If you found this project useful, feel free to star the repo.
