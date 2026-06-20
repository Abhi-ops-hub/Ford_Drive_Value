# 🚗 Ford Drive Value Prediction

An end-to-end Machine Learning project that predicts the resale value of Ford vehicles using data preprocessing, feature engineering, encoding techniques, model training, evaluation, and deployment.

The project compares different encoding approaches and deploys the best-performing model through an interactive Streamlit web application.

## 🌐 Live Application

🚀 **Streamlit Deployment:**  
https://forddrivevalue.streamlit.app/

---

# 📌 Project Overview

The objective of this project is to build a Machine Learning model that can predict Ford vehicle prices based on various vehicle features.

The complete workflow includes:

- Data collection
- Data cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Handling categorical variables
- Model training
- Model evaluation
- Deployment using Streamlit

The final model is integrated into a web application where users can input vehicle details and get an estimated vehicle value.

---

# ✨ Key Features

✅ Complete Machine Learning pipeline  
✅ Data preprocessing and feature engineering  
✅ Comparison between Label Encoding and One Hot Encoding  
✅ Regression model training  
✅ Model performance evaluation  
✅ Interactive Streamlit prediction application  
✅ Deployment-ready ML project structure  

---

# 🛠️ Technologies Used

## Programming Language
- Python

## Libraries & Frameworks

- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- Jupyter Notebook

---

# 📂 Project Structure

```
FORD_DRIVE_VALUE
│
├── app.py
│   └── Streamlit web application
│
├── data sets/
│   ├── ford.csv (Original dataset)
│   ├── label_encoded.csv (Label encoded dataset)
│   └── one_hot_encoded.csv (One Hot encoded dataset)
│
├── notebooks/
│   ├── code.ipynb (Data preprocessing and analysis)
│   ├── 1hotencodedmodel.ipynb (Model training using One Hot Encoding)
│   └── one_label_encoding_model.ipynb (Model training using Label Encoding)
│
├── requirements.txt
└── README.md
```

---

# 🔄 Machine Learning Workflow

```
Dataset
   |
   ↓
Data Cleaning
   |
   ↓
Exploratory Data Analysis
   |
   ↓
Feature Engineering
   |
   ↓
Categorical Encoding
   |
   ↓
Model Training
   |
   ↓
Model Evaluation
   |
   ↓
Best Model Selection
   |
   ↓
Streamlit Deployment
```

---

# 🧠 Feature Engineering & Encoding

Categorical features cannot directly be used by Machine Learning algorithms, therefore different encoding techniques were tested.

## 1️⃣ Label Encoding

Label Encoding converts categorical values into numerical labels.

Example:

```
Manual → 0
Automatic → 1
```

### Model Performance:

**Accuracy: 73%**

---

## 2️⃣ One Hot Encoding

One Hot Encoding creates separate binary columns for each category.

Example:

```
Fuel Type

Petrol
Diesel

↓

Petrol = 1,0
Diesel = 0,1
```

### Model Performance:

**Accuracy: 84%**

---

# 📊 Model Comparison

| Encoding Technique | Accuracy |
|--------------------|----------|
| Label Encoding | 73% |
| One Hot Encoding | 84% |

---

# 🏆 Final Model Selection

After evaluating both approaches, the **One Hot Encoding based model was selected as the final production model** because it achieved better performance.

The deployed Streamlit application uses the model trained with **One Hot Encoded features**.

---

# 🚀 How to Run Locally

## 1. Clone the Repository

```bash
git clone https://github.com/Abhi-ops-hub/Ford_Drive_Value.git
```

## 2. Navigate to Project Directory

```bash
cd Ford_Drive_Value
```

## 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 📈 Future Improvements

Future improvements planned:

- Hyperparameter tuning for better accuracy
- Testing advanced regression algorithms
- Automated ML pipeline
- Model monitoring
- Cloud deployment
- Adding more vehicle datasets

---

# 📌 Learning Outcomes

Through this project, I gained practical experience in:

- Data preprocessing
- Feature engineering
- Handling categorical variables
- Comparing ML approaches
- Model evaluation
- Deploying ML applications using Streamlit

---

# 👨‍💻 Author

**Abhishek Goswami**

GitHub:  
https://github.com/Abhi-ops-hub

---

⭐ If you found this project useful, consider giving it a star!
