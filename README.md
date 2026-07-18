# 🚀 ML-Pipeline-DVC-AWS-S3
### End-to-End Machine Learning Pipeline with DVC, AWS S3, GitHub Actions & MLOps

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-green)
![AWS](https://img.shields.io/badge/AWS-S3-yellow)
![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub%20Actions-black)
![License](https://img.shields.io/badge/License-MIT-red)

---

## 📌 Project Overview

This project demonstrates an **End-to-End Machine Learning Pipeline** following industry-standard MLOps practices.

The pipeline automates every stage of the machine learning lifecycle—from data ingestion to model deployment—while ensuring reproducibility, version control, automation, and scalability.

The project integrates:

- Data Version Control (DVC)
- AWS S3 for remote artifact storage
- Modular ML pipeline architecture
- GitHub Actions for CI/CD
- Docker containerization
- Experiment tracking
- Reproducible workflows


---

# 📖 Machine Learning Pipeline

The pipeline consists of multiple independent components.

```
Raw Data
    │
    ▼
Data Ingestion
    │
    ▼
Data Validation
    │
    ▼
Data Transformation
    │
    ▼
Model Training
    │
    ▼
Model Evaluation
    │
    ▼
Model Saving
    │
    ▼
Deployment
```

---

# 🏗 Project Structure

```
ML-Pipeline-DVC-AWS-S3/

│
├── artifacts/
│
├── config/
│
├── notebook/
│
├── src/
│   ├── components/
│   ├── configuration/
│   ├── constants/
│   ├── entity/
│   ├── exception/
│   ├── logging/
│   ├── pipeline/
│   ├── utils/
│   └── cloud/
│
├── data/
│
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
├── setup.py
├── .gitignore
├── Dockerfile
├── README.md
│
└── main.py
```

---

# ⚙ Pipeline Components

## 1️⃣ Data Ingestion

Responsible for:

- Reading raw dataset
- Splitting train/test data
- Saving artifacts
- Creating reproducible datasets

Output:

```
artifacts/
    data_ingestion/
        train.csv
        test.csv
```

---

## 2️⃣ Data Validation

Ensures data quality by checking

- Missing values
- Duplicate records
- Schema validation
- Data drift
- Invalid columns

Output

```
Validation Report
Validated Dataset
```

---

## 3️⃣ Data Transformation

Performs preprocessing such as

- Missing value handling
- Encoding categorical variables
- Feature engineering
- Feature scaling
- Pipeline creation

Output

```
preprocessor.pkl
Transformed Train Data
Transformed Test Data
```

---

## 4️⃣ Model Training

Responsible for

- Training multiple models
- Hyperparameter tuning
- Model comparison
- Selecting best model

Example Algorithms

- Random Forest
- Decision Tree
- Logistic Regression
- XGBoost
- Gradient Boosting

Output

```
model.pkl
```

---

## 5️⃣ Model Evaluation

Evaluates the trained model using

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

Best model is selected automatically.

---

## 6️⃣ Model Saving

Stores

```
trained_model.pkl
preprocessor.pkl
```

These artifacts can later be used during inference.

---

# 📦 DVC Integration

Data Version Control (DVC) is used for

- Dataset Versioning
- Model Versioning
- Pipeline Reproducibility
- Experiment Tracking

Pipeline execution

```bash
dvc repro
```

Visualize pipeline

```bash
dvc dag
```

Check experiment

```bash
dvc metrics show
```

---

# ☁ AWS S3 Integration

AWS S3 is used as remote storage for

- Dataset
- Models
- Artifacts
- DVC Cache

Configure remote

```bash
dvc remote add -d storage s3://your-bucket-name
```

Push artifacts

```bash
dvc push
```

Pull artifacts

```bash
dvc pull
```

---

# 🔄 CI/CD Pipeline

GitHub Actions automates

- Code checkout
- Dependency installation
- Pipeline execution
- Testing
- Docker image build
- Push image to container registry
- Deployment

Workflow

```
Developer
      │
      ▼
Git Push
      │
      ▼
GitHub Actions
      │
      ▼
Run Tests
      │
      ▼
Train Model
      │
      ▼
Build Docker Image
      │
      ▼
Push to Registry
      │
      ▼
Deployment
```

---