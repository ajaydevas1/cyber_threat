# Cyber Threat Detection System

## 📌 Project Overview
This project focuses on **detecting cyber threats by transforming raw security data into meaningful, analyzable datasets** and applying intelligent techniques such as **Fuzzy Logic, Natural Language Processing (NLP), and Deep Learning (DL)**.

The core idea is simple but powerful:
> Raw cybersecurity data is noisy, unstructured, and mostly useless until it is cleaned, engineered, and modeled correctly.

This system demonstrates how different datasets can be processed and analyzed to identify malicious activity, spam patterns, and abnormal network behavior.

---

## 🚀 Key Objectives
- Convert **raw cyber data into structured and meaningful datasets**
- Detect threats using **rule-based + learning-based approaches**
- Experiment with **fuzzy decision logic**, **NLP-based text analysis**, and **deep learning models**
- Provide a foundation that can later scale into a real-time cyber threat monitoring system

---

## 🛠️ Technologies & Libraries Used

### Core Language
- **Python**

### Machine Learning & Deep Learning
- `scikit-learn`
- `TensorFlow / Keras`
- `NumPy`
- `Pandas`

### NLP & Fuzzy Logic
- `NLTK`
- `spaCy`
- `fuzzywuzzy`
- `scikit-fuzzy`

### Data Processing & Visualization
- `Matplotlib`
- `Seaborn`

---

## 🧠 Logic & Methodology

### 1️⃣ Data Preprocessing
- Raw data cleaning (null values, duplicates, noise removal)
- Feature engineering and normalization
- Label encoding and categorical transformations

### 2️⃣ Fuzzy Logic Layer
- Used where **binary decisions are unrealistic**
- Helps handle uncertainty in threat classification
- Assigns threat levels instead of hard labels

### 3️⃣ NLP-Based Detection
- Applied mainly to **Spam Detection**
- Text vectorization (TF-IDF / Bag of Words)
- Classification of malicious vs benign content

### 4️⃣ Deep Learning Models
- Used for learning complex patterns in large datasets
- Handles non-linear relationships in network traffic data

---

## 📊 Datasets Used

1. **KDD Train Dataset**
   - Used for training cyber intrusion detection models
2. **KDD Test Dataset**
   - Used for evaluating model generalization
3. **Spam Detection Dataset**
   - Used for NLP-based spam classification

> These datasets simulate real-world cyber threats such as intrusions, malicious payloads, and spam attacks.

---

## 🔢 Input & Output

### Inputs
- Three datasets:
  - KDD Train
  - KDD Test
  - Spam Detection Dataset

### Outputs
- Classified threats (benign vs malicious)
- Threat confidence / fuzzy risk levels
- Performance metrics (accuracy, precision, recall, confusion matrix)

### How to run
Clone the repository
git clone https://github.com/ajaydevas1/cyber_threat.git
cd cyber_threat

### Install dependencies
pip install -r requirements.txt

