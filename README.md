
# # AI-Driven Network Intrusion Detection System (NIDS)

**Course:** Quality Assurance

**Topic:** Machine Learning for Cybersecurity

**Author:** Mahmoud Abdel Nasser

---

## 📄 Executive Summary

This project delivers a comprehensive Network Intrusion Detection System (NIDS) built on the **CICIDS2017** dataset (**1 Million records**). By applying rigorous Quality Assurance (QA) protocols to data cleaning and feature engineering, I developed a **Random Forest Classifier** that accurately classifies network traffic into **multiple attack types** (not just binary classification). The system provides both a Jupyter Notebook for in-depth analysis and an interactive **Streamlit Web Application** for real-time detection and visualization.

---

## 📊 Dataset Specifications

* **Source:** CICIDS2017 (Canadian Institute for Cybersecurity).
* **Initial Volume:** ~1,000,000 records from the raw flows.
* **Final Cleaned Dataset:** **919,899 records** (after removing duplicates and corrupted entries).
* **Target Classes (Multi-Class Classification):**
  - **Benign:** Normal web activity
  - **Botnet:** Bot-controlled malicious traffic
  - **DoS Attacks:** Denial of Service (Hulk, GoldenEye, Slowloris, Slowhttptest)
  - **DDoS:** Distributed Denial of Service
  - **PortScan:** Network reconnaissance attacks
  - **Web Attacks:** XSS, Brute Force, and SQL Injection



---

## ✅ Project Phases (Updated)

1. **✅ Data Collection & Sampling:** Stratified sampling from CICIDS2017 dataset (~300K records).
2. **✅ Data Cleaning (QA):** Handling infinite values (`inf`), nulls, and removing duplicates.
3. **✅ Exploratory Data Analysis (EDA):** Benign vs. Malicious feature comparison (Boxplots), Port 80 targets, and TCP flag patterns.
4. **✅ Feature Engineering:** Multi-class label encoding, standardization via `StandardScaler`, and explicit removal of redundant features (`Avg_Bwd_Segment_Size`, etc.).
5. **✅ Statistical Analysis:** Top 20 Feature Importance ranking and correlation analysis.
6. **✅ Multi-Class Modeling & Validation:** Training, testing, and 5-Fold Cross-Validation for attack type prediction.
7. **✅ Web Application:** Interactive Streamlit app for real-time detection and analysis.

---

## ⚙️ Methodology & Quality Assurance

### Data Preprocessing

To ensure the highest data integrity, I implemented a robust pipeline:

* **Infinity Handling:** Replaced `inf` values caused by division-by-zero errors in flow calculations with maximum valid integers.
* **Feature Selection:** Dropped highly correlated features (e.g., `Avg_Bwd_Segment_Size`) to prevent multicollinearity and optimize inference speed.
* **Scaling:** Applied Standardization to ensure all network metrics (Durations vs. Packet counts) share the same statistical weight.

### Modeling: Random Forest Classifier

The choice of **Random Forest** was driven by its ensemble nature, providing robustness against overfitting—a critical factor in security QA.

---

## 🏆 Key Results

The model was evaluated on an unseen test set using **multi-class classification**:

| Metric | Score | Details |
|--------|-------|---------|
| **Overall Accuracy** | **High (>95%)** | Excellent performance across all attack types |
| **Per-Class Precision** | **>95% per class** | Low false positives for each attack type |
| **Per-Class Recall** | **>90% per class** | Strong detection of specific attacks |
| **Confusion Matrix** | **Clear Separation** | Distinct boundaries between attack types |
| **5-Fold Cross-Validation** | **Stable & Robust** | Consistent performance across data splits |

### Multi-Class Classification Advantages

- Identifies **specific attack types** instead of just "malicious/benign"
- Enables **targeted incident response** based on attack category
- Provides **attack severity scoring** for prioritization
- Better support for **zero-day attack detection** through pattern analysis

---

## 💻 How to Use the System

### Option 1: Jupyter Notebook (In-Depth Analysis)
```bash
jupyter notebook Project.ipynb
```
- Full data cleaning pipeline
- Exploratory data analysis with visualizations
- Step-by-step model training
- Detailed performance metrics

### Option 2: Streamlit Web Application (Real-Time Detection)
```bash
streamlit run streamlit_app.py
```
- **Upload uncleaned CSV data**
- **Automatic data cleaning & preprocessing**
- **Interactive exploratory analysis**
- **Real-time attack type prediction**
- **Download results and trained model**
- **Supports files up to 800 MB**

---

## 🛠️ Technologies

* **Environment:** Jupyter Notebook (Anaconda).
* **Core Libraries:** Python 3.13.9, Pandas (Data Manipulation), Scikit-learn (ML Pipeline), Seaborn/Matplotlib (Visualization).
* **Serialization:** Joblib (for model persistence).