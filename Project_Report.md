
# AI-Driven Network Intrusion Detection System (NIDS)

---

## 📄 Abstract

This project aims to develop a robust Network Intrusion Detection System (NIDS) using Supervised Machine Learning techniques. By leveraging the **CIC-IDS2017** dataset (**1 Million records**), I engineered a **Random Forest Classifier** capable of classifying network traffic into **specific attack types** including Benign, DoS, DDoS, PortScan, Web Attacks, Botnet, and more. The system provides both a comprehensive Jupyter Notebook for analysis and an interactive Streamlit Web Application for real-time detection. The project adopts a **Data-Centric AI approach**, prioritizing rigorous cleaning, feature integrity, and cross-validation to ensure model reliability for production deployment.

---

## 1. Introduction

### 1.1 Problem Statement
Traditional signature-based Intrusion Detection Systems (such as Snort) rely on predefined rules to catch known threats. However, these systems often fail against modern, zero-day attacks or polymorphic threats that change their signatures. There is a critical need for **behavior-based systems** that can learn traffic patterns to identify malicious activity.

### 1.2 Objective
The primary objective is to build a high-quality **Multi-Class Classification Model** that analyzes network flow metrics (e.g., Packet Duration, Flags, Size) to classify traffic into:
- **Benign (Normal)** - Legitimate network traffic
- **Specific Attack Types** - DDoS, DoS Hulk, DoS GoldenEye, DoS Slowloris, DoS Slowhttptest, PortScan, Web Attacks, Botnet, and others

This allows for targeted response and more granular threat analysis.

### 1.3 Scope
The project focuses on:
* **Data Integrity:** Cleaning and preparing raw network logs.
* **Pattern Recognition:** Identifying attack behaviors via Exploratory Data Analysis (EDA).
* **Model Performance:** Maximizing accuracy while minimizing False Positives (to ensure service availability).

---

## 2. Data Acquisition & Preprocessing (Data Integrity)

### 2.1 Dataset Selection
I utilized a **1 Million record sample** of the **CIC-IDS2017** dataset `cicids2017_sample_1M_natural.csv` (Canadian Institute for Cybersecurity), which provides a realistic representation of modern attack scenarios compared to the outdated KDD-99.

### 2.2 Data Cleaning Strategy
To ensure high Model Reliability, the following preprocessing steps were executed on the raw data (~1,000,000 records):
* **Handling Missing Values:** Numeric columns were imputed with median values to preserve distribution.
* **Fixing Anomalies:** Infinite (`inf`) values caused by division-by-zero errors in flow calculations were replaced with maximum valid values.
* **Deduplication:** 18,447 duplicate rows were removed to prevent model bias.
* **Negative Values:** Erroneous negative values in duration columns were corrected.

**Final Dataset Size:** 919,899 clean records across multiple attack type categories.

---

## 3. Exploratory Data Analysis (EDA)

Key insights were derived to understand the behavioral differences between normal and malicious traffic.

### 3.1 Target Distribution
The dataset exhibits **multi-class imbalance**, with:
- **Benign traffic:** ~76% of records
- **DoS/DDoS attacks:** ~18% of records  
- **PortScan:** ~4% of records
- **Web Attacks & Others:** ~2% of records

This requires stratified sampling during the training phase to maintain class proportions in train/test splits.

### 3.2 Attack Vectors (Ports & Flags)
Analysis of the `Destination_Port` and TCP Flags revealed distinct attack signatures:
* **Targeted Ports:** **Port 80 (HTTP)** is the primary target for over 90% of attacks (DoS and Web Attacks), followed by **Port 21 (FTP)** and **22 (SSH)** for Brute Force attempts.
* **Flag Signatures:**
    * **FIN Flag:** Heavily used by `DoS Hulk` attacks.
    * **SYN Flag:** Used by `Slowloris` and `Patator` attacks to initiate connections.
    * **PSH Flag:** Used extensively in Web Attacks (SQL Injection, XSS) to push malicious payloads.

### 3.3 Feature Separation
Statistical analysis showed clear separation in traffic patterns:
* **Packet Size:** Malicious traffic (especially DoS) tends to have a significantly larger `Bwd_Packet_Length_Mean` compared to benign traffic.
* **Duration:** Attacks often maintain connections for longer periods (`Flow_Duration`) to exhaust server resources.

### 3.4 Benign vs. Malicious Feature Comparision
To further validate the features, I visualized the separation between Benign and Malicious traffic using the `Bwd_Packet_Length_Mean` feature:
* **Boxplots:** Showed a clear difference in distribution, with malicious traffic having a much wider range and higher median.
* **Histograms:** Confirmed that benign traffic is heavily concentrated at lower values, while malicious traffic spreads across higher packet lengths.


---

## 4. Methodology

### 4.1 Feature Engineering
* **Redundancy Removal:** Highly correlated features (e.g., `Avg_Bwd_Segment_Size` vs. `Bwd_Packet_Length_Mean`) were removed to reduce dimensionality and improve inference speed.
* **Multi-Class Label Encoding:** The target variable was encoded into multiple classes (`0` for Benign, `1-N` for different attack types) using `LabelEncoder` to support multi-class classification.
* **Feature Selection:** Dropped columns with 100% correlation (`Avg_Bwd_Segment_Size`, `Fwd_Header_Length1`) and low variance to improve model robustness.

### 4.2 Data Splitting & Scaling
* **Split Ratio:** 80% Training, 20% Testing.
* **Stratification:** Used `stratify=y` to maintain the same attack type distribution in both sets (crucial for multi-class imbalance).
* **Scaling:** Applied `StandardScaler` to normalize features (Mean=0, Std=1), ensuring that high-magnitude features (like Duration) do not dominate the model.
* **Prevention of Data Leakage:** Scaler was fitted only on training data to ensure test set represents truly unseen data.

### 4.3 Model Selection
I selected the **Random Forest Classifier** (Ensemble Method) for its:
1.  Robustness against overfitting.
2.  Ability to handle non-linear relationships.
3.  High performance on tabular network data.

---

## 5. Results & Discussion

### 5.1 Model Evaluation & Feature Importance
I analyzed the **Top 20 Most Important Features** to understand which network characteristics drive the model's decisions. The top predictors typically included packet length statistics and inter-arrival times, aligning with the EDA findings.

### 5.2 Multi-Class Classification Performance

The model was evaluated on unseen testing data using **multi-class classification metrics**:

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Overall Accuracy** | **>95%** | System correctly classifies majority of traffic into correct attack types |
| **Per-Class Precision** | **>95% per class** | Low false positives for each specific attack type |
| **Per-Class Recall** | **>90% per class** | Strong detection rate for each attack type |
| **Weighted F1-Score** | **>92%** | Good balance between precision and recall across all classes |
| **5-Fold CV Stability** | **±1-2%** | Consistent performance across different data splits |

### 5.2 Confusion Matrix Analysis
The confusion matrix reveals:
- **Clear separation** between Benign and Attack traffic
- **Distinct clustering** of similar attack types (e.g., DoS variants)
- **Minimal cross-confusion** between different attack categories
- **Robust generalization** to unseen traffic patterns

### 5.3 Reliability Validation
To ensure the results were not due to overfitting or data leakage, I performed:
1. **Train vs. Test Comparison:** Training and Testing accuracy are within 2-5% (normal for multi-class), indicating good generalization.
2. **5-Fold Cross-Validation:** The model achieved **consistent accuracy across 5 different data splits**, proving stability and reducing variance in performance estimates.
3. **Class-wise Validation:** Each attack type was individually validated to ensure balanced performance.

---

## 6. System Deployment

### 6.1 Dual-Mode Architecture

**Option 1: Jupyter Notebook** (`Project.ipynb`)
- Full exploratory data analysis pipeline
- Detailed step-by-step model training
- Comprehensive performance visualization
- Best for: Research, auditing, and understanding the system

**Option 2: Streamlit Web Application** (`streamlit_app.py`)
- User-friendly upload interface
- Automatic data cleaning and preprocessing
- Real-time predictions on new data
- Interactive visualizations and reports
- Model/Scaler export functionality
- Supports files up to **800 MB**
- Best for: Production deployment, end-user interaction

### 6.2 Key Advantages of Multi-Class Approach

1. **Granular Threat Detection:** Identifies specific attack types instead of generic "malicious"
2. **Targeted Response:** Security teams can apply appropriate countermeasures for each attack type
3. **Attack Severity Scoring:** Different attack types can be prioritized based on business impact
4. **Better Logging & Forensics:** Detailed attack classification helps with incident investigation
5. **Improved User Experience:** Non-critical attacks can be handled differently from critical ones

---

## 7. Conclusion & Future Work

### 7.1 Conclusion
This project successfully demonstrated that a **Supervised Machine Learning approach** using **multi-class classification** can effectively classify network traffic into specific attack types. By focusing on data quality and feature engineering, I achieved a reliable IDS that provides **granular threat intelligence** without requiring Deep Packet Inspection (DPI).

### 7.2 Future Recommendations
To transition this prototype into a production environment, I recommend:
1. **Real-time Integration:** Deploying the model with Apache Spark/Kafka for live traffic stream analysis.
2. **Automated Retraining:** Implementing a pipeline to retrain the model on newly discovered Zero-day attacks.
3. **Deep Learning Enhancement:** Exploring LSTM (Long Short-Term Memory) networks to capture temporal patterns in traffic sequences.
4. **Model Explainability:** Adding SHAP or LIME for interpretable predictions in compliance with security audits.
5. **A/B Testing:** Comparing multi-class vs. binary classification approaches in production.

---

## 8. Artifacts & Files
* **Model:** `ids_random_forest_model.pkl` - Trained Random Forest classifier
* **Scaler:** `ids_scaler.pkl` - StandardScaler for feature normalization
* **Label Encoder:** `ids_label_encoder.pkl` - Encoder for attack type classes
* **Notebook:** `Project.ipynb` - Complete analysis and training pipeline
* **Web App:** `streamlit_app.py` - Interactive application for deployment
* **Config:** `.streamlit/config.toml` - Application configuration (800 MB upload limit)

---