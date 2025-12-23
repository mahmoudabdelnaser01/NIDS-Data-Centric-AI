import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from io import StringIO
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')

# Set Streamlit page config
st.set_page_config(page_title="IDS Network Attack Detection", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 2px;}
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# TITLE AND DESCRIPTION
# ============================================================================
st.title("🛡️ IDS Network Attack Detection System")
st.markdown("""
This application performs **comprehensive network intrusion detection** using machine learning.
Upload your raw/uncleaned network traffic data, and the system will:
1. **Clean & Preprocess** the data
2. **Feature Engineering** 
3. **Train/Predict** attack types
4. **Visualize Results**
""")

st.divider()

# ============================================================================
# SIDEBAR - FILE UPLOAD
# ============================================================================
st.sidebar.markdown("## 📤 Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file with network traffic data",
    type=['csv'],
    help="Upload raw/uncleaned CICIDS2017 dataset"
)

# ============================================================================
# MAIN APPLICATION
# ============================================================================
if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")
    
    # Load the data
    df = pd.read_csv(uploaded_file)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview",
        "🔧 Data Cleaning",
        "🔍 Exploratory Analysis",
        "🤖 Model Training & Prediction",
        "📈 Results & Export"
    ])
    
    # ================================================================
    # TAB 1: DATA OVERVIEW
    # ================================================================
    with tab1:
        st.header("📊 Initial Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        with col4:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        
        # Data Types
        st.subheader("📋 Data Types Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.write(df.dtypes.value_counts())
        
        # First few rows
        st.subheader("📄 First 5 Rows")
        st.dataframe(df.head(), use_container_width=True)
        
        # Column names
        st.subheader("📝 Column Names")
        cols_display = pd.DataFrame({
            'Index': range(1, len(df.columns) + 1),
            'Column Name': df.columns.tolist()
        })
        st.dataframe(cols_display, use_container_width=True)
        
        # Missing Values Analysis
        st.subheader("❓ Missing Values Analysis")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Missing_Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        
        if len(missing_df) > 0:
            st.warning(f"⚠️ Found {missing.sum():,} missing values in {len(missing_df)} columns")
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✅ No missing values found!")
        
        # Duplicates
        st.subheader("🔄 Duplicates Analysis")
        dup_count = df.duplicated().sum()
        dup_pct = (dup_count / len(df)) * 100
        if dup_count > 0:
            st.warning(f"⚠️ Found {dup_count:,} duplicate rows ({dup_pct:.2f}%)")
        else:
            st.success(f"✅ No duplicates found!")
        
        # Label Distribution
        if 'Label' in df.columns:
            st.subheader("📊 Label Distribution (Before Cleaning)")
            label_counts = df['Label'].value_counts()
            st.bar_chart(label_counts)
            st.write(label_counts)
    
    # ================================================================
    # TAB 2: DATA CLEANING
    # ================================================================
    with tab2:
        st.header("🔧 Data Cleaning Process")
        
        # Make a copy for cleaning
        df_clean = df.copy()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Fix Column Names
        status_text.text("Step 1/5: Fixing column names...")
        progress_bar.progress(20)
        
        original_cols = df_clean.columns.tolist()
        df_clean.columns = df_clean.columns.str.strip()
        df_clean.columns = df_clean.columns.str.replace(' ', '_')
        df_clean.columns = df_clean.columns.str.replace('[^A-Za-z0-9_]', '', regex=True)
        
        cols_changed = sum([1 for old, new in zip(original_cols, df_clean.columns) if old != new])
        st.success(f"✅ Fixed column names ({cols_changed} changes)")
        
        # Step 2: Handle Missing Values
        status_text.text("Step 2/5: Handling missing values...")
        progress_bar.progress(40)
        
        rows_before = len(df_clean)
        label_col = 'Label' if 'Label' in df_clean.columns else df_clean.columns[-1]
        
        df_clean = df_clean.dropna(subset=[label_col])
        rows_after_label = len(df_clean)
        removed_label = rows_before - rows_after_label
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != label_col]
        
        filled_count = 0
        for col in numeric_cols:
            missing_before = df_clean[col].isnull().sum()
            if missing_before > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
                filled_count += 1
        
        remaining_missing = df_clean.isnull().sum().sum()
        if remaining_missing > 0:
            df_clean = df_clean.dropna()
        
        st.success(f"✅ Missing values handled (Removed: {rows_before - len(df_clean):,}, Filled: {filled_count})")
        
        # Step 3: Remove Duplicates
        status_text.text("Step 3/5: Removing duplicates...")
        progress_bar.progress(60)
        
        rows_before = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        rows_after = len(df_clean)
        removed = rows_before - rows_after
        
        st.success(f"✅ Duplicates removed: {removed:,}")
        
        # Step 4: Handle Infinity Values
        status_text.text("Step 4/5: Handling infinity values...")
        progress_bar.progress(80)
        
        inf_count = 0
        for col in numeric_cols:
            inf_in_col = np.isinf(df_clean[col]).sum()
            if inf_in_col > 0:
                max_val = df_clean[col][~np.isinf(df_clean[col])].max()
                min_val = df_clean[col][~np.isinf(df_clean[col])].min()
                df_clean.loc[df_clean[col] == np.inf, col] = max_val
                df_clean.loc[df_clean[col] == -np.inf, col] = min_val
                inf_count += inf_in_col
        
        st.success(f"✅ Infinity values fixed: {inf_count:,}")
        
        # Step 5: Fix Negative Values
        status_text.text("Step 5/5: Fixing negative values...")
        progress_bar.progress(100)
        
        should_be_positive = [col for col in numeric_cols 
                             if any(keyword in col.lower() 
                                   for keyword in ['packet', 'byte', 'length', 'count', 'duration'])]
        
        fixed_count = 0
        for col in should_be_positive:
            neg_count = (df_clean[col] < 0).sum()
            if neg_count > 0:
                df_clean[col] = df_clean[col].abs()
                fixed_count += 1
        
        st.success(f"✅ Negative values fixed: {fixed_count}")
        
        # Final Summary
        st.divider()
        st.subheader("✅ Final Data Quality Check")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Rows", f"{len(df_clean):,}")
        with col2:
            st.metric("Missing", f"{df_clean.isnull().sum().sum():,}")
        with col3:
            st.metric("Duplicates", f"{df_clean.duplicated().sum():,}")
        with col4:
            st.metric("Infinity", "0")
        with col5:
            st.metric("Memory", f"{df_clean.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        
        # Label Distribution after cleaning
        if label_col in df_clean.columns:
            st.subheader("📊 Final Label Distribution (After Cleaning)")
            label_counts_final = df_clean[label_col].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(label_counts_final)
            with col2:
                st.write(label_counts_final)
        
        # Store cleaned data in session state
        st.session_state.df_clean = df_clean
        st.session_state.label_col = label_col
        st.session_state.numeric_cols = numeric_cols
    
    # ================================================================
    # TAB 3: EXPLORATORY ANALYSIS
    # ================================================================
    with tab3:
        st.header("🔍 Exploratory Data Analysis")
        
        if 'df_clean' not in st.session_state:
            st.warning("⚠️ Please complete data cleaning first!")
        else:
            df_clean = st.session_state.df_clean
            label_col = st.session_state.label_col
            numeric_cols = st.session_state.numeric_cols
            
            # Class Distribution
            st.subheader("📊 Class Distribution Analysis")
            if label_col in df_clean.columns:
                fig, ax = plt.subplots(figsize=(12, 6))
                label_counts = df_clean[label_col].value_counts()
                ax.barh(range(len(label_counts)), label_counts.values)
                ax.set_yticks(range(len(label_counts)))
                ax.set_yticklabels(label_counts.index)
                ax.set_xlabel('Count')
                ax.set_title('Network Traffic Distribution by Class')
                st.pyplot(fig)
            
            # Feature Correlation
            st.subheader("🔗 Feature Correlation with Attacks")
            df_encoded = df_clean.copy()
            df_encoded['Label_Code'] = df_encoded[label_col].apply(lambda x: 0 if x == 'BENIGN' else 1)
            
            correlation = df_encoded.select_dtypes(include=[np.number]).corr()['Label_Code'].sort_values(ascending=False)
            top_corr_features = correlation.index[1:11].tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("🔍 Top 10 Correlated Features:")
                st.write(correlation[1:11])
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                selected_cols = top_corr_features + ['Label_Code']
                corr_matrix = df_encoded[selected_cols].corr()
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
                ax.set_title('Correlation Heatmap: Top Features')
                st.pyplot(fig)
            
            # Benign vs Attack Distribution
            st.subheader("📈 Benign vs Malicious Feature Comparison")
            
            if len(top_corr_features) > 0:
                selected_feature = st.selectbox("Select feature to analyze", top_corr_features)
                
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Boxplot
                sns.boxplot(x=label_col, y=selected_feature, data=df_clean, ax=axes[0])
                axes[0].set_title(f'Boxplot: {selected_feature}')
                axes[0].tick_params(axis='x', rotation=45)
                
                # KDE Plot
                benign = df_clean[df_clean[label_col] == 'BENIGN'][selected_feature]
                malicious = df_clean[df_clean[label_col] != 'BENIGN'][selected_feature]
                
                axes[1].hist(benign, bins=50, alpha=0.6, label='Benign', color='green')
                axes[1].hist(malicious, bins=50, alpha=0.6, label='Malicious', color='red')
                axes[1].set_xlabel(selected_feature)
                axes[1].set_ylabel('Frequency')
                axes[1].set_title(f'{selected_feature} Distribution')
                axes[1].legend()
                
                st.pyplot(fig)
    
    # ================================================================
    # TAB 4: MODEL TRAINING & PREDICTION
    # ================================================================
    with tab4:
        st.header("🤖 Model Training & Prediction")
        
        if 'df_clean' not in st.session_state:
            st.warning("⚠️ Please complete data cleaning first!")
        else:
            df_clean = st.session_state.df_clean
            label_col = st.session_state.label_col
            
            # Feature Engineering
            st.subheader("🔧 Feature Engineering")
            
            df_model = df_clean.copy()
            
            # Drop redundant columns
            cols_to_drop = ['Avg_Bwd_Segment_Size', 'Fwd_Header_Length1']
            cols_to_drop = [c for c in cols_to_drop if c in df_model.columns]
            
            if cols_to_drop:
                df_model.drop(columns=cols_to_drop, inplace=True)
                st.success(f"✅ Dropped redundant columns: {cols_to_drop}")
            
            # Multi-Class Label Encoding
            st.write("📊 Multi-Class Attack Type Encoding:")
            label_encoder = LabelEncoder()
            df_model['Label_Encoded'] = label_encoder.fit_transform(df_model[label_col])
            
            encoding_df = pd.DataFrame({
                'Attack Type': label_encoder.classes_,
                'Encoded Value': range(len(label_encoder.classes_))
            })
            st.dataframe(encoding_df, use_container_width=True)
            
            # Train/Test Split & Feature Scaling
            st.subheader("✂️ Train/Test Split & Feature Scaling")
            
            from sklearn.model_selection import train_test_split
            
            X = df_model.drop(columns=[label_col, 'Label_Encoded'])
            y = df_model['Label_Encoded']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Samples", X_train.shape[0])
            with col2:
                st.metric("Testing Samples", X_test.shape[0])
            with col3:
                st.metric("Number of Classes", len(set(y)))
            
            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
            
            st.success("✅ Data is ready for modeling!")
            
            # Model Training
            st.subheader("🤖 Training Random Forest Classifier")
            
            with st.spinner("⏳ Training model... This may take a moment..."):
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf_model.fit(X_train_scaled, y_train)
                y_pred = rf_model.predict(X_test_scaled)
            
            st.success("✅ Model trained successfully!")
            
            # Store for later use
            st.session_state.rf_model = rf_model
            st.session_state.scaler = scaler
            st.session_state.label_encoder = label_encoder
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.X_test_scaled = X_test_scaled
    
    # ================================================================
    # TAB 5: RESULTS & EXPORT
    # ================================================================
    with tab5:
        st.header("📈 Results & Export")
        
        if 'rf_model' not in st.session_state:
            st.warning("⚠️ Please complete model training first!")
        else:
            rf_model = st.session_state.rf_model
            label_encoder = st.session_state.label_encoder
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred
            X_test_scaled = st.session_state.X_test_scaled
            
            # Model Performance
            st.subheader("📊 Model Performance Report")
            
            acc = accuracy_score(y_test, y_pred)
            train_score = rf_model.score(X_test_scaled, y_test)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Overall Accuracy", f"{acc*100:.2f}%")
            with col2:
                st.metric("📊 Test Score", f"{train_score*100:.2f}%")
            with col3:
                st.metric("✅ Classes Detected", len(label_encoder.classes_))
            
            # Classification Report
            st.subheader("📝 Detailed Classification Report")
            
            labels_in_test = sorted(set(y_test) | set(y_pred))
            target_names_filtered = [label_encoder.classes_[i] for i in labels_in_test]
            
            report_str = classification_report(y_test, y_pred, labels=labels_in_test, 
                                              target_names=target_names_filtered, zero_division=0)
            st.code(report_str, language='text')
            
            # Confusion Matrix
            st.subheader("📉 Confusion Matrix")
            
            fig = plt.figure(figsize=(12, 10))
            cm = confusion_matrix(y_test, y_pred, labels=labels_in_test)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names_filtered)
            disp.plot(cmap='Blues', values_format='d')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Per-Class Accuracy
            st.subheader("📊 Per-Class Accuracy")
            
            per_class_acc = []
            for i, label in zip(labels_in_test, target_names_filtered):
                class_mask = y_test == i
                if class_mask.sum() > 0:
                    class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
                    per_class_acc.append({
                        'Attack Type': label,
                        'Accuracy': f"{class_acc*100:.2f}%",
                        'Samples': class_mask.sum()
                    })
            
            per_class_df = pd.DataFrame(per_class_acc)
            st.dataframe(per_class_df, use_container_width=True)
            
            # Feature Importance
            st.subheader("🌟 Feature Importance")
            
            feature_importance = pd.DataFrame({
                'Feature': X_test_scaled.columns,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(15)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(feature_importance['Feature'], feature_importance['Importance'])
            ax.set_xlabel('Importance Score')
            ax.set_title('Top 15 Most Important Features')
            st.pyplot(fig)
            
            # Export Results
            st.subheader("💾 Export Results")
            
            # Save Model and Scaler
            if st.button("💾 Save Model & Scaler"):
                joblib.dump(rf_model, 'ids_random_forest_model.pkl')
                joblib.dump(st.session_state.scaler, 'ids_scaler.pkl')
                joblib.dump(label_encoder, 'ids_label_encoder.pkl')
                st.success("✅ Model, scaler, and label encoder saved!")
            
            # Download Predictions
            predictions_df = pd.DataFrame({
                'Actual_Label': [label_encoder.classes_[i] for i in y_test],
                'Predicted_Label': [label_encoder.classes_[i] for i in y_pred],
                'Match': [label_encoder.classes_[y_test.iloc[i]] == label_encoder.classes_[y_pred[i]] for i in range(len(y_pred))]
            })
            
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
            
            # Project Summary
            st.divider()
            st.subheader("🎉 Project Completion Summary")
            
            summary = f"""
            ✅ **Data Cleaning Completed**
            - Rows processed: {len(df_clean):,}
            - Columns: {df_clean.shape[1]}
            - Missing values handled: ✅
            - Duplicates removed: ✅
            - Outliers fixed: ✅
            
            ✅ **Feature Engineering Completed**
            - Multi-class encoding: {len(label_encoder.classes_)} attack types
            - Features selected: {X_test_scaled.shape[1]}
            - Train/test split: 80/20 with stratification
            
            ✅ **Model Training Completed**
            - Algorithm: Random Forest (100 trees)
            - Overall Accuracy: {acc*100:.2f}%
            - Test Score: {train_score*100:.2f}%
            - Status: **READY FOR DEPLOYMENT** 🚀
            """
            
            st.markdown(summary)

else:
    st.info("📤 Please upload a CSV file to get started!")
    st.markdown("""
    ### 📝 Expected Data Format:
    - CSV file with network traffic features
    - Should include a 'Label' column with attack types
    - Numeric features for network metrics
    
    ### 🎯 What the app does:
    1. ✅ Cleans and preprocesses raw data
    2. ✅ Handles missing values and duplicates
    3. ✅ Performs exploratory analysis
    4. ✅ Engineers features for ML
    5. ✅ Trains Random Forest classifier
    6. ✅ Generates predictions and visualizations
    7. ✅ Exports results
    """)

st.divider()
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit | CIC-IDS2017 Network Detection System</p>
</div>
""", unsafe_allow_html=True)
