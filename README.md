# Semiconductor Manufacturing Process Feature Selection & Quality Control

## 📋 Problem Statement

This project addresses a critical challenge in semiconductor manufacturing: **detecting production failures using high-dimensional sensor data**. The dataset contains 592 sensor measurements from manufacturing processes, but suffers from:

- **Extreme class imbalance**: Only 6.6% of samples are failures (104 out of 1,567 total samples)
- **High dimensionality**: 592 features create computational complexity and risk of overfitting
- **Quality control imperative**: Missing failures can lead to costly product defects and customer dissatisfaction

## 🎯 Goals

1. **Reduce feature dimensionality** while maintaining model performance
2. **Address class imbalance** to enable failure detection
3. **Develop a reliable quality control system** that can identify manufacturing defects
4. **Balance accuracy with failure detection sensitivity**

## 📊 Dataset Overview

- **Source**: [UCI Semiconductor Manufacturing Dataset (Kaggle)](https://www.kaggle.com/datasets/paresh2047/uci-semcom/data) 
- **Total Samples**: 1,567 manufacturing runs
- **Features**: 592 sensor measurements (after removing time column)
- **Target Variable**: Pass/Fail (-1 = Pass, 1 = Fail)
- **Class Distribution**:
  - Pass: 1,463 samples (93.4%)
  - Fail: 104 samples (6.6%)
  - Imbalance Ratio: 14.1:1

## 🔬 Methodology

### 1. Data Preprocessing
- Loaded dataset and removed non-feature columns (Time)
- Handled missing values (filled with 0)
- Separated features (X) and target (y)
- Train/test split (80/20)

### 2. Feature Selection
- **Method**: Univariate feature selection (SelectKBest with f_classif)
- **Optimization**: Tested k values from 10-100 features
- **Result**: Optimal k = 10 features (98.3% dimensionality reduction)
- **Selected Features**: 21, 26, 28, 59, 103, 348, 431, 434, 435, 510

### 3. Class Imbalance Handling
Tested multiple approaches:
- **Standard Random Forest**: Baseline (predicts all passes)
- **Class Weighting**: Penalizes misclassification of minority class
- **SMOTE Oversampling**: Generates synthetic failure samples
- **Full vs. Selected Features**: Compared performance with all 590 features

### 4. Model Training & Evaluation
- **Algorithm**: Random Forest Classifier (100 estimators)
- **Evaluation Metrics**:
  - Accuracy (overall performance)
  - Recall (failure detection rate)
  - Precision (quality of failure predictions)
  - F1-Score (balanced metric)
  - Confusion Matrix (detailed breakdown)

## 📈 Results & Findings

### Feature Selection Results
```
Optimal Configuration: k = 10 features
- Dimensionality Reduction: 592 → 10 features (98.3% reduction)
- Performance Maintained: Comparable accuracy with massive simplification
```

### Class Imbalance Handling Comparison

| Approach | Features | Accuracy | Fail Recall | Fail Precision | Fail F1 |
|----------|----------|----------|-------------|----------------|---------|
| **Standard** | Top 10 | 92.04% | 0.0% | 0.0% | 0.0% |
| **Class Weighted** | Top 10 | 92.36% | 0.0% | 0.0% | 0.0% |
| **SMOTE (Top 10)** | Top 10 | **85.67%** | **33.3%** | 21.6% | 26.2% |
| **SMOTE (Full)** | 590 | 92.04% | 0.0% | 0.0% | 0.0% |

### Key Performance Insights

#### ✅ **SMOTE + Top 10 Features = WINNER**
- **Failure Detection**: 33.3% (8/24 failures caught) vs 0% for other methods
- **Balanced Performance**: 85.7% accuracy with meaningful failure detection
- **Computational Efficiency**: 98.3% fewer features to process

#### ❌ **Full Feature Set Pitfalls**
- More features introduced noise that prevented failure detection
- Demonstrates "curse of dimensionality" with imbalanced data
- Feature selection was crucial for success

#### 📊 **Confusion Matrix - Best Model (SMOTE + Top 10)**
```
Predicted:     Pass    Fail
Actual: Pass   261     29    ← 261 correct passes, 29 false alarms
        Fail    16      8    ← 16 missed failures, 8 correct detections
```

## 💡 Key Insights

### 1. **Feature Selection is Critical**
- Top 10 features outperformed full 590-feature model
- Strategic selection beats throwing more data at the problem
- SelectKBest identified most discriminative features for failure detection

### 2. **SMOTE Transforms Imbalanced Learning**
- Generated 1,093 synthetic failure samples (from 80 to 1,173)
- Enabled model to learn failure patterns previously invisible
- Balanced training set: 50% pass / 50% fail

### 3. **Business Impact**
- **Before**: Model misses all failures (93.4% accuracy, useless for QC)
- **After**: Catches 33% of failures (85.7% accuracy, actionable for prevention)
- Trade-off of slightly lower accuracy for dramatically better failure detection

### 4. **Quality Control Implications**
- Semiconductor manufacturing requires failure detection over overall accuracy
- Missing failures costs money; false alarms can be investigated
- 33% detection rate provides significant quality improvement

## 🏆 Final Recommendations

### ✅ **Recommended Approach: SMOTE + Top 10 Features**
```python
# Feature Selection
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_train, y_train)

# SMOTE Oversampling
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_selected, y_train)

# Model Training
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_balanced, y_balanced)
```

### 📋 **Selected Features (by importance)**
1. Feature 59 (F-score: 34.68)
2. Feature 103 (F-score: 35.13)
3. Feature 510 (F-score: 27.93)
4. Feature 28 (F-score: 25.35)
5. Feature 431 (F-score: 24.77)
6. Feature 21 (F-score: 20.58)
7. Feature 434 (F-score: 20.84)
8. Feature 26 (F-score: 20.04)
9. Feature 348 (F-score: 19.89)
10. Feature 435 (F-score: 19.69)

### 🔄 **Next Steps for Production**
1. **Model Validation**: Cross-validation on larger datasets
2. **Threshold Tuning**: Adjust decision threshold for specific business needs
3. **Ensemble Methods**: Combine multiple models for better performance
4. **Real-time Deployment**: Implement for live manufacturing monitoring
5. **Cost-sensitive Learning**: Incorporate business costs of false positives/negatives

## 🛠️ Technical Requirements

- **Python 3.8+**
- **Libraries**:
  - pandas, numpy
  - scikit-learn
  - imbalanced-learn (for SMOTE)
  - matplotlib, seaborn
  - kagglehub (for data download)

## 📁 Project Structure
```
semiconductor-mfg-feature-selection/
├── README.md                           # This file
├── semiconductor-process-feature-selection.ipynb  # Complete analysis
└── uci-secom.csv                      # Dataset (downloaded automatically)
```

## 🤝 Contributing

This analysis demonstrates the importance of combining feature selection with imbalance handling for real-world manufacturing quality control. The methodology can be adapted to other high-dimensional, imbalanced classification problems.

---

**Key Takeaway**: Strategic feature selection + SMOTE oversampling achieved what traditional approaches couldn't - turning a failure-blind model into a quality control system that actually catches manufacturing defects! 🎯
