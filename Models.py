import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


def prepare_data(input_file: str, test_size=0.2, random_state=42):
    """
    - Reads the numerical CSV file
    - Drops the first column
    - Separates features (X) and target (y)
    - Splits the data into train and test sets (80% train, 20% test)
    """
    
    # Load the dataset
    #input_file="C:/Users/user/Intelligent Monitoring System/preprocess_dataset.csv"
    df = pd.read_csv(input_file)
    print(f"Original dataset shape: {df.shape}")
    
    # Step 1: Drop the first column
    df = df.drop(df.columns[0], axis=1)
    print(f"Shape after dropping first column: {df.shape}")
    
    # Step 2: Separate features (X) and target (y)
    y = df.iloc[:, -1]          # Last column as target
    X = df.iloc[:, :-1]         # All columns except the last one as features
    
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    
    # Step 3: Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size,      # 20% test
        random_state=random_state # For reproducibility
    )
    
    print("\n✅ Data Split Completed!")
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Testing set : {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test,y


if __name__ == "__main__":
    print("=== Dataset Preparation & Train-Test Split ===\n")
    
    # Change this to your actual file name
    #input_csv = input("Enter the path to your numerical CSV file: ").strip()
    input_csv = "C:/Users/cherr/OneDrive/Desktop/PROJECT INST/clean_data"
    
    if not input_csv:
        input_csv = "clean_data.csv"   # default name from previous script
    
    X_train, X_test, y_train, y_test,y = prepare_data(input_csv)
    target_names = [str(name) for name in sorted(pd.unique(y_train))]
    
    '''
    # Optional: Save the split datasets
    save_option = input("\nDo you want to save the split datasets? (y/n): ").strip().lower()
    if save_option == 'y':
        X_train.to_csv('X_train.csv', index=False)
        X_test.to_csv('X_test.csv', index=False)
        y_train.to_csv('y_train.csv', index=False)
        y_test.to_csv('y_test.csv', index=False)
        print("Split datasets saved successfully!")
    '''
    
#Support Vector Machine (SVM)
# Python Implementation of SVM (Support Vector Machine)
# with Confusion Matrix and ROC-AUC Curve
# Using scikit-learn and matplotlib/seaborn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    roc_curve, 
    auc, 
    accuracy_score
)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")



# Feature scaling (very important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# =============================================
# 3. Train SVM Model
# =============================================
# Using RBF kernel (most common for non-linear data)
svm_model = SVC(
    kernel='rbf',      # 'linear', 'poly', 'rbf', 'sigmoid'
    C=1.0,             # Regularization parameter
    gamma='scale',     # Kernel coefficient
    probability=True,  # Required for ROC curve
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = svm_model.predict(X_test_scaled)
y_pred_proba = svm_model.predict_proba(X_test_scaled)[:, 1]  # Probability for positive class

# =============================================
# 4. Model Evaluation
# =============================================
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# =============================================
# 5. Confusion Matrix
# =============================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, 
            yticklabels=target_names)
plt.title('SVM Confusion Matrix', fontsize=16)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()



# Random Forest
rf = RandomForestClassifier(n_estimators=80)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest Accuracy:")
print(accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
print("XGBoost Accuracy:")
print(accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))


import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd

# Assuming your model is already trained as 'model'
# Example: model = xgb.XGBClassifier() or XGBRegressor()
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# ============================
# 1. Plot Feature Importance
# ============================

plt.figure(figsize=(10, 8))

# Method 1: Built-in plot (Most Popular)
xgb.plot_importance(model, 
                    importance_type='gain',   # Options: 'weight', 'gain', 'cover'
                    max_num_features=20,      # Show top 20 features
                    height=0.8,
                    show_values=True)

plt.title('XGBoost Feature Importance (Gain)', fontsize=14)
plt.xlabel('Importance Score')
plt.tight_layout()

# Save the plot
plt.savefig('xgboost_feature_importance_gain.png', dpi=300, bbox_inches='tight')
plt.show()


'''
def plot_multiclass_roc(y_test, y_pred_proba, class_names=None):
    """Robust ROC Curve for Binary and Multiclass"""
    
    y_test = np.array(y_test)
    y_pred_proba = np.array(y_pred_proba)
    
    n_classes = len(np.unique(y_test))
    
    plt.figure(figsize=(12, 8))
    
    if n_classes == 2:
        # Binary Classification
        print("✅ Binary Classification Detected")
        if y_pred_proba.ndim == 1:
            prob_positive = y_pred_proba
        else:
            prob_positive = y_pred_proba[:, 1]
            
        fpr, tpr, _ = roc_curve(y_test, prob_positive)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2.5, 
                 label=f'ROC curve (AUC = {roc_auc:.4f})')
        
    else:
        # Multiclass Classification
        print(f"✅ Multiclass Classification Detected ({n_classes} classes)")
        
        if y_pred_proba.ndim == 1:
            raise ValueError("For multiclass, you must use model.predict_proba(X_test) "
                           "which should return a 2D array of shape (n_samples, n_classes)")
        
        if y_pred_proba.shape[1] != n_classes:
            raise ValueError(f"Probability shape mismatch: Expected {n_classes} columns, "
                           f"got {y_pred_proba.shape[1]}")
        
        y_test_bin = label_binarize(y_test, classes=np.sort(np.unique(y_test)))
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            label = class_names[i] if class_names is not None else f'Class {i}'
            plt.plot(fpr, tpr, color=colors[i], lw=2, 
                     label=f'{label} (AUC = {roc_auc:.4f})')
        
        # Micro-average
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
        roc_auc_micro = auc(fpr["micro"], tpr["micro"])
        plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
                 label=f'Micro-average (AUC = {roc_auc_micro:.4f})')
    
    # Diagonal reference line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.50)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.show()
        
plot_multiclass_roc(y_test, y_pred_proba,class_names= target_names)
'''

'''
# ========================== HOW TO USE ==========================

# After training your model and getting predictions:
# y_pred_proba = model.predict_proba(X_test)   # Important: probabilities, not predictions

# Example usage:
# class_names = ['Normal', 'Fraud', 'Suspicious', 'High Risk']   # Replace with your actual class names

plot_multiclass_roc(y_test, y_pred_proba, class_names=None)
''' 













'''
# =============================================
# 6. ROC-AUC Curve
# =============================================
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# =============================================
# Optional: Hyperparameter Tuning Example (GridSearch)
# =============================================
# from sklearn.model_selection import GridSearchCV
#
# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'gamma': [1, 0.1, 0.01, 0.001],
#     'kernel': ['rbf']
# }
# 
# grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, cv=5)
# grid.fit(X_train_scaled, y_train)
# print("Best parameters:", grid.best_params_)
'''
