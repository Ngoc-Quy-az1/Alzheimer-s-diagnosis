import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def train_and_predict(model, model_name, X_train, y_train, X_test, y_test, Accuracy):
    # Huấn luyện mô hình
    model.fit(X_train, y_train)
    
    # Dự đoán nhãn của tập X_test
    prediction = model.predict(X_test)
    
    # In ra ma trận nhầm lẫn
    print(f'\nConfusion Matrix for {model_name}:')
    cm = confusion_matrix(y_test, prediction)
    
    group_names = ["True Negative", "False Positive", "False Negative", "True Positive"]
    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten() / np.sum(cm)]
    
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    
    sns.heatmap(cm, annot=labels, fmt="", cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    print(cm)
    
    print(f'\nClassification Report for {model_name}:')
    print(classification_report(y_test, prediction))
    
    weighted_f1 = f1_score(y_test, prediction, average='weighted')
    final_score = weighted_f1 * 100
    
    print(f'{model_name} Weighted F1-Score: {final_score:.2f}')
    
    # Thêm F1-score vào Accuracy theo tên mô hình
    Accuracy[model_name].append(final_score)
