import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def visualize_and_evaluate(y_test, prediction, final_score):
    """
    Hiển thị ma trận nhầm lẫn và in báo cáo phân loại.
    """
    # In ra ma trận nhầm lẫn
    print('Confusion Matrix')
    cm = confusion_matrix(y_test, prediction)

    # Tạo nhãn cho ma trận nhầm lẫn
    group_names = ["True Negative", "False Positive", "False Negative", "True Positive"]
    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cm, annot=labels, fmt="", cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # In ma trận nhầm lẫn và báo cáo phân loại
    print(cm)
    print(classification_report(y_test, prediction))

    # In F1-score
    print('Weighted F1 Score:', final_score)