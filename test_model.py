import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_recall_curve, average_precision_score



def evaluate(predictions, ground_truths, labels=['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']):
    # 扁平化预测标签和真实标签（因为每个病人可能有多个标签）
    flattened_predictions = [item for sublist in predictions for item in sublist]
    flattened_ground_truths = [item for sublist in ground_truths for item in sublist]

    # 计算准确率
    accuracy = accuracy_score(flattened_ground_truths, flattened_predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # 计算精度、召回率和F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(flattened_ground_truths, flattened_predictions,
                                                               average='weighted')
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # 混淆矩阵
    conf_matrix = confusion_matrix(flattened_ground_truths, flattened_predictions, labels=labels)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    print("Confusion Matrix:")
    print(conf_matrix_df)

    # 绘制混淆矩阵热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix Heatmap')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('./static/result/confusion_matrix_heatmap.png')  # 保存图像
    plt.close()

    # ROC曲线和AUC
    lb = LabelBinarizer()
    binarized_truth = lb.fit_transform(flattened_ground_truths)
    binarized_preds = lb.transform(flattened_predictions)

    fpr, tpr, thresholds = roc_curve(binarized_truth.ravel(), binarized_preds.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('./static/result/roc_curve.png')  # 保存图像
    plt.close()

    # PR 曲线和平均精度 (AP)
    precision_vals, recall_vals, _ = precision_recall_curve(binarized_truth.ravel(),
                                                            binarized_preds.ravel())
    avg_precision = average_precision_score(binarized_truth.ravel(),
                                            binarized_preds.ravel())

    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('./static/result/pr_curve.png')  # 保存图像
    plt.close()

    # 返回评估指标和图像路径
    return accuracy, precision, recall, f1, conf_matrix_df
