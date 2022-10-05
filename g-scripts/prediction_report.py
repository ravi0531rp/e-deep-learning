import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report

def prediction(model, data_gen, threshold=0.5):
    Y_preds = model.predict(data_gen)
    y_pred = []
    for x in Y_preds:
        y_pred.append([1 if i>=threshold else 0 for i in x])
    y_pred = np.array(y_pred)
    return y_pred

def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names)

    try:
        sns.set(font_scale=5.0)
        heatmap = sns.heatmap(df_cm, annot=True, fmt=".2f", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title(class_label)
    
def plot_confusion_matrix(test_labels, y_pred, labels):
    vis_arr = multilabel_confusion_matrix(np.array(test_labels), y_pred)

    fig, ax = plt.subplots(2, 7, figsize=(100, 30))

    for axes, cfs_matrix, label in zip(ax.flatten(), vis_arr, labels):
        print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])

    fig.tight_layout()
    plt.show()

    print(classification_report(np.array(test_labels), y_pred))