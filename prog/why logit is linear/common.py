from sklearn.metrics import roc_curve
import numpy as np

def get_p_bar(Y, probs_hat):
    fpr, tpr, thresholds = roc_curve(Y, probs_hat)

    # вот он p-штрих как максимальное
    return thresholds[np.argmax(np.abs(fpr - tpr))]