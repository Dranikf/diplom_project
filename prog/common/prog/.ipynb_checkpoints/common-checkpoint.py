import numpy as np
from sklearn.metrics import roc_curve

def get_p_bar(Y, probs_hat):
    fpr, tpr, thresholds = roc_curve(Y, probs_hat)

    return thresholds[np.argmax(np.abs(fpr - tpr))]