from fastai.metrics import skm_to_fastai
from sklearn.metrics import confusion_matrix

# Define a sklearn-style function to calculate specificity
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Convert it to a fastai metric
def Fastai_Specificity():
    return skm_to_fastai(specificity_score)


