import tsai
from tsai.all import *
from fastai.callback.tracker import EarlyStoppingCallback

import utils
from utils.data_provider import load_interstate_data, convert_to_L
from utils.data_splitter import train_val_split_indices, _mkdirs_if_not_exist
from utils.constants import RESULTS_DIR

import os
from os.path import join

import models
from models.metrics import Fastai_Specificity
from models.callback import SavePredsCallback

def create_dls_list(index, X, y, tfms, batch_tfms):
    index_L = convert_to_L(index)
    dls = []
    for i, idx in enumerate(index_L):
        dl = get_ts_dls(X, y,
                        splits = index_L[i],
                        tfms = tfms,
                        batch_tfms = batch_tfms,
                        bs = [8,16])
        dls.append(dl)
    return dls

# For now there is no mechanism to pass random_state and paradigm to this function, so this must be 
# done manually

### This section is for CNN-based
def CNN_train_val_loop(archs, dls, epochs, paradigm=None, random_state=None, save_results = False):
    results = pd.DataFrame(columns=['arch', 'train loss', 'valid loss', 'accuracy', 'f1_score', 'auc', 'precision', 'recall', 'specificity' , 'time'])
    for i, (arch, k) in enumerate(archs):
        model = create_model(arch, dls=dls[i], **k)  # Use the ith dataloader
        print(model.__class__.__name__)
        learn = Learner(dls[i], model, metrics=[accuracy, F1Score(), RocAucBinary(), Precision(), Recall(), Fastai_Specificity()],
                        cbs=[EarlyStoppingCallback(monitor='accuracy', min_delta=0.01, patience=3), SavePredsCallback()])  # Use the ith dataloader
        start = time.time()
        lr = learn.lr_find(show_plot=False)
        learn.fit_one_cycle(epochs, lr)  # Reduce number of epochs to 10, as per your last code block
        elapsed = time.time() - start
        vals = learn.recorder.values[-1]
        results.loc[i] = [arch.__name__, vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], int(elapsed)]
        clear_output()
        display(results)
    if save_results:
        _mkdirs_if_not_exist(RESULTS_DIR)
        filename = f'{model.__class__.__name__}_{paradigm}_{epochs}e_{random_state}rs_results.csv'
        filepath = join(RESULTS_DIR, filename)
        results.to_csv(filepath)
  

        
# def _save_result():
#     _mkdirs_if_not_exist(RESULTS_DIR)
#     filename = 

    



    


