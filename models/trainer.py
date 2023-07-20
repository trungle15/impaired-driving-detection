import tsai
from tsai.all import *
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback

import utils
from utils.data_provider import load_interstate_data, convert_to_L
from utils.data_splitter import train_val_split_indices, _mkdirs_if_not_exist
from utils.constants import RESULTS_DIR, BESTMODEL_PATH, AGG_TABLE_DIR, RAW_PRED_DIR

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

### This section is for CNN-based classifier
def CNN_train_val_loop(archs, dls, epochs, paradigm=None, random_state=None, save_results = False, save_raw_preds = False, wd = None):
    results = pd.DataFrame(columns=['arch', 'valid loss', 'accuracy', 'f1_score', 'auc', 'precision', 'recall', 'specificity' , 'time'])
    _mkdirs_if_not_exist(RESULTS_DIR)
    
    raw_results = pd.DataFrame(columns=['loop_no', 'split_idx', 'pos_class_proba', 'target'])
    
    for i, (arch, k) in enumerate(archs):
        
        # Model definition
        model = create_model(arch, dls=dls[i], **k)  
        print(model.__class__.__name__)
        learn = Learner(dls[i], model, metrics=[accuracy, F1Score(), RocAucBinary(), Precision(), Recall(), Fastai_Specificity()], wd = wd,
                        cbs=[EarlyStoppingCallback(monitor='valid_loss', min_delta=0, patience=50),
                             SaveModelCallback(monitor='valid_loss', comp=np.less, fname='bestmodel')])  # Use the ith dataloader
        
        # Model training
        start = time.time()
        lr = learn.lr_find(show_plot=False)
        learn.fit_one_cycle(epochs, lr)
        elapsed = time.time() - start
        
        # Load best model:
        learn.load('bestmodel')
        probas, targets, preds = learn.get_preds(with_decoded=True)
        vals = _get_vals_from_best_model(learn, probas, targets, preds)
        
        # Results
        results.loc[i] = [arch.__name__, vals[0], vals[1], vals[2], vals[3], 
                          vals[4], vals[5], vals[6], int(elapsed)]
        clear_output()
        display(results)
        
        # Delete bestmodel in this loop since we don't need this
        _delete_model_path(BESTMODEL_PATH)
        
        # Save raw preds:
        if save_raw_preds:
            _mkdirs_if_not_exist(RAW_PRED_DIR)
            raw_pred_i = _save_raw_preds(loop_no=i, split_idxs= learn.dls.valid.dataset.split_idxs, 
                                         proba = probas, targets = targets, preds = preds)
            raw_results = raw_results.append(raw_pred_i, ignore_index=True)
    if save_results: 
        _mkdirs_if_not_exist(AGG_TABLE_DIR)
        _save_result_table(results, arch.__name__, paradigm, epochs, random_state)
    # Delete raw results if save_raw_preds flag is not on, else write out
    if raw_results.empty:
        del raw_results
    else: 
        raw_results.to_csv(join(RAW_PRED_DIR, f'{model.__class__.__name__}_{paradigm}_{epochs}e_{random_state}rs_results.csv'))
    return results

def _save_result_table(results, arch, paradigm, epochs, random_state):
    filename = f'{arch}_{paradigm}_{epochs}e_{random_state}rs_results.csv'
    filepath = join(AGG_TABLE_DIR, filename)
    results.to_csv(filepath)
    
def _save_raw_preds(loop_no,split_idxs, proba, targets, preds):
    val_size = len(split_idxs)
    pred_dict = {'loop_no': [loop_no]*val_size, 'split_idx': split_idxs, 'pos_class_proba': proba[:,1].tolist(),
                 'target': targets, 'pred': preds}
    pred_table = pd.DataFrame(pred_dict)
    return pred_table
    # ### add filepath
    # pred_table.to_csv()
    
def _get_vals_from_best_model(learn, preds_probas, targets, preds_class):
    best_vals = [learn.loss_func(preds_probas, targets).item(),
             accuracy(preds_probas, targets).item(),
             F1Score()(preds_class, targets).item(),
             RocAucBinary()(preds_probas[:,1], targets).item(),
             Precision()(preds_class, targets).item(),
             Recall()(preds_class, targets).item(),
             Fastai_Specificity()(preds_class, targets).item()]
    return best_vals
    
def _delete_model_path(filepath):
    if os.path.isfile(filepath):
        os.remove(filepath)
    else:
        print(f"Error: {filepath} not found" )
        
        
def get_X_minirocket_features_list(X, split_idxs):
    X_feat_list = []
    for i, split in enumerate(split_idxs):
        mrf = MiniRocketFeatures(X.shape[1], X.shape[2]).to(default_device())
        X_train = X[split_idxs[i][0]]
        mrf.fit(X_train)
        X_feat = get_minirocket_features(X, mrf, chunksize=1024, to_np=True)
        X_feat_list.append(X_feat)
    X_feat_array = np.stack(X_feat_list, axis = 0)
    return X_feat_array


def create_mr_dls_list(index, X_feat_array, y, tfms, batch_tfms, bs):
    index_L = convert_to_L(index)
    dls = []
    
    for i, idx in enumerate(index_L):
        dl = get_ts_dls(X_feat_array[i], y,
                        splits = index_L[i],
                        tfms = tfms,
                        batch_tfms = batch_tfms,
                        bs = bs)
        dls.append(dl)
    return dls



def mr_hyperparam_search(archs, dls, epochs, wd_list):
    results = []
    for wd in wd_list:
        print(f"Weight Decay: {wd}")
        result = CNN_train_val_loop(archs, dls, epochs, wd)
        avg_valid_loss = result['valid loss'].mean()
        results.append({
            'weight_decay': wd,
            'avg_valid_loss': avg_valid_loss
        })
    return results
        