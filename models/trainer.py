import tsai
from tsai.all import *

import utils
from utils.data_provider import load_interstate_data, convert_to_L
from utils.data_splitter import train_val_split_indices

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

def CNN_train_val_loop(archs, dls, epochs):
    results = pd.DataFrame(columns=['arch', 'hyperparams', 'total params', 'train loss', 'valid loss', 'accuracy', 'f1_score', 'auc', 'time'])
    for i, (arch, k) in enumerate(archs):
        model = create_model(arch, dls=dls[i], **k)  # Use the ith dataloader
        print(model.__class__.__name__)
        learn = Learner(dls[i], model, metrics=[accuracy, F1Score(), RocAucBinary()])  # Use the ith dataloader
        start = time.time()
        learn.fit_one_cycle(epochs, 1e-3)  # Reduce number of epochs to 10, as per your last code block
        elapsed = time.time() - start
        vals = learn.recorder.values[-1]
        results.loc[i] = [arch.__name__, k, count_parameters(model), vals[0], vals[1], vals[2], vals[3], vals[4], int(elapsed)]
        clear_output()
        display(results)
    return results
    


