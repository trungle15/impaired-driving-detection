from fastai.callback.core import Callback
import pandas as pd
from tsai.all import *

class SavePredsCallback(Callback):
    def after_epoch(self):
        preds, targs = self.learn.get_X_preds(X=self.dls.valid.items)
        pred_probs, pred_classes = preds.max(dim=1)
        pred_df = pd.DataFrame({'sample_index': self.dls.valid.items, 'model_id': self.learn.model_idx, 'pred_proba': pred_probs, 'pred_class': pred_classes})
        pred_df.to_csv(f'{self.learn.model.__class__.__name__}_{self.learn.paradigm}_{self.learn.epochs}e_{self.learn.random_state}rs_preds_{self.learn.epoch}.csv', index=False)

# Doesn't work
