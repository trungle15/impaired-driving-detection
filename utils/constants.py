from os.path import join, dirname, abspath

## DIRS AND FILEPATHS
ROOT = dirname(dirname(abspath(__file__)))

DATA_DIR = join(ROOT, 'data')
METADATA_DIR = join(ROOT, 'metadata')
RESULTS_DIR = join(ROOT, 'results')

INTERSTATE_DIR = join(DATA_DIR, 'full_interstate_not_alcohol' ,'current_samples' ,'60_frames_per_second')
OVERLAP_SAMPLES = join(INTERSTATE_DIR, 'full_interstate_60s_overlap.csv')
NON_OVERLAP_SAMPLES = join(INTERSTATE_DIR, 'full_interstate_60s_non_overlap_with_refs_updated.csv')

## METADATA (TRAIN & VALIDATION ORGANIZATION)
RANDOM_STATE_DIR = join(METADATA_DIR, 'rand_state_')
INDEX_SPLIT_FILENAME = "rand_state_{random_state}_val_size_{val_size}_{suffix}.json"

SPLIT_AGRS_NAMES = {
    "stratified": "s",
    "individual-specific": "a",
    "individual-agnostic": "b"
}

# RESULTS and RAW PREDS
RESULTS_DIR = join(ROOT, 'results')
AGG_TABLE_DIR = join(RESULTS_DIR, 'agg_table')
RAW_PRED_DIR = join(RESULTS_DIR, 'raw_pred')


# MODEL PATH
MODEL_DIR = join(ROOT, 'model')
BESTMODEL_PATH = join(MODEL_DIR, 'bestmodel.pth')