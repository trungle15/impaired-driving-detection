from os.path import join, dirname, abspath

## DIRS AND FILEPATHS
DATA_DIR = join(dirname(dirname(abspath(__file__))), 'data')
INTERSTATE_DIR = join(DATA_DIR, 'full_interstate_not_alcohol' ,'current_samples' ,'60_frames_per_second')
OVERLAP_SAMPLES = join(INTERSTATE_DIR, 'full_interstate_60s_overlap.csv')
NON_OVERLAP_SAMPLES = join(INTERSTATE_DIR, 'full_interstate_60s_non_overlap.csv')

print(INTERSTATE_DIR)
