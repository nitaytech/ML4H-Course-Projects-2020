import pandas as pd

HEADER_FIELDS = ['record_name', 'n_sig', 'fs', 'counter_freq', 'base_counter', 'sig_len', 'base_time',
                 'base_date', 'comments']
SIGNAL_FIELDS = ['record_name', 'file_name', 'sig_name', 'fmt', 'samps_per_frame', 'skew', 'byte_offset', 'adc_gain',
                 'baseline', 'units', 'adc_res', 'adc_zero', 'init_value', 'checksum', 'block_size']
SIGNAL_DATA_FIELD = 'p_signal'
DX_CODE_DF = pd.read_csv('ecg_lm/dx_codes.csv').rename(columns={'Dx': 'Description', 'SNOMED CT Code': 'Code'})
CODE2ABB = {str(row[1]): row[2] for row in DX_CODE_DF.values}
ABBS = DX_CODE_DF['Abbreviation'].unique().tolist()

PEAK_DIRECTION = {}
PEAK_DIRECTION.update({lead: -1 for lead in ['aVR', 'V1', 'V2', 'V3']})
PEAK_DIRECTION.update({lead: 1 for lead in ['I', 'II', 'III', 'aVL', 'aVF', 'V4', 'V5', 'V6']})
MAX_HR = 220
BASE_FS = 500
SEGMENT_SECS = 0.05
QRS_SECS = 0.15
