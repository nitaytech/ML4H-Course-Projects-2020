import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import torch.utils.data as data
from scipy.io import loadmat
from scipy import signal, stats
from torch import from_numpy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
from wfdb import processing


LABELS = 'AF I-AVB LBBB Normal PAC PVC RBBB STD STE'.split(' ')
CODES = '164889003 270492004 164909002 426783006 284470004 164884008 59118001 429622005 164931005'.split(' ')
LABELS_TO_INDEX = {x: i for i,x in enumerate(LABELS)}
LABELS_TO_INDEX.update({x: i for i,x in enumerate(CODES)})

WINDOW_SIZE = 800
N_FRAMES = 10   
FRAME_FILL = 'append'

def split_data(train_size=0.8):
    shutil.rmtree('./train', ignore_errors=True)
    shutil.rmtree('./val', ignore_errors=True)
    os.mkdir('train')
    os.mkdir('val')
    #rand = np.random.RandomState(100)
    all_files = list(os.listdir('in'))
    no_duplicates = sorted(set([x[:-4] for x in all_files]))
    train, test = train_test_split(no_duplicates, train_size=train_size, random_state=100)
    for f in train:
        shutil.copyfile('in/'+f + '.hea', 'train/'+f+ '.hea')
        shutil.copyfile('in/'+f + '.mat', 'train/'+f+ '.mat')
    for f in test:
        shutil.copyfile('in/'+f + '.hea', 'val/'+f+ '.hea')
        shutil.copyfile('in/'+f + '.mat', 'val/'+f+ '.mat')


def create_frames(input_directory):
    input_files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith(
                'mat'):
            input_files.append(f)

    locs = []
    for i, f in enumerate(input_files):
        if not i % 10:
            print('Loading    {}/{}...'.format(i + 1, len(input_files)))
        tmp_input_file = os.path.join(input_directory, f)
        c_data, header_data = load_challenge_data(tmp_input_file)
        shift = 0
        if c_data.shape[1] > 6000:
            c_data = c_data[:, 500:-500]
            shift = 500
        result = []
        for i in range(0, 12):
            qrs_inds = processing.xqrs_detect(sig=c_data[i], fs=500, verbose=False)
            result.append(qrs_inds)
        lengths = [len(x) for x in result]
        if not all([x == lengths[0] for x in lengths]):
            median = stats.mode([x for x in lengths if x > 6])[0]
            result = [x for x in result if len(x) == median]
        arr = np.array(result)
        final = np.median(arr, axis=0).astype(int)
        if type(final) == np.int32 or len(final) < 4:
            print(f)
        frame_locations = []
        for i in range(2, len(final) - 1):
            left = int((2 * final[i - 1] + final[i]) / 3) + shift
            right = int((2 * final[i + 1] + final[i]) / 3) + shift
            if right - left > WINDOW_SIZE:
                diff = int((right - left - WINDOW_SIZE) / 2)
                left = left + diff
                right = left + WINDOW_SIZE
            frame_locations.append((left, right))

        if len(frame_locations) < N_FRAMES and FRAME_FILL == 'append':
            for i in range(1, N_FRAMES - len(frame_locations) + 1):
                frame_locations.append(frame_locations[-i])

        if len(frame_locations) > N_FRAMES:
            diff = int((len(frame_locations) - N_FRAMES) / 2)
            frame_locations = frame_locations[diff:diff + N_FRAMES]
        locs.append(frame_locations)
    backup = input_directory + '_locations.pkl'
    with open(backup, 'wb') as f:
        pickle.dump(locs, f)


def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.int16)

    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file, 'r') as f:
        header_data = f.readlines()

    return data, header_data


class ECGDataset(data.Dataset):
    def __init__(self, settings, train=True):
        self.train = train
        self.window_size = WINDOW_SIZE
        self.frame_fill = FRAME_FILL
        self.single_lead = settings['single_lead']
        self.n_frames = N_FRAMES
        self.use_frames = settings['use_frames']
        input_files = []
        input_directory = settings['train_folder'] if train else settings['val_folder']
        backup = input_directory.split('_')[0] + '_locations.pkl'
        for f in os.listdir(input_directory):
            if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
                input_files.append(f)
        self.data = []
        self.labels = []
        self.ids = []
        self.genders = []
        self.ages = []
        self.all_data = []
        self.frames = []
        if self.use_frames:
            if os.path.exists(backup):
                with open(backup, 'rb') as f:
                    frame_locations = pickle.load(f)
            else:
                print('Please create frames')
                exit(0)

        for i, f in enumerate(input_files):
            if not i % 500:
                print('Loading    {}/{}...'.format(i + 1, len(input_files)))
            tmp_input_file = os.path.join(input_directory, f)
            c_data, header_data = load_challenge_data(tmp_input_file)
            try:
                labels = header_data[15][5:-1].split(',')
                indices = [LABELS_TO_INDEX[x] for x in labels]
            except KeyError:
                #os.remove(tmp_input_file)
                #os.remove(tmp_input_file[:-4] + '.hea')
                continue
            self.ids.append(f)
            self.genders.append(header_data[14].split(' ')[1])
            self.ages.append(header_data[13].split(' ')[1])
            labels = np.zeros(9)
            for i in indices:
                labels[i] = 1
            self.labels.append(from_numpy(labels.astype(bool)))

            if settings['cutoff_low']:
                sos = signal.butter(4, settings['cutoff_low'], 'hp', fs=500, output='sos')
                c_data = signal.sosfilt(sos, c_data)
            if settings['cutoff_high']:
                sos = signal.butter(4, settings['cutoff_high'], 'lp', fs=500, output='sos')
                c_data = signal.sosfilt(sos, c_data)
            if settings['normalize']:
                c_data = MinMaxScaler().fit_transform(c_data)
                c_data = StandardScaler().fit_transform(c_data)
            #self.all_data.append(c_data)
            if self.use_frames:
                res = np.zeros((12, self.n_frames, self.window_size))
                i = 0
                for left, right in frame_locations[i]:
                    frame = c_data[:, left:right]
                    start_index = int((self.window_size - frame.shape[1]) / 2) if frame.shape[1] < self.window_size else 0
                    res[:, i, start_index:start_index + frame.shape[1]] = frame
                    i += 1
                for j in range(0, self.n_frames - i):
                    res[:, :, i + j] = res[:, :, i - j - 1]
                res = np.array([np.mean(res[:, :, i:i+settings['downsample']], axis=2) for i in range(0, WINDOW_SIZE, settings['downsample'])]).T
                res = np.swapaxes(res, 0, 1)
                self.data.append(res)
            else:
                if c_data.shape[1] < 5000:
                    new_data = np.zeros((12, 5000))
                    new_data[:, :c_data.shape[1]] = c_data
                    c_data = new_data

                # TODO: consider other method instead of selecting each i'th value. E.g. average of neighbors
                #c_data = c_data[:, range(0, 5000, settings['downsample'])]
                c_data = np.vstack([np.mean(c_data[:, i:i+settings['downsample']], axis=1) for i in range(0, 5000, settings['downsample'])]).T
                self.data.append(from_numpy(c_data))

    def __getitem__(self, item):
        result = {'ecg': self.data[item][1] if self.single_lead else self.data[item], 'target': self.labels[item],
                  'id': self.ids[item], 'gender': self.genders[item], 'age': self.ages[item]}
        return result

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    #split_data()
    create_frames('train')
    create_frames('val')