import torch
import h5py
import librosa
import pandas as pd
from tqdm import tqdm 
from transformers import logging
from transformers import AutoFeatureExtractor, ClapAudioModel
import json

logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder_name = "laion/clap-htsat-fused"
feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_name) 
model = ClapAudioModel.from_pretrained(encoder_name).to(device)

def load_data(data_path):

    data = {'train': [], 'val': [], 'test':[]}

    #read json files
    with open(data_path + 'train.json') as f:
        train = json.load(f)
    with open(data_path + 'val.json') as f:
        dev = json.load(f)
    with open(data_path + 'test.json') as f:
        test = json.load(f)
        
    for item in train:
        data['train'].append({'audio_id': item['wav'].split('/')[-1], 'file_name': item['wav']})
    print('train data loaded')

    for item in dev:
        data['val'].append({'audio_id': item['wav'].split('/')[-1], 'file_name': item['wav']})
    print('val data loaded')

    for item in test:
        data['test'].append({'audio_id': item['wav'].split('/')[-1], 'file_name': item['wav']})
    print('test data loaded')

    return data

    

def encode_split(data, split):
    df = pd.DataFrame(data[split])

    bs = 256
    h5py_file = h5py.File(features_dir + '{}.hdf5'.format(split), 'w')

    audio_id_all = list(df['audio_id'])
    file_name_all = list(df['file_name'])

    all_unique_file_ids = []

    for idx in tqdm(range(0, len(df), bs)):
        audio_ids = audio_id_all[idx:idx + bs]
        file_names = file_name_all[idx:idx + bs]
        audio_read = [librosa.resample(librosa.load(i, sr=16000)[0],orig_sr=16000,target_sr=48000) for i in file_names]
        audio_input = feature_extractor(audio_read, sampling_rate=48000, return_tensors="pt").to(device)
        with torch.no_grad():
            encodings = model(input_features=audio_input['input_features'],is_longer=audio_input['is_longer'],output_hidden_states=True)
            encodings = torch.flatten(encodings.last_hidden_state,2)
            encodings = encodings.permute(0,2,1).detach().cpu().numpy()

        for audio_id, encoding in zip(audio_ids, encodings):
            if str(audio_id) not in all_unique_file_ids:
                h5py_file.create_dataset(str(audio_id), (64, 768), data=encoding)
                all_unique_file_ids.append(str(audio_id))


data_dir = 'data/AudioCaps/'
features_dir = 'features/AudioCaps/'

data = load_data(data_dir)

print('Encoding train, val and test splits')
print('------------------------------------')
print('train')
encode_split(data, 'train')
print('val')
encode_split(data, 'val')
print('test')
encode_split(data, 'test')