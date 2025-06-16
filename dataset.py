import json
import random
from re import sub
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import librosa

def collate_fn(examples):
    encoder_type = examples[0][4]
    feats = [x[0] for x in examples]
    labels = [x[1] for x in examples]
    names = [x[2] for x in examples]
    if encoder_type == "clap":
        padded_feats = feats
    else:
        padded_feats = pad_sequence(feats, batch_first=True, padding_value=0)
    captions = [x[3] for x in examples]
    return {"audios": padded_feats, "text": labels, "names": names, "caps": captions}


def handle_wav(wav_file, target_rate, max_sample_length):
    """
    handle one wav file.
    Return:
        waveform: Tensor(1D)
    """
    waveform, sample_rate = torchaudio.load(wav_file)
    # print(sample_rate)
    if sample_rate != target_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_rate
        )(waveform)

    waveform = waveform[0]  # just get one channel data
    # if audio length is longer than max_length_sample, we randomly crop it to max length
    if waveform.shape[-1] > max_sample_length:
        max_start = waveform.shape[-1] - max_sample_length
        start = random.randint(0, max_start)
        waveform = waveform[start : start + max_sample_length]
    return waveform

def handle_wav_clap(wav_file, target_rate):
    wav_tok = librosa.load(wav_file, sr=16000)[0]
    wav_tok = librosa.resample(wav_tok,orig_sr=16000,target_sr=target_rate)
    return torch.as_tensor(wav_tok)

def _text_preprocess(sentence):
    sentence = sentence.lower()
    sentence = sub(r'\s([,.!?;:"](?:\s|$))', r"\1", sentence).replace("  ", " ")
    sentence = sub('[(,.!?;:|*")]', " ", sentence).replace("  ", " ")
    return sentence


class AudioDataset(Dataset):
    def __init__(self, data_file, sample_rate=16000, max_length=10, rag = False, encoder_type=None):
        super().__init__()
        # self.lists = []
        # with open(data_file, "r", encoding="utf8") as fin:
        #     for line in fin:
        #         self.lists.append(line)

        # self.all_data = []
        # for line in self.lists:
        #     obj = json.loads(line)
        #     self.all_data.append(obj)
        
        self.rag = rag
        self.all_data = json.load(open(data_file))
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.max_length_sample = self.max_length * self.sample_rate
        self.encoder_type = encoder_type
        if self.encoder_type == "clap":
            print("========================Use CLAP dataloader==================")

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        obj = self.all_data[index]
        key = obj["key"]
        wav_file = obj["wav"]

        if str(self.rag).lower() == 'true':
            caps = '\n\n'.join(obj["caps"][0:3])
        else:
            caps = None

        # print(wav_file)
        caption = _text_preprocess(obj["label"])
        if self.encoder_type == "clap":
            waveform = handle_wav_clap(wav_file, target_rate=self.sample_rate)
        else:
            waveform = handle_wav(
                wav_file,
                target_rate=self.sample_rate,
                max_sample_length=self.max_length_sample,
            )
        # print('caps:', caps)
        return waveform, caption, key, caps, self.encoder_type
