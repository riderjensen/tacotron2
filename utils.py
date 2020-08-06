import numpy as np
from scipy.io.wavfile import read
import torch
import eng_to_ipa as ipa
import re
import epitran
from unidecode import unidecode
from text.cleaners import english_cleaners

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

def convert_to_ipa(texts):
    print("Converting training files to IPA notation...")
    epi = epitran.Epitran('eng-Latn', ligatures = True)
    for text_mel_pair in texts:
        text_mel_pair[1] = ipa.convert(english_cleaners(text_mel_pair[1]))
        foreign_words = re.findall(r"[^ ]{0,}\*", text_mel_pair[1])
        for word in foreign_words:
            text_mel_pair[1] = text_mel_pair[1].replace(word, epi.transliterate(word[0:len(word)-1]))

def convert_to_arpa(texts):
    print("Converting training files to arpabet notation...")
    for text_mel_pair in texts:
        text_mel_pair[1] = arpa_convert(english_cleaners(text_mel_pair[1]), "text/cmudict-0.7b")
    print("Sample Arpabet Sentence: " + texts[0])


# Arpabet Utils

def load_arpadict(cmudict_path):
    # load dictionary as lookup table
    arpadict = {unidecode(line.split()[0]): unidecode(' '.join(line.split()[1:]).strip()) for line in open(cmudict_path, 'r', encoding="latin-1")}
    return arpadict

def get_arpa(text, punc, arpadict):
    """Convert block of text into ARPAbet."""
    out = []
    for word in text.split(" "):
        end_chars = ''; start_chars = ''
        while any(elem in word for elem in punc) and len(word) > 1:
            if word[-1] in punc:
                end_chars = word[-1] + end_chars
                word = word[:-1]
            elif word[0] in punc:
                start_chars = start_chars + word[0]
                word = word[1:]
            else:
                break
        try:
            word = "{" + str(arpadict[word.upper()]) + "}"
        except KeyError:
            pass
        out.append((start_chars + (word or '') + end_chars).rstrip())
    return ' '.join(out)

def arpa_convert(text, cmudict_path):
  punc = "!?,.;:␤#-_'\"()[]\n"
  arpadict = load_arpadict(cmudict_path)
  text = get_arpa(text, punc, arpadict)
  return text