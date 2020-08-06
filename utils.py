import numpy as np
from scipy.io.wavfile import read
import torch
import eng_to_ipa as ipa
import re
import epitran
from unidecode import unidecode
from text.cleaners import english_cleaners
from g2p_en import G2p

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
        text_mel_pair[1] = make_arpabet(text_mel_pair[1])
    print("Sample Arpabet Sentence: " + str(text_mel_pair[1]))

def make_arpabet(text):
  # g2p functions
  g2p = G2p()

  # Define punctuation, prevent punctuation curlies, and make replacement dictionary to fix spacing
  punc = "!?,.;:␤#-_'\"()[]\n,."
  punc = list(punc)
  punc_key = list(punc)
  punc_alt = [" " + item for item in punc]
  punc_dict = {}
  for key in punc_alt:
    for value in punc_key:
      punc_dict[key] = value
      punc_key.remove(value)
      break
  
  # Text processing
  text = " ".join(g2p(english_cleaners(text))).split("  ")
  outlist = []
  for item in text:
    item = item.strip()
    if item not in punc:
      item =  "{" + item + "}"
    outlist.append(item)
  text =" ".join(outlist)
  for key, replacement in punc_dict.items():
    text = text.replace(key, replacement)
  return text
