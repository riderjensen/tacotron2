# Vars to pass: TT2 path, waveglow path, text (.txt file with newlines, or an array), save location, toggle one/multi-file

import argparse
import os
import matplotlib
import matplotlib.pylab as plt

import sys
sys.path.append('waveglow/')
import numpy as np
import torch
import epitran
import eng_to_ipa as ipa
import re

from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser
from text.cleaners import english_cleaners
from utils import make_arpabet
from hparams import create_hparams
from scipy.io.wavfile import write

def generate_from_file(tacotron2_path, waveglow_path, text_file, output_directory):

  # Make synthesis paths

  if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print("Creating directory " + output_directory + "...")

  hparams = create_hparams()
  hparams.sampling_rate = 22050

  print("Loading models...")
  model = load_model(hparams)
  model.load_state_dict(torch.load(tacotron2_path)['state_dict'])
  _ = model.cuda().eval().half()

  waveglow = torch.load(waveglow_path)['model']
  waveglow.cuda().eval().half()
  for k in waveglow.convinv:
      k.float()
  denoiser = Denoiser(waveglow)

  genlist = []
  with open(text_file) as file:
    for line in file:
      genlist.append(line.strip())

  for entry in genlist:
    wav_name = "_".join(entry.split(" ")[:4]).lower() + ".wav"

    epi = epitran.Epitran('eng-Latn', ligatures = True)
    if hparams.preprocessing == "ipa":
      entry = ipa.convert(english_cleaners(entry))
      foreign_words = re.findall(r"[^ ]{0,}\*", entry)
      for word in foreign_words:
        entry = entry.replace(word, epi.transliterate(word[0:len(word)-1]))
    if hparams.preprocessing == "arpabet":
      entry = make_arpabet(entry)

    # Text sequencer
    if hparams.preprocessing is not None:
      sequence = np.array(text_to_sequence(entry, None))[None, :]
    else:
      sequence = np.array(text_to_sequence(entry, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
      torch.from_numpy(sequence)).cuda().long()

    # Synthesis
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    with torch.no_grad():
      audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    audio_denoised = denoiser(audio, strength=0.01)[:, 0]

    # Save audio
    print ("Saving " + wav_name)
    write(os.path.join(output_directory, wav_name), hparams.sampling_rate, audio_denoised[0].data.cpu().numpy())
    

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('-t', '--tacotron2_path', type=str, default="tacotron2_statedict.pt",
                        required=False, help='Tacotron2 checkpoint to load')
  parser.add_argument('-w', '--waveglow_path', type=str, default="waveglow/waveglow_256channels_universal_v5.pt",
                        required=False, help='waveglow checkpoint to load')
  parser.add_argument('-f', '--text_file', type=str, 
                        help='Text file or list to generate audio from.')
  parser.add_argument('-o', '--output_directory', type=str, default="savedir",
                        required=False, help='Output directory to save to. Defaults to savedir.')
  
  args = parser.parse_args()

  generate_from_file(args.tacotron2_path, args.waveglow_path, args.text_file, args.output_directory)
