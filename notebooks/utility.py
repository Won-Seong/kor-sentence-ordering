try:
  import unsloth
  from unsloth import FastLanguageModel
  from trl import SFTTrainer, SFTConfig
  from trl.trainer import DataCollatorForCompletionOnlyLM
except Exception as e:
  print("Cannot find some libraries for training.")

import os
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import gc
import pickle
import torch
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel

from huggingface_hub import login
from langchain_core.prompts import PromptTemplate
from datasets import Dataset
from sklearn.model_selection import train_test_split

if os.getenv('HF_TOKEN') is not None:
  print("HF_TOKEN found. Log in to Hugging Face.")
  HF_TOKEN = os.environ('HF_TOKEN')
  login(HF_TOKEN)

def load_config(config_path):
  with open(config_path, "r") as file:
    config = yaml.safe_load(file)
  return config

def set_all_seed(seed):
  set_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  print("Set seed " + str(seed))

def make_df(sentences_list, answers_list=None):
  assert len(sentences_list) == 4, "The list must contain exactly 4 elements."
  if answers_list is not None:
    assert len(answers_list) == 4, "The list must contain exactly 4 elements."

  tmp_dict = {}
  for i in range(4):
    tmp_dict[f'sentence_{i}'] = sentences_list[i]

  if answers_list is not None:
    for i in range(4):
      tmp_dict[f'answer_{i}'] = answers_list[i]

  return pd.DataFrame(tmp_dict)

def contains_wrong_tokens(sentences):
    regex = r'[^가-힣A-Za-z0-9\s!@#$%^&*()-_+=\[\]{}|;:\'",.<>\/?`~·℃仁]'

    if re.search(regex, sentences):
        return True
    else:
        return False

def contains_non_four_sentences(sentences):
  dot_count = sentences.count('.')
  if dot_count != 4:
    return True
  else:
    return False

def make_error_idx(complete_text_list):
  error_idx = []
  for i, s in enumerate(complete_text_list):
    if contains_wrong_tokens(s):
      print(s)
      error_idx.append(i)
    elif contains_non_four_sentences(s):
      print(s)
      error_idx.append(i)
  print("Number of Error Sentences:", len(error_idx))
  return error_idx

def make_df_from_complete_text_list(complete_text_list, error_idx):
  idx_ptr = 0
  sentences = [[] for _ in range(4)]

  for i, s in tqdm(enumerate(complete_text_list)):
    if idx_ptr < len(error_idx) and i == error_idx[idx_ptr]:
      idx_ptr += 1
      print("Skip this sentence:", s)
      continue

    split_sentence = s.split('. ')
    for i in range(3):
      split_sentence[i] += '.'

    for i in range(4):
      split_sentence[i] = split_sentence[i].replace('\n', '')
      sentences[i].append(split_sentence[i])
  return make_df(sentences)

def convert_row_to_sentence(row):
  assert 'sentence_0' in row and 'answer_0' in row
  sentence = ''

  for i in range(4):
    sentence += row.iloc[ row.loc[f'answer_{i}'] ] + ' '
  return sentence[:-1]

def formatting_prompts(row, chat_template, with_answers=False):
  if with_answers == True:
    assert len(row) >= 8

  text = chat_template.format(
    **dict(row)
  )
  return text