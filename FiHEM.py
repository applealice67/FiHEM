# -*- coding: utf-8 -*-
import logging
import os
import json
import re
import random
import csv
import math
from typing import Dict, List, Tuple, Iterable
from collections import Counter

from torch.nn import GELU

import heapq
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import RobertaModel, RobertaTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from efficient_kan import KAN as EfficientKAN

import numpy as np

torch.set_float32_matmul_precision('high')
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def dual_output(message):
    print(message)
    logger.info(message)


POOL_MODE = "cls"


def sentence_pool(last_hidden: torch.Tensor, attn: torch.Tensor, mode: str = POOL_MODE) -> torch.Tensor:
    if mode == "cls":
        return last_hidden[:, 0, :]
    elif mode == "mean":
        mask = attn.unsqueeze(-1)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        return summed / denom
    else:
        raise ValueError(f"Unknown POOL_MODE: {mode}")


stop_words = set(
    ['such', "you'd", 'y', 't', 'down', 'i', 'by', 'whom', 'most', 'his', 'does', 'are', 'between', 're', 'isn', 'only',
     'she', 'of', 'had', 'through', 'other', 'needn', 'be', 'below', 'should', 'when', 'on', 'for', "don't", 'until',
     'can', 'to', 'a', 'from', 'has', "you'll", 'few', 'were', "that'll", 'while', 'just', "she's", "didn't", 'again',
     'under', 'him', 'these', 'your', 'this', 'that', 'being', 'doing', 'all', 'with', "haven't", 'didn', 'nor', 'they',
     'where', 'our', 'them', 'couldn', 'm', "needn't", 'me', 'you', 'we', 'than', "wouldn't", "shan't", 'ma', 'won',
     'yourselves', 'wouldn', 'haven', "it's", 'against', 'ain', 'have', 's', 'any', 'do', 'himself', 'there', 'what',
     'myself', 'both', 've', 'up', 'mustn', 'or', 'wasn', 'into', 'which', "shouldn't", 'hadn', 'as', 'own', 'o',
     'mightn', 'an', 'don', 'her', 'weren', 'itself', 'those', 'how', 'hers', "mightn't", 'is', 'was', "wasn't",
     'before', 'if', 'it', 'will', 'once', 'did', 'same', "hadn't", 'now', 'll', 'no', 'shan', "you're", 'too', 'aren',
     'he', 'some', 'my', 'over', "doesn't", 'shouldn', "isn't", 'ourselves', 'd', 'am', 'themselves', "aren't", 'off',
     'having', 'in', "hasn't", 'further', "mustn't", 'yourself', 'ours', 'theirs', 'here', 'more', 'so', "won't",
     'very', "should've", 'out', 'the', 'and', 'who', 'their', 'but', "couldn't", 'hasn', 'doesn', 'not', 'above',
     'because', 'about', 'its', 'during', "weren't", 'herself', 'been', 'yours', "you've", 'why', 'after', 'then',
     'each', 'at'])


def noStopwords(line):
    line = re.sub(r'(?<!\d)\.(?!\d)', ' ', line)
    line = re.sub(r'[^\w\s\.]', ' ', line)
    line = re.sub(r'\s+', ' ', line).strip()
    tokens = line.lower().split()
    filtered_tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(filtered_tokens)


def parse_json_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"解析行时出错（跳过）: {line}\n错误: {e}")
    return data


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _price_closeness(p1, p2):
    p1 = _to_float(p1);
    p2 = _to_float(p2)
    if p1 is None or p2 is None or p1 <= 0 or p2 <= 0:
        return 0.0
    diff = abs(p1 - p2)
    return max(0.0, 1.0 - diff / max(p1, p2))


def simple_tokens(text: str) -> List[str]:
    return [t for t in text.split() if t]


def tokenize(text):
    return re.findall(r'\w+', text)


def compare_texts(text1, text2):
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    set1, set2 = set(tokens1), set(tokens2)
    common, diff1, diff2 = [], [], []
    for w in tokens1:
        if w in set2 and w not in common:
            common.append(w)
        elif w not in set2:
            diff1.append(w)
    for w in tokens2:
        if w not in set1:
            diff2.append(w)
    return ' '.join(common), ' '.join(diff1), ' '.join(diff2)


def build_string(title, price, currency, brand, des):
    parts = []
    if title:
        parts.append(f"title {title}")
    if price:
        parts.append(f"price {price}")
    if currency:
        parts.append(f"currency {currency}")
    if brand:
        parts.append(f"brand {brand}")
    if des:
        parts.append(f"description {des}")
    return " ".join(parts)


def _process_data_entry(entry):
    left_title = (entry.get('title_left', '') or '').strip()
    right_title = (entry.get('title_right', '') or '').strip()
    left_brand = (entry.get('brand_left', '') or '').strip()
    right_brand = (entry.get('brand_right', '') or '').strip()
    left_price = (entry.get('price_left', '') or '').strip()
    right_price = (entry.get('price_right', '') or '').strip()
    left_currency = (entry.get('priceCurrency_left', '') or '').strip()
    right_currency = (entry.get('priceCurrency_right', '') or '').strip()
    left_des = (entry.get('description_left', '') or '').strip()
    right_des = (entry.get('description_right', '') or '').strip()
    left_des = noStopwords(left_des)
    right_des = noStopwords(right_des)

    s1 = build_string(left_title, left_price, left_currency, left_brand, left_des)
    s2 = build_string(right_title, right_price, right_currency, right_brand, right_des)

    return [(s1, s2),
            (left_title, right_title),
            int(entry.get('label', 0))]


def _load_and_process_data(path, file):
    file_path = f'./{path}/{file}/{file}.json'
    json_data = parse_json_file(file_path)
    processed_data = []
    for entry in json_data:
        processed_entry = _process_data_entry(entry)
        processed_data.append(processed_entry)
    return processed_data


def random_swap(sentence, keep_original_prob=0.1):
    if not sentence.strip() or random.random() < keep_original_prob:
        return sentence
    words = sentence.split()
    if len(words) < 2:
        return sentence
    new_words = words.copy()
    i, j = random.sample(range(len(new_words)), 2)
    new_words[i], new_words[j] = new_words[j], new_words[i]
    return ' '.join(new_words)


from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=64):
        assert batch_size % 2 == 0, "batch_size 必须是偶数！"
        self.dataset = dataset
        self.batch_size = batch_size
        self.half = batch_size // 2

        self.pos_idx = [i for i in range(len(dataset)) if dataset[i]['labels'].item() == 1]
        self.neg_idx = [i for i in range(len(dataset)) if dataset[i]['labels'].item() == 0]

        self.num_batches = max(len(self.pos_idx), len(self.neg_idx)) // self.half

    def __iter__(self):
        pos = self.pos_idx[:]
        neg = self.neg_idx[:]
        random.shuffle(pos)
        random.shuffle(neg)

        batches = []
        for i in range(self.num_batches):
            if len(pos) < (i + 1) * self.half:
                extra = random.choices(self.pos_idx, k=self.half)
                p_batch = pos[i * self.half:] + extra[:(self.half - len(pos[i * self.half:]))]
            else:
                p_batch = pos[i * self.half:(i + 1) * self.half]

            if len(neg) < (i + 1) * self.half:
                extra = random.choices(self.neg_idx, k=self.half)
                n_batch = neg[i * self.half:] + extra[:(self.half - len(neg[i * self.half:]))]
            else:
                n_batch = neg[i * self.half:(i + 1) * self.half]

            batch = p_batch + n_batch
            random.shuffle(batch)
            batches.append(batch)

        random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return self.num_batches


def data_augment(list_data):
    result_data, t_data, f_data = [], [], []
    for row in list_data:
        if int(row[-1]):
            t_data.append(row)
        else:
            f_data.append(row)
    len_t, len_f = len(t_data), len(f_data)
    len_max = max(len_t, len_f)
    dual_output(f"Original t_len: {len_t}, f_len: {len_f}")
    for i in range(len_max):
        result_data.append(t_data[i % len_t])
        result_data.append(f_data[i % len_f])
    dual_output(result_data[:3])
    return result_data


def getData(path, trainpath, testpath):
    origtrain_data = _load_and_process_data(path, trainpath)
    augtrain_data = data_augment(origtrain_data)
    test_data = _load_and_process_data(path, testpath)
    dual_output(f"origtrain data size: {len(origtrain_data)}")
    dual_output(f"augtrain data size: {len(augtrain_data)}")
    dual_output(f"test data size: {len(test_data)}")
    pairnums = len(origtrain_data[0]) - 1 if origtrain_data else 0
    return origtrain_data, augtrain_data, test_data, pairnums


def build_class_idf_from_pairs(pairs_data,
                               deduplicate: bool = False,
                               k: float = 1.0,
                               beta_pos: float = 1.0,
                               beta_neg: float = 1.0):
    from collections import Counter, defaultdict
    import math, re

    def _simple_tokens(text: str):
        return [t for t in text.lower().split() if t]

    if deduplicate:
        seen = set()
        unique_rows = []
        for row in pairs_data:
            s1, s2 = row[0]
            label = int(row[-1])
            key = (s1.strip(), s2.strip(), label)
            if key not in seen:
                seen.add(key)
                unique_rows.append(row)
        data_iter = unique_rows
    else:
        data_iter = pairs_data

    tf_pos, tf_neg = Counter(), Counter()
    df_pos, df_neg = Counter(), Counter()
    N_pos = N_neg = 0

    for row in data_iter:
        s1, s2 = row[0]
        label = int(row[-1])
        toks1 = set(_simple_tokens(s1))
        toks2 = set(_simple_tokens(s2))
        co_tokens = toks1 & toks2
        if not co_tokens:
            continue

        if label == 1:
            tf_pos.update(co_tokens)
            df_pos.update(co_tokens)
            N_pos += 1
        else:
            tf_neg.update(co_tokens)
            df_neg.update(co_tokens)
            N_neg += 1

    def _tfidf(tf_counter, df_counter, N):
        return {
            t: (tf_counter[t] / (df_counter[t] + k)) * (math.log((N + k) / (df_counter[t] + k)) + 1.0)
            for t in df_counter
        }

    idf_pos = {t: beta_pos * score for t, score in _tfidf(tf_pos, df_pos, N_pos).items()}
    idf_neg = {t: beta_neg * score for t, score in _tfidf(tf_neg, df_neg, N_neg).items()}

    return idf_pos, idf_neg, N_pos, N_neg


def build_global_idf_from_pairs(pairs_data):
    df_counter = Counter()
    tf_counter = Counter()
    N_docs = 0

    def _simple_tokens(text: str):
        return [t for t in text.split() if t]

    for row in pairs_data:
        s1, s2 = row[0]
        for doc_text in (s1, s2):
            toks = _simple_tokens(doc_text)
            if not toks:
                continue
            toks_set = set(toks)
            tf_counter.update(toks_set)
            df_counter.update(toks_set)
            N_docs += 1

    if N_docs == 0:
        return {}, 0

    idf_raw = {t: math.log((N_docs + 1) / (df + 1)) + 1.0 for t, df in df_counter.items()}

    total_tf = sum(tf_counter.values()) or 1
    idf = {t: (tf_counter[t] / total_tf) * idf_raw.get(t, 0.0) for t in tf_counter}

    return idf, N_docs


def prob_to_logit(p: float, eps: float = 1e-5) -> float:
    p = max(eps, min(1 - eps, p))
    return math.log(p / (1 - p))


def build_bayes_prob_from_pairs(pairs_data,
                                deduplicate: bool = False,
                                k: float = 1.0,
                                beta_pos: float = 1.0,
                                beta_neg: float = 1.0):
    def _simple_tokens(text: str):
        return [t for t in text.split() if t]

    seen = set()
    unique_rows = []
    for row in pairs_data:
        s1, s2 = row[0]
        label = int(row[-1])
        key = (s1.strip(), s2.strip(), label)
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    data_iter = unique_rows

    freq_pos, freq_neg = Counter(), Counter()
    N_pos = N_neg = 0

    for row in data_iter:
        s1, s2 = row[0]
        label = int(row[-1])
        toks1 = set(_simple_tokens(s1))
        toks2 = set(_simple_tokens(s2))
        co_tokens = toks1 & toks2
        if not co_tokens:
            continue

        if label == 1:
            freq_pos.update(co_tokens)
            N_pos += 1
        else:
            freq_neg.update(co_tokens)
            N_neg += 1

    frequency_ratios = {}
    for w in set(freq_pos) | set(freq_neg):
        f_pos = freq_pos.get(w, 0)
        f_neg = freq_neg.get(w, 0)
        ratio = f_pos / (f_neg + k) if f_neg > 0 else f_pos / k
        frequency_ratios[w] = ratio

    return frequency_ratios


def pair_bayes_prob_weighted(
        s1: str, s2: str,
        word2prob: Dict[str, float],
        idf_dict: Dict[str, float] | None = None,
        idf_pos: Dict[str, float] | None = None,
        idf_neg: Dict[str, float] | None = None,
        smooth: str = "sqrt",
        clip_max: float = 5.0,
        default_prob: float = 0.5,
        eps: float = 1e-8
) -> float:
    toks1 = simple_tokens(s1)
    toks2 = simple_tokens(s2)
    words = list(set(toks1).union(set(toks2)))
    if not words:
        return default_prob

    def base_idf(t: str) -> float:
        if idf_pos is not None and idf_neg is not None:
            b = 0.5 * idf_pos.get(t, idf_dict.get(t, 1.0) if idf_dict else 1.0) + \
                0.5 * idf_neg.get(t, idf_dict.get(t, 1.0) if idf_dict else 1.0)
        elif idf_dict is not None:
            b = idf_dict.get(t, 1.0)
        else:
            b = 1.0
        if smooth == "sqrt":
            b = math.sqrt(max(b, 0.0))
        elif smooth == "clip":
            b = min(b, clip_max)
        return b

    ws = [base_idf(t) for t in words]
    ps = [word2prob.get(t, default_prob) for t in words]

    num = sum(w * p for w, p in zip(ws, ps))
    den = sum(ws) + eps
    p_pair = num / den
    return float(max(1e-5, min(1 - 1e-5, p_pair)))


corecases = 80
unseen = 100
size = 'medium'

path = f'data/wdc/{corecases}pair'
trainpath = f'train_{size}'
testpath = f'test{unseen}'

result_path = f'{path}/{testpath}/wdc{corecases}{unseen}{size}result.csv'
save_path = f'{path}/{testpath}/wdc{corecases}{unseen}{size}best_model.pth'

index = 1
dual_output(f"elements index {index}")

max_len = 256
num_epochs = 80
batch_size = 128
targetProbs = 0.5
N = 5

USE_LEARNED_UNCERT_GATE = True
LAMBDA_UNCERT = 0.5
LAMBDA_SPARSITY = 0.1
GATE_HIDDEN = 8

USE_BAYES_CALIBRATOR = False

dual_output(f"corecases: {corecases}, unseen: {unseen}, size: {size}")
dual_output(
    f"LAMBDA_UNCERT:{LAMBDA_UNCERT},LAMBDA_SPARSITY:{LAMBDA_SPARSITY},GATE_HIDDEN:{GATE_HIDDEN},batch_size:{batch_size}")
dual_output(f"num_epochs: {num_epochs}")

model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)


class SentenceSimilarityDataset(Dataset):
    def __init__(self, data, max_len, tokenizer, global_tfidf=None):
        self.max_length = max_len
        self.data = data
        self.tokenizer = tokenizer
        self.global_tfidf = global_tfidf or {}

    def __len__(self):
        return len(self.data)

    def _tfidf_trim(self, text: str, tfidf_dict, tokenizer):
        if not text or not tfidf_dict:
            return text

        tokens = text.split()
        if len(tokens) <= self.max_length:
            return text

        scored = [(i, tfidf_dict.get(tokens[i], 0.0)) for i in range(len(tokens))]
        scored.sort(key=lambda x: (-x[1], x[0]))

        k = self.max_length
        keep_idx = sorted(i for i, _ in scored[:k])
        trimmed_tokens = [tokens[i] for i in keep_idx]
        return " ".join(trimmed_tokens)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = item[-1]
        elements = item[:-1]

        encodings_list = []

        for element in elements:
            if isinstance(element, tuple):
                text1, text2 = element
                enc = self.tokenizer(
                    text1, text2,
                    padding='max_length', truncation=True, max_length=self.max_length,
                    return_tensors='pt', add_special_tokens=True
                )
            else:
                text = element
                enc = self.tokenizer(
                    text,
                    padding='max_length', truncation=True, max_length=self.max_length,
                    return_tensors='pt', add_special_tokens=True
                )

            encodings_list.append(enc)

        input_ids = [enc['input_ids'].flatten() for enc in encodings_list]
        token_type_ids = [enc.get('token_type_ids', torch.zeros_like(enc['input_ids'])).flatten()
                          for enc in encodings_list]
        attention_mask = [enc['attention_mask'].flatten() for enc in encodings_list]

        first_pair = elements[0] if isinstance(elements[0], tuple) else ("", "")
        s1, s2 = first_pair

        enc1 = self.tokenizer(
            s1, padding='max_length', truncation=True, max_length=self.max_length,
            return_tensors='pt', add_special_tokens=True
        )
        enc2 = self.tokenizer(
            s2, padding='max_length', truncation=True, max_length=self.max_length,
            return_tensors='pt', add_special_tokens=True
        )

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long),
            'raw_pair': (s1, s2),
            'raw_input_ids1': enc1['input_ids'].flatten(),
            'raw_attention1': enc1['attention_mask'].flatten(),
            'raw_input_ids2': enc2['input_ids'].flatten(),
            'raw_attention2': enc2['attention_mask'].flatten()
        }


class HeuristicBackupHead(nn.Module):
    def __init__(self,
                 tokenizer: RobertaTokenizer,
                 max_len_attr: int = max_len,
                 normalize_feats: bool = False,
                 idf_dict: dict = None,
                 idf_pos: dict = None,
                 idf_neg: dict = None,
                 diff_alpha: float = 1.0,
                 precomputed_bayes: dict = None,
                 hmodel=None,
                 feature_mask: List[bool] = [True, False, True]):

        super().__init__()
        self._encode_fn = None
        self.tokenizer = tokenizer
        self.normalize_feats = bool(normalize_feats)

        if len(feature_mask) != 3:
            raise ValueError("feature_mask must be a list of 3 booleans")
        self.feature_mask = feature_mask

        self.idf_dict = idf_dict or {}
        self.idf_pos = idf_pos or {}
        self.idf_neg = idf_neg or {}
        self.diff_alpha = float(diff_alpha)

        self.precomputed_bayes_prob = precomputed_bayes
        self._key_patterns = {
            "title": re.compile(r'\btitle\b\s*[:：]?\s*(.+?)(?=\b(price|currency|cententy|brand|title)\b|$)', re.I),
            "name": re.compile(r'\bname\b\s*[:：]?\s*(.+?)(?=\b(price|currency|cententy|brand|title)\b|$)', re.I),
        }
        self.hmodel = hmodel

    @staticmethod
    def _simple_tokens(text: str):
        if not isinstance(text, str):
            return []
        text = text.lower()
        text = re.sub(r"[^\w\s]+", " ", text)
        return [t for t in text.split() if t]

    def _w_pos(self, t):
        return self.idf_pos.get(t, self.idf_dict.get(t, 1.0))

    def _w_neg(self, t):
        return self.idf_neg.get(t, self.idf_dict.get(t, 1.0))

    def _overlap_minus_diff(self, tokens1, tokens2, eps=1e-8):
        set1, set2 = set(tokens1), set(tokens2)
        union = set1 | set2
        if not union:
            return 0.0

        inter = set1 & set2
        diff = union - inter

        num_pos = sum(self._w_pos(t) for t in inter)
        den_pos = sum(self._w_pos(t) for t in union) + eps
        overlap_pos = num_pos / den_pos

        num_neg = sum(self._w_neg(t) for t in diff)
        den_neg = sum(self._w_neg(t) for t in union) + eps
        diff_neg = num_neg / den_neg

        return overlap_pos - self.diff_alpha * diff_neg

    def _normalize_feats(self, feats: torch.Tensor) -> torch.Tensor:
        if not self.normalize_feats:
            return feats
        f = feats.clone()
        if self.feature_mask[0]:
            alpha = float(self.diff_alpha)
            f[:, 0] = (f[:, 0] + alpha) / (1.0 + alpha)
            f[:, 0] = f[:, 0].clamp(0.0, 1.0)
        if self.feature_mask[1]:
            f[:, 1] = (f[:, 1] + 1.0) * 0.5
        if self.feature_mask[2]:
            f[:, 2] = f[:, 2].clamp(0.0, 1.0)
        return f

    def forward(self,
                v1: torch.Tensor, v2: torch.Tensor,
                texts1: List[str], texts2: List[str],
                hidden_size: int):

        device = v1.device
        B = len(texts1)
        feats = torch.zeros(B, 3, dtype=torch.float32, device=device)

        if self.feature_mask[0]:
            for i, (s1, s2) in enumerate(zip(texts1, texts2)):
                toks1 = self._simple_tokens(s1)
                toks2 = self._simple_tokens(s2)
                feats[i, 0] = self._overlap_minus_diff(toks1, toks2)

        if self.feature_mask[1]:
            v1n = v1 / (v1.norm(dim=1, keepdim=True) + 1e-6)
            v2n = v2 / (v2.norm(dim=1, keepdim=True) + 1e-6)
            feats[:, 1] = (v1n * v2n).sum(dim=1).clamp(-1.0, 1.0)

        if self.feature_mask[2]:
            bayes_probs = torch.zeros(B, dtype=torch.float32, device=device)
            if self.precomputed_bayes_prob is not None:
                for i, (s1, s2) in enumerate(zip(texts1, texts2)):
                    p_pair = pair_bayes_prob_weighted(
                        s1, s2,
                        word2prob=self.precomputed_bayes_prob,
                        idf_dict=self.idf_dict,
                        idf_pos=self.idf_pos,
                        idf_neg=self.idf_neg,
                        smooth="none"
                    )
                    bayes_probs[i] = float(p_pair)
            feats[:, 2] = bayes_probs

        gate_feats = self._normalize_feats(feats)

        z_backup = None
        p_backup = None
        if self.hmodel is not None:
            gate_feats_cpu = gate_feats.detach().cpu().numpy()
            p_backup_np = self.hmodel.predict_proba(gate_feats_cpu)[:, 1]
            p_backup = torch.tensor(p_backup_np, dtype=torch.float32, device=device).view(-1, 1)
            eps = 1e-6
            z_backup = torch.logit(p_backup.clamp(eps, 1 - eps))

        return z_backup, gate_feats, p_backup


class GateNet(nn.Module):
    def __init__(self,
                 in_dim: int = 2,
                 hidden: int | None = None,
                 init_bias: float = -2.0,
                 temperature: float = 1.0,
                 use_kan: bool = False,
                 layer_norm: bool = False):
        super().__init__()
        self.temperature = float(temperature)
        self.layer_norm = bool(layer_norm)

        self.use_kan = bool(use_kan) and (EfficientKAN is not None)

        if self.use_kan:
            self.gate = EfficientKAN([in_dim, 1])
            self.bias = nn.Parameter(torch.tensor(float(init_bias)))
            self._backend = f"KAN({EfficientKAN})"
        else:
            hidden_dim = hidden if hidden is not None else in_dim
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            with torch.no_grad():
                self.mlp[-1].bias.fill_(float(init_bias))
            self._backend = "MLP(Linear+ReLU)"

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = feats
        if self.layer_norm:
            x = F.layer_norm(x, [x.size(1)])

        if self.use_kan:
            logit = self.gate(x) + self.bias
        else:
            logit = self.mlp(x)

        return torch.sigmoid(logit / self.temperature)


class _FuzzyMembership(nn.Module):
    def __init__(self, terms_per_input, init_centers=None, init_widths=None):
        super().__init__()
        self.num_inputs = len(terms_per_input)
        self.terms_per_input = terms_per_input
        self.centers = nn.ParameterList()
        self.scales = nn.ParameterList()
        for i, K in enumerate(terms_per_input):
            c = torch.tensor(init_centers[i], dtype=torch.float32) if (init_centers and init_centers[i] is not None) \
                else torch.linspace(0.1, 0.9, steps=K)
            w = torch.tensor(init_widths[i], dtype=torch.float32) if (init_widths and init_widths[i] is not None) \
                else torch.full((K,), 0.20)
            self.centers.append(nn.Parameter(c))
            self.scales.append(nn.Parameter(w))

    @staticmethod
    def _gauss(x, c, s):
        s = F.softplus(s) + 1e-3
        return torch.exp(-0.5 * ((x.unsqueeze(-1) - c) / s) ** 2)

    def forward(self, x):
        assert x.size(1) == self.num_inputs, f"输入维度不匹配：期望 {self.num_inputs}，实际 {x.size(1)}"
        mu = [self._gauss(x[:, i], self.centers[i], self.scales[i]) for i in range(self.num_inputs)]
        return torch.cat(mu, dim=1)


class FuzzyDecisionLayer(nn.Module):
    def __init__(
            self,
            terms_per_input=(3, 3, 2, 2),
            init_centers=(
                    (0.05, 0.50, 0.95),
                    (0.10, 0.50, 0.90),
                    (0.20, 0.80),
                    (0.10, 0.90),
            ),
            init_widths=(
                    (0.10, 0.10, 0.10),
                    (0.15, 0.15, 0.15),
                    (0.15, 0.15),
                    (0.15, 0.15),
            ),
            mlp_hidden=16,
            temperature: float = 1.5
    ):
        super().__init__()
        self.fuzzy = _FuzzyMembership(
            terms_per_input=terms_per_input,
            init_centers=init_centers,
            init_widths=init_widths
        )
        in_dim = sum(terms_per_input)

        self.mlp_delta = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )
        self.mlp_gate = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )
        with torch.no_grad():
            self.mlp_delta[-1].bias.zero_()
            self.mlp_gate[-1].bias.zero_()

        self._lambda = nn.Parameter(torch.tensor(0.5))
        self.temperature = float(temperature)

        dual_output(init_centers)
        dual_output(init_widths)
        dual_output(mlp_hidden)

    @staticmethod
    def _conf_from(p: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        if y is not None:
            c = 1.0 - (p - y).abs().mul(2.0)
        else:
            c = (p - 0.5).abs().mul(2.0)
        return c.clamp(0.0, 1.0)

    def forward(self, p_main, p_backup=None, y=None):
        eps = 1e-6
        if p_main.ndim == 1:
            p_main = p_main.unsqueeze(1)
        if p_backup is None:
            p_backup = torch.full_like(p_main, 0.5)
        elif p_backup.ndim == 1:
            p_backup = p_backup.unsqueeze(1)
        if (y is not None) and (y.ndim == 1):
            y = y.unsqueeze(1)

        p_b = torch.sigmoid(torch.logit(p_backup.clamp(eps, 1 - eps)))

        c_m = self._conf_from(p_main, y)
        c_b = self._conf_from(p_b, y)

        feats = torch.cat([p_main, c_m, p_b, c_b], dim=1).clamp(0.0, 1.0)
        mu = self.fuzzy(feats)

        delta_z = self.mlp_delta(mu)
        gate_logit = self.mlp_gate(mu)
        lam_gate = torch.sigmoid(gate_logit / self.temperature)
        lam_eff = torch.sigmoid(self._lambda) * lam_gate

        z_main = torch.logit(p_main.clamp(eps, 1 - eps))
        z_final = z_main + lam_eff * delta_z
        p_final = torch.sigmoid(z_final)
        return z_final, p_final


class FuzzyScalarGate(nn.Module):
    def __init__(
            self,
            terms_per_input=(3, 3, 3),
            init_centers=((0.05, 0.50, 0.95),
                          (0.10, 0.50, 0.90),
                          (0.10, 0.50, 0.90)),
            init_widths=((0.10, 0.10, 0.10),
                         (0.15, 0.15, 0.15),
                         (0.15, 0.15, 0.15)),
            mlp_hidden=16,
            temperature: float = 1.5
    ):
        super().__init__()
        self.fuzzy = _FuzzyMembership(
            terms_per_input=terms_per_input,
            init_centers=init_centers,
            init_widths=init_widths
        )
        in_dim = sum(terms_per_input)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )
        with torch.no_grad():
            self.mlp[-1].bias.zero_()
        self.temperature = float(temperature)

    @staticmethod
    def _safe_cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6):
        a = a / (a.norm(dim=1, keepdim=True) + eps)
        b = b / (b.norm(dim=1, keepdim=True) + eps)
        return (a * b).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)

    @staticmethod
    def _mean_abs(x: torch.Tensor) -> torch.Tensor:
        return x.abs().mean(dim=1, keepdim=True)

    @staticmethod
    def _norm01_batch_scalar(x: torch.Tensor, eps: float = 1e-6):
        xmin = x.min(dim=0, keepdim=True).values
        xmax = x.max(dim=0, keepdim=True).values
        return (x - xmin) / (xmax - xmin + eps)

    def forward(self, x0: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        cos01 = (self._safe_cos(x0, xi) + 1.0) * 0.5
        r = xi.norm(dim=1, keepdim=True) / (x0.norm(dim=1, keepdim=True) + xi.norm(dim=1, keepdim=True) + eps)
        dbar = self._mean_abs(x0 - xi)
        dbar01 = self._norm01_batch_scalar(dbar)

        feats = torch.cat([cos01, r, dbar01], dim=1)
        mu = self.fuzzy(feats)
        logit = self.mlp(mu)
        gate = torch.sigmoid(logit / self.temperature)
        return gate


class WeightedClassifier(nn.Module):
    def __init__(self,
                 per_chunk_dim: int,
                 num_chunks: int,
                 learnable_alpha: bool = True,
                 alpha_init=None,
                 use_kan: bool = False,
                 kan_hidden: int | None = None,
                 mlp_hidden_dim: int = 768):
        super().__init__()
        assert num_chunks >= 2, "num_chunks 至少为 2（x0 和至少一个 xi）"
        self.num_chunks = num_chunks
        self.per_chunk_dim = per_chunk_dim
        self.concat_size = per_chunk_dim * 2

        dual_output("mlp_hidden_dim: " + str(mlp_hidden_dim))
        self.gates = nn.ModuleList([
            FuzzyScalarGate(mlp_hidden=16, temperature=1.5)
            for _ in range(num_chunks - 1)
        ])

        self.head_in_dim = self.concat_size * (num_chunks - 1)

        if use_kan:
            if kan_hidden is None:
                self.classifier = EfficientKAN([self.head_in_dim, 1])
            else:
                self.classifier = EfficientKAN([self.head_in_dim, kan_hidden, 1])
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.head_in_dim, mlp_hidden_dim),
                GELU(),
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                GELU(),
                nn.Linear(mlp_hidden_dim, 1)
            )

        if alpha_init is not None:
            alpha_init = torch.tensor(alpha_init, dtype=torch.float32)
            assert alpha_init.numel() == num_chunks - 1
            self.alpha = nn.Parameter(alpha_init, requires_grad=learnable_alpha)
        else:
            self.alpha = nn.Parameter(
                torch.ones(num_chunks - 1) / (num_chunks - 1),
                requires_grad=learnable_alpha
            )

    def forward(self, x, return_gates: bool = False):
        B, D = x.size()
        expected = self.per_chunk_dim * self.num_chunks
        assert D == expected, f"输入维度不符：got {D}, expected {expected}"

        x_chunks = torch.split(x, self.per_chunk_dim, dim=1)
        x0 = x_chunks[0]

        alpha_raw = F.softmax(self.alpha, dim=0)

        gated_chunks, gate_values = [], []
        for i in range(1, self.num_chunks):
            xi = x_chunks[i]
            gate_score = self.gates[i - 1](x0, xi)
            part = torch.cat([x0, xi], dim=1)
            scaled_part = part * (gate_score * alpha_raw[i - 1])
            gated_chunks.append(scaled_part)

            if return_gates:
                gate_values.append(gate_score.detach().cpu())

        concat_output = torch.cat(gated_chunks, dim=1)
        logits = self.classifier(concat_output)
        return (logits, gate_values) if return_gates else logits


import copy
import numpy as np


class CustomLossEntropy(torch.nn.Module):
    def __init__(self,
                 beta_f=1.0,
                 mode='macro',
                 switch_epoch=20,
                 total_epochs=50,
                 pos_weight_value=1.0,
                 schedule='cos',
                 eps=1e-8):
        super().__init__()
        assert mode in ('micro', 'macro')
        self.beta_f = float(beta_f)
        self.mode = mode
        self.switch_epoch = int(switch_epoch)
        self.pos_weight_value = float(pos_weight_value)
        self._pos_weight_tensor = None
        self._current_epoch = 1
        self.eps = float(eps)
        self.total_epochs = max(1, int(total_epochs))
        self.schedule = schedule

    def _gamma_conf(self):
        return float(0.5)

    def set_epoch(self, epoch: int):
        self._current_epoch = max(1, int(epoch))

    def set_switch_epoch(self, epoch: int):
        self.switch_epoch = max(1, int(epoch))

    def set_pos_weight(self, v: float):
        self.pos_weight_value = float(v)
        self._pos_weight_tensor = None

    def _ensure_pos_weight_tensor(self, device):
        if (self._pos_weight_tensor is None) or (self._pos_weight_tensor.device != device):
            self._pos_weight_tensor = torch.tensor(self.pos_weight_value, device=device)

    @staticmethod
    def _flatten_binary(preds: torch.Tensor, targets: torch.Tensor):
        if preds.ndim == 1:
            preds = preds.unsqueeze(1)
            targets = targets.unsqueeze(1)
        else:
            C = preds.shape[-1]
            preds = preds.reshape(-1, C)
            targets = targets.reshape(-1, C)
        return preds, targets

    def _soft_fbeta_loss(self, logits: torch.Tensor, targets: torch.Tensor):
        p = torch.sigmoid(logits).clamp(self.eps, 1 - self.eps)
        t = targets.float()

        p, t = self._flatten_binary(p, t)

        TP = (p * t).sum(dim=0)
        FP = (p * (1 - t)).sum(dim=0)
        FN = ((1 - p) * t).sum(dim=0)

        beta2 = self.beta_f * self.beta_f
        numerator = (1 + beta2) * TP
        denominator = (1 + beta2) * TP + beta2 * FN + FP + self.eps
        fbeta_per_class = numerator / denominator

        if self.mode == 'macro':
            fbeta = fbeta_per_class.mean()
        else:
            TP_m = TP.sum()
            FP_m = FP.sum()
            FN_m = FN.sum()
            numerator_m = (1 + beta2) * TP_m
            denominator_m = (1 + beta2) * TP_m + beta2 * FN_m + FP_m + self.eps
            fbeta = numerator_m / denominator_m

        loss = 1.0 - fbeta
        return loss

    def _mix_weight(self):
        e = self._current_epoch
        if e <= self.switch_epoch:
            return 1.0
        t = (e - self.switch_epoch) / max(1, (self.total_epochs - self.switch_epoch))
        t = min(max(t, 0.0), 1.0)
        if self.schedule == 'cos':
            return 0.5 * (1 + math.cos(math.pi * t))
        else:
            return 1.0 - t

    def forward(self, logits, targets, p_main=None, p_backup=None, threshold=None):
        device = logits.device
        self._ensure_pos_weight_tensor(device)

        targets = targets.float().to(device)

        loss_bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self._pos_weight_tensor
        )

        total_loss = loss_bce

        conf_main = torch.where(targets > 0.5, p_main, 1.0 - p_main)
        conf_backup = torch.where(targets > 0.5, p_backup, 1.0 - p_backup)

        margin_wrong = F.relu(0.5 - conf_main)
        loss_conf_main = (margin_wrong ** 2).mean()

        correct_mask = (conf_backup >= 0.5).float()
        low_conf_gap = (1.0 - conf_backup) * correct_mask
        loss_conf_backup = (low_conf_gap ** 2).mean()

        total_loss = loss_bce + loss_conf_main + loss_conf_backup

        return total_loss


loss_fn = CustomLossEntropy()


def get_param_groups(module: nn.Module,
                     lr: float,
                     wd_decay: float,
                     wd_no_decay: float = 0.0,
                     no_decay_keywords: Iterable[str] = ("bias", "LayerNorm.weight", "LayerNorm.bias")) -> List[Dict]:
    decay, no_decay = [], []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.endswith(k) for k in no_decay_keywords):
            no_decay.append(p)
        else:
            no_decay.append(p) if p.ndim == 1 else decay.append(p)

    groups = []
    if len(decay) > 0:
        groups.append({"params": decay, "lr": lr, "weight_decay": wd_decay})
    if len(no_decay) > 0:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": wd_no_decay})
    return groups


def _log_lrs(optim):
    lrs = [g['lr'] for g in optim.param_groups]
    dual_output("[LRs] " + ", ".join(f"{lr:.6e}" for lr in lrs))


def _decay_lr_by(optim, factor: float, min_lr: float = 1e-6):
    for g in optim.param_groups:
        old_lr = g['lr']
        new_lr = max(old_lr * factor, min_lr)
        g['lr'] = new_lr

    dual_output(f"[LR DECAY] multiplied by {factor:.3f}, min_lr={min_lr}")
    _log_lrs(optim)


def compute_entropy(probabilities):
    epsilon = 1e-10
    probabilities = torch.clamp(probabilities, min=epsilon, max=1.0 - epsilon)
    entropy = - (probabilities * torch.log(probabilities) + (1 - probabilities) * torch.log(1 - probabilities))
    return entropy


class BayesCalibrator(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z_main: torch.Tensor, b_logit: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_main, b_logit], dim=1)
        delta = self.mlp(x)
        return z_main + delta


USE_SUPCON_PRETRAIN = False
SUPCON_EPOCHS = 0
SUPCON_TAU = 0.05
SUPCON_LR = 3e-5
SUPCON_LAMBDA = 0.3


def _encode_side(input_ids: torch.Tensor, attn: torch.Tensor, backbone: RobertaModel) -> torch.Tensor:
    out = backbone(input_ids=input_ids, attention_mask=attn)
    return sentence_pool(out.last_hidden_state, attn, POOL_MODE)


def _pair_pos_mask(labels: torch.Tensor) -> torch.Tensor:
    B = labels.size(0)
    y = labels.view(-1).long()
    M = torch.zeros((2 * B, 2 * B), device=labels.device, dtype=torch.float32)
    pos = (y == 1).nonzero(as_tuple=False).view(-1)
    if pos.numel() > 0:
        M[pos, pos + B] = 1.0
        M[pos + B, pos] = 1.0
    idx = torch.arange(2 * B, device=labels.device)
    M[idx, idx] = 0.0
    return M


def supcon_loss_minimal(emb: torch.Tensor, pos_mask: torch.Tensor, tau: float = SUPCON_TAU,
                        eps: float = 1e-12) -> torch.Tensor:
    z = F.normalize(emb, dim=1)
    sim = torch.matmul(z, z.T) / tau
    eye = torch.eye(sim.size(0), device=sim.device)
    sim = sim - 1e4 * eye

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    pos_cnt = pos_mask.sum(dim=1)
    valid = pos_cnt > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=emb.device, requires_grad=True)

    pos_log = (pos_mask * log_prob).sum(dim=1) / pos_cnt.clamp_min(1.0)
    loss = -pos_log[valid].mean()
    return loss


def supcon_pretrain(train_data, tokenizer, backbone: RobertaModel, max_len: int,
                    epochs: int = SUPCON_EPOCHS, lr: float = SUPCON_LR, tau: float = SUPCON_TAU):
    dual_output(f"[SupCon] start: epochs={epochs}, tau={tau}, lr={lr}")
    backbone.train()

    dataset = SentenceSimilarityDataset(train_data, max_len, tokenizer)
    sampler = BalancedBatchSampler(dataset, batch_size=batch_size)
    loader = DataLoader(dataset, batch_sampler=sampler)

    optim = torch.optim.AdamW(get_param_groups(backbone, lr=lr, wd_decay=0.01, wd_no_decay=0.0))

    for ep in range(epochs):
        ep_loss, cnt = 0.0, 0
        for batch in loader:
            labels = batch['labels'].view(-1, 1).to(device)
            ids1 = batch['raw_input_ids1'].to(device)
            att1 = batch['raw_attention1'].to(device)
            ids2 = batch['raw_input_ids2'].to(device)
            att2 = batch['raw_attention2'].to(device)

            v1 = _encode_side(ids1, att1, backbone)
            v2 = _encode_side(ids2, att2, backbone)
            feats = torch.cat([v1, v2], dim=0)

            pos_mask = _pair_pos_mask(labels)
            loss = supcon_loss_minimal(feats, pos_mask, tau=tau)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            ep_loss += float(loss.item());
            cnt += 1

        avg = ep_loss / max(1, cnt)
        dual_output(f"[SupCon][Epoch {ep + 1}/{epochs}] loss={avg:.4f}")

    dual_output("[SupCon] done.")


def train(trainData, testData, load=False,
          idf_dict=None,
          idf_pos=None, idf_neg=None,
          diff_alpha: float = 1.0,
          switch_epoch=1,
          grad_clip=1.0,
          precomputed_bayes=None,
          hmodel=None
          ):
    model.to(device)

    test_dataset = SentenceSimilarityDataset(testData, max_len, tokenizer, idf_dict)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    classifier = WeightedClassifier(
        per_chunk_dim=model.config.hidden_size,
        num_chunks=pairNum
    ).to(device)

    heuristic_head = HeuristicBackupHead(
        tokenizer=tokenizer, max_len_attr=max_len,
        idf_dict=idf_dict,
        idf_pos=idf_pos, idf_neg=idf_neg,
        diff_alpha=diff_alpha,
        precomputed_bayes=precomputed_bayes,
        hmodel=hmodel
    ).to(device)
    heuristic_head._encode_fn = lambda kw: model(kw)

    fuzzy_decision = FuzzyDecisionLayer().to(device)

    LR_BACKBONE = 5e-5
    LR_CLASSIFIER = 3e-4
    LR_FUZZY = 1e-3

    WD_BACKBONE = 0.01
    WD_HEAD = 1e-4
    WD_NO_DECAY = 0.0

    param_groups: List[Dict] = []

    PATIENCE = 1
    FACTOR = 0.95
    bad_epochs = 0

    def add_module_groups(mod, base_lr, wd_decay, wd_no_decay):
        groups = get_param_groups(mod, lr=base_lr, wd_decay=wd_decay, wd_no_decay=wd_no_decay)
        param_groups.extend(groups)

    add_module_groups(model, LR_BACKBONE, WD_BACKBONE, WD_NO_DECAY)
    add_module_groups(classifier, LR_CLASSIFIER, WD_HEAD, WD_NO_DECAY)
    add_module_groups(fuzzy_decision, LR_FUZZY, WD_HEAD, WD_NO_DECAY)

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    _log_lrs(optimizer)

    best_f1 = 0.0
    best_epoch = 0
    best_snapshots = None
    best_threshold = 0.5
    start_epoch = 0

    if load and os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'fd_state_dict' in checkpoint:
            fuzzy_decision.load_state_dict(checkpoint['fd_state_dict'])

        best_f1 = float(checkpoint.get('best_f1', 0.0))
        best_epoch = int(checkpoint.get('epoch', 0))
        best_threshold = float(checkpoint.get('best_threshold', 0.5))
        start_epoch = best_epoch + 1

        best_snapshots = {
            "model": copy.deepcopy(model.state_dict()),
            "classifier": copy.deepcopy(classifier.state_dict()),
            "fuzzy_decision": copy.deepcopy(fuzzy_decision.state_dict())

        }

        dual_output(
            f"Loaded best from {save_path} with F1 {best_f1:.4f} at epoch {best_epoch + 1} | thr={best_threshold:.2f}"
        )
        _log_lrs(optimizer)

    train_dataset = SentenceSimilarityDataset(trainData, max_len, tokenizer, idf_dict)
    train_sampler = BalancedBatchSampler(train_dataset, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)

    for epoch in range(start_epoch, num_epochs):
        loss_fn.set_epoch(epoch + 1)

        model.train();
        classifier.train();
        heuristic_head.train();
        fuzzy_decision.train()
        train_loss = 0.0

        USE_SUPCON = (epoch < SUPCON_EPOCHS)
        BACKBONE_TRAIN_VIA_MAIN = (epoch >= SUPCON_EPOCHS)

        for batch_idx, batch in enumerate(train_loader):
            input_ids_list = [batch['input_ids'][i].to(device) for i in range(pairNum)]
            attention_mask_list = [batch['attention_mask'][i].to(device) for i in range(pairNum)]
            labels = batch['labels'].view(-1, 1).float().to(device)
            if BACKBONE_TRAIN_VIA_MAIN:
                sent_list = []
                for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    sent = sentence_pool(outputs.last_hidden_state, attention_mask, POOL_MODE)
                    sent_list.append(sent)
                combined_outputs = torch.cat(sent_list, dim=1)

            else:
                with torch.no_grad():
                    sent_list = []
                    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        sent = sentence_pool(outputs.last_hidden_state, attention_mask, POOL_MODE)
                        sent_list.append(sent)
                    combined_outputs = torch.cat(sent_list, dim=1)
            z_main = classifier(combined_outputs)
            p_main = torch.sigmoid(z_main)

            enc1_ids = batch['raw_input_ids1'].to(device)
            enc1_attn = batch['raw_attention1'].to(device)
            enc2_ids = batch['raw_input_ids2'].to(device)
            enc2_attn = batch['raw_attention2'].to(device)

            texts1, texts2 = batch['raw_pair']

            if USE_SUPCON:
                out1 = model(input_ids=enc1_ids, attention_mask=enc1_attn)
                out2 = model(input_ids=enc2_ids, attention_mask=enc2_attn)
                v1 = sentence_pool(out1.last_hidden_state, enc1_attn, POOL_MODE)
                v2 = sentence_pool(out2.last_hidden_state, enc2_attn, POOL_MODE)
                feats = torch.cat([v1, v2], dim=0)
                pos_mask = _pair_pos_mask(batch['labels'].to(device))
                loss_sup = supcon_loss_minimal(feats, pos_mask, tau=SUPCON_TAU)

            else:
                with torch.no_grad():
                    out1 = model(input_ids=enc1_ids, attention_mask=enc1_attn)
                    out2 = model(input_ids=enc2_ids, attention_mask=enc2_attn)
                    v1 = sentence_pool(out1.last_hidden_state, enc1_attn, POOL_MODE)
                    v2 = sentence_pool(out2.last_hidden_state, enc2_attn, POOL_MODE)
                loss_sup = 0

            z_backup, gate_feats, p_backup = heuristic_head(
                v1.detach(), v2.detach(), texts1, texts2, hidden_size=model.config.hidden_size
            )

            z_final, p_final = fuzzy_decision(p_main, p_backup, labels)
            loss_main = loss_fn(z_final, labels, p_main, p_backup)

            loss = loss_main + SUPCON_LAMBDA * loss_sup

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            optimizer.step()
            train_loss += float(loss.item())

            if batch_idx % 10 == 0:
                dual_output(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        model.eval();
        classifier.eval();
        heuristic_head.eval();
        fuzzy_decision.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids_list = [batch['input_ids'][i].to(device) for i in range(pairNum)]
                attention_mask_list = [batch['attention_mask'][i].to(device) for i in range(pairNum)]
                labels = batch['labels'].view(-1, 1).float().to(device)

                sent_list = []
                for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    sent = sentence_pool(outputs.last_hidden_state, attention_mask, POOL_MODE)
                    sent_list.append(sent)
                combined_outputs = torch.cat(sent_list, dim=1)
                z_main = classifier(combined_outputs)
                p_main = torch.sigmoid(z_main)

                enc1_ids = batch['raw_input_ids1'].to(device)
                enc1_attn = batch['raw_attention1'].to(device)
                enc2_ids = batch['raw_input_ids2'].to(device)
                enc2_attn = batch['raw_attention2'].to(device)
                with torch.no_grad():
                    out1 = model(input_ids=enc1_ids, attention_mask=enc1_attn)
                    out2 = model(input_ids=enc2_ids, attention_mask=enc2_attn)
                    v1 = sentence_pool(out1.last_hidden_state, enc1_attn, POOL_MODE)
                    v2 = sentence_pool(out2.last_hidden_state, enc2_attn, POOL_MODE)

                texts1, texts2 = batch['raw_pair']
                z_backup, gate_feats, p_backup = heuristic_head(
                    v1, v2, texts1, texts2, hidden_size=model.config.hidden_size
                )

                z_final, p_final = fuzzy_decision(p_main, p_backup)

                all_probs.extend(p_final.cpu().numpy().flatten().tolist())
                all_labels.extend(labels.cpu().numpy().astype(int).flatten().tolist())

        probs_arr = np.asarray(all_probs, dtype=np.float32)
        labels_arr = np.asarray(all_labels, dtype=np.int64)

        thresholds = np.arange(0.50, 0.90, 0.005)

        f1_best = 0.0
        for t in thresholds:
            preds = (probs_arr > t).astype(np.int64)
            acc_t = accuracy_score(labels_arr, preds)
            pre_t = precision_score(labels_arr, preds, zero_division=0)
            rec_t = recall_score(labels_arr, preds, zero_division=0)
            f1_t = f1_score(labels_arr, preds, zero_division=0)
            if f1_t >= f1_best:
                f1_best, thr_best = f1_t, float(t)
                acc_best, pre_best, rec_best = acc_t, pre_t, rec_t

        dual_output(
            f"Epoch {epoch + 1} (thr={thr_best:.2f}) Test Metrics: "
            f"Acc {acc_best:.4f} P {pre_best:.4f} R {rec_best:.4f} F1 {f1_best:.4f}"
        )

        improved = f1_best >= best_f1
        if improved:
            bad_epochs = 0
            best_f1 = f1_best
            best_threshold = thr_best
            best_epoch = epoch
            ckpt = {
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_f1': float(best_f1),
                'best_threshold': float(best_threshold),
                'fd_state_dict': fuzzy_decision.state_dict()

            }
            torch.save(ckpt, save_path)

            best_snapshots = {
                "model": copy.deepcopy(model.state_dict()),
                "classifier": copy.deepcopy(classifier.state_dict()),
                "fuzzy_decision": copy.deepcopy(fuzzy_decision.state_dict())
            }
            dual_output(f"New best saved at epoch {epoch + 1} with F1: {best_f1:.4f} @ thr={best_threshold:.2f}")

        else:
            bad_epochs += 1
            if best_snapshots is not None and (epoch + 1) >= switch_epoch:
                model.load_state_dict(best_snapshots["model"])
                classifier.load_state_dict(best_snapshots["classifier"])
                fuzzy_decision.load_state_dict(best_snapshots["fuzzy_decision"])

                dual_output(
                    f"No improvement, rolled back params to epoch {best_epoch + 1} "
                    f"(best_F1={best_f1:.4f} @ thr={best_threshold:.2f})"
                )

        if not improved and bad_epochs >= PATIENCE and epoch > switch_epoch and epoch > SUPCON_EPOCHS:
            _decay_lr_by(optimizer, factor=FACTOR)
            bad_epochs = 0

    dual_output(
        f"Training complete. Best F1: {best_f1:.4f} at epoch {best_epoch + 1} | "
        f"best_threshold={best_threshold:.2f}"
    )


def eval_detailed(
        test_data,
        result_file=None,
        idf_dict=None,
        idf_pos=None, idf_neg=None,
        diff_alpha: float = 1.0,
        precomputed_bayes=None,
        hmodel=None
):
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    classifier = WeightedClassifier(
        per_chunk_dim=model.config.hidden_size,
        num_chunks=pairNum,
        learnable_alpha=False
    ).to(device)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    heuristic_head = HeuristicBackupHead(
        tokenizer=tokenizer, max_len_attr=max_len,
        idf_dict=idf_dict,
        idf_pos=idf_pos, idf_neg=idf_neg,
        diff_alpha=diff_alpha,
        precomputed_bayes=precomputed_bayes,
        hmodel=hmodel
    ).to(device)
    heuristic_head._encode_fn = lambda kw: model(kw)

    fuzzy_decision = FuzzyDecisionLayer().to(device)
    if 'fd_state_dict' in checkpoint:
        fuzzy_decision.load_state_dict(checkpoint['fd_state_dict'])

    best_threshold = checkpoint.get('best_threshold', 0.5)
    best_f1_ckpt = checkpoint.get('best_f1', None)
    best_epoch_ckpt = checkpoint.get('epoch', None)

    model.eval();
    classifier.eval();
    heuristic_head.eval();
    fuzzy_decision.eval()

    test_dataset = SentenceSimilarityDataset(test_data, max_len, tokenizer, idf_dict)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_true, all_pred = [], []
    all_p_main, all_p_final, all_p_backup = [], [], []

    if result_file is None:
        base_dir = os.path.dirname(result_path)
        os.makedirs(base_dir, exist_ok=True)
        result_file = os.path.join(base_dir, f"wdc{corecases}{unseen}{size}_result_detailed.csv")

    with torch.no_grad():
        with open(result_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([f"# used_threshold={best_threshold:.2f}"])
            writer.writerow(["ID", "s1", "s2", "True", "Pred", "p_final", "p_main", "p_backup"])

            idx_global = 0
            for batch in test_loader:
                input_ids_list = [batch['input_ids'][i].to(device) for i in range(pairNum)]
                attention_mask_list = [batch['attention_mask'][i].to(device) for i in range(pairNum)]
                labels = batch['labels'].long().to(device)

                sent_list = []
                for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    sent = sentence_pool(outputs.last_hidden_state, attention_mask, POOL_MODE)
                    sent_list.append(sent)
                combined_outputs = torch.cat(sent_list, dim=1)
                z_main = classifier(combined_outputs)
                p_main = torch.sigmoid(z_main)

                enc1_ids = batch['raw_input_ids1'].to(device)
                enc1_attn = batch['raw_attention1'].to(device)
                enc2_ids = batch['raw_input_ids2'].to(device)
                enc2_attn = batch['raw_attention2'].to(device)

                out1 = model(input_ids=enc1_ids, attention_mask=enc1_attn)
                out2 = model(input_ids=enc2_ids, attention_mask=enc2_attn)
                v1 = sentence_pool(out1.last_hidden_state, enc1_attn, POOL_MODE)
                v2 = sentence_pool(out2.last_hidden_state, enc2_attn, POOL_MODE)

                texts1, texts2 = batch['raw_pair']
                z_backup, gate_feats, p_backup = heuristic_head(
                    v1, v2, texts1, texts2, hidden_size=model.config.hidden_size
                )
                if p_backup is None:
                    p_backup = gate_feats[:, 2:3].clamp(1e-6, 1 - 1e-6)

                z_final, p_final = fuzzy_decision(p_main, p_backup)

                preds = (p_final > best_threshold).long().flatten()

                B = labels.size(0)
                for i in range(B):
                    s1_i, s2_i = texts1[i], texts2[i]
                    t_i = int(labels[i].item())
                    pred_i = int(preds[i].item())
                    pf_i = float(p_final[i].item())
                    pm_i = float(p_main[i].item())
                    pb_i = float(p_backup[i].item())

                    writer.writerow([
                        idx_global, s1_i, s2_i,
                        t_i, pred_i,
                        f"{pf_i:.6f}",
                        f"{pm_i:.6f}",
                        f"{pb_i:.6f}"
                    ])

                    all_true.append(t_i);
                    all_pred.append(pred_i)
                    all_p_main.append(pm_i);
                    all_p_final.append(pf_i);
                    all_p_backup.append(pb_i)
                    idx_global += 1

    acc = accuracy_score(all_true, all_pred)
    pre = precision_score(all_true, all_pred, zero_division=0)
    rec = recall_score(all_true, all_pred, zero_division=0)
    f1 = f1_score(all_true, all_pred, zero_division=0)

    dual_output("\n[Eval-Detailed] Overall metrics (fixed best_threshold):")
    dual_output(f"Accuracy: {acc:.4f}  Precision: {pre:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    if best_f1_ckpt is not None:
        epoch_show = (best_epoch_ckpt + 1) if best_epoch_ckpt is not None else '?'
        dual_output(f"[History(best)] F1={best_f1_ckpt:.4f} @ thr={best_threshold:.2f} (epoch {epoch_show})")
    dual_output(f"Per-sample CSV written to: {result_file}")

    return {
        'accuracy': acc,
        'precision': pre,
        'recall': rec,
        'f1': f1,
        'threshold_used': best_threshold,
        'history_best_f1': best_f1_ckpt,
        'history_best_epoch': (best_epoch_ckpt + 1) if best_epoch_ckpt is not None else None,
        'csv': result_file
    }


from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV


def train_svm_model(train_features, train_labels):
    svmmodel = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    model = CalibratedClassifierCV(svmmodel, method='isotonic', cv=5)
    model.fit(train_features, train_labels)
    return model


def extract_features_and_labels(train_data, tokenizer, max_len, batch_size, idf_dict, idf_pos, idf_neg,
                                precomputed_bayes, pairNum):
    heuristic_head = HeuristicBackupHead(
        tokenizer=tokenizer, max_len_attr=max_len,
        idf_dict=idf_dict,
        idf_pos=idf_pos, idf_neg=idf_neg,
        precomputed_bayes=precomputed_bayes
    ).to(device)

    model.to(device)
    heuristic_head._encode_fn = lambda kw: model(kw)
    heuristic_head.to(device)

    train_dataset = SentenceSimilarityDataset(train_data, max_len, tokenizer, idf_dict)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    all_features = []
    all_labels = []

    heuristic_head.eval()

    with torch.no_grad():
        for batch in train_loader:
            labels = batch['labels'].view(-1, 1).float().to(device)

            enc1_ids = batch['raw_input_ids1'].to(device)
            enc1_attn = batch['raw_attention1'].to(device)
            enc2_ids = batch['raw_input_ids2'].to(device)
            enc2_attn = batch['raw_attention2'].to(device)

            with torch.no_grad():
                out1 = model(input_ids=enc1_ids, attention_mask=enc1_attn)
                out2 = model(input_ids=enc2_ids, attention_mask=enc2_attn)

                v1 = sentence_pool(out1.last_hidden_state, enc1_attn, POOL_MODE)
                v2 = sentence_pool(out2.last_hidden_state, enc2_attn, POOL_MODE)

            texts1, texts2 = batch['raw_pair']
            z_backup, gate_feats, p_backup = heuristic_head(
                v1, v2, texts1, texts2, hidden_size=model.config.hidden_size
            )

            features = gate_feats.cpu().numpy()
            all_features.extend(features)

            all_labels.extend(labels.cpu().numpy().flatten())

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    return all_features, all_labels


def svm_evaluate_model(svm_model, test_features, test_labels):
    predictions = svm_model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)

    dual_output(f"Accuracy: {accuracy:.4f}")
    dual_output(f"Precision: {precision:.4f}")
    dual_output(f"Recall: {recall:.4f}")
    dual_output(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    dual_output("load trainData, testData")
    trainData, augTrainData, testData, pairNum = getData(path, trainpath, testpath)

    IDF_CORPUS_SOURCE = 'train'
    if IDF_CORPUS_SOURCE == 'train':
        idf_dict, N_docs = build_global_idf_from_pairs(trainData)
    else:
        idf_dict, N_docs = build_global_idf_from_pairs(testData)
    dual_output(f"[Global IDF] built with {N_docs} docs, vocab size={len(idf_dict)}")

    idf_pos, idf_neg, N_pos, N_neg = build_class_idf_from_pairs(trainData)
    dual_output(f"[Class IDF] pos_docs={N_pos}, neg_docs={N_neg}, |idf_pos|={len(idf_pos)}, |idf_neg|={len(idf_neg)}")

    precomputed_bayes = build_bayes_prob_from_pairs(trainData, k=1.0, beta_pos=1.0, beta_neg=1.0)
    dual_output(f"[Bayes Prob] built with {len(precomputed_bayes)} tokens")

    dual_output("Extracting features and labels for training and testing")
    svm_features, svm_labels = extract_features_and_labels(augTrainData, tokenizer, max_len, batch_size=64,
                                                           idf_dict=idf_dict, idf_pos=idf_pos, idf_neg=idf_neg,
                                                           precomputed_bayes=precomputed_bayes, pairNum=pairNum)
    svm_test_features, svm_test_labels = extract_features_and_labels(testData, tokenizer, max_len, batch_size=64,
                                                                     idf_dict=idf_dict, idf_pos=idf_pos,
                                                                     idf_neg=idf_neg,
                                                                     precomputed_bayes=precomputed_bayes,
                                                                     pairNum=pairNum)

    dual_output("Training SVM model")
    svm_model = train_svm_model(svm_features, svm_labels)
    dual_output(f"Evaluating the SVM model")
    svm_evaluate_model(svm_model, svm_test_features, svm_test_labels)

    train(augTrainData, testData, load=False,
          idf_dict=idf_dict, idf_pos=idf_pos, idf_neg=idf_neg,
          precomputed_bayes=precomputed_bayes, hmodel=svm_model)

    dual_output("Starting evaluation (detailed per-sample dump)...")
    eval_detailed(testData,
                  idf_dict=idf_dict, idf_pos=idf_pos, idf_neg=idf_neg,
                  precomputed_bayes=precomputed_bayes, hmodel=svm_model)