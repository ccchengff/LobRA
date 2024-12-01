import os
import json
import pickle
import argparse
import time
import bisect
import pyarrow.parquet as pq
from tqdm import tqdm
from types import SimpleNamespace
from data_utils.tokenizer import build_tokenizer
from data_utils.utils import jload

class Encoder(object):
    def __init__(self, args):
        self.args = args
        self.tokenizer = build_tokenizer(self.args)

    def pad_id(self):
        return self.tokenizer.pad

    def encode(self, data):
        doc = data[self.args.key] # key: content for web, text for wiki
        assert self.args.tokenizer_type == 'GPT2BPETokenizer', 'Now only support GPT2BPETokenizer!'
        doc_ids = self.tokenizer.tokenize(doc)
        
        return doc_ids

def _load_dataset_to_json(dataset_name, root_folder='data', override=False):
    """Load dataset from parquet file and save as json file."""
    json_file = f'{root_folder}/{dataset_name}/{dataset_name}.json'
    if os.path.exists(json_file) and not override:
        return json_file
    parquet_files = []
    # 遍历f'{root_folder}/{dataset_name}'目录下的所有文件
    for _, _, files in os.walk(f'{root_folder}/{dataset_name}'):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(file)
    
    json_data = ''
    for parquet_file in parquet_files:
        tabel = pq.read_table(f'{root_folder}/{dataset_name}/{parquet_file}')
        df = tabel.to_pandas()
        json_data += df.to_json(orient='records', lines=True)
    
    f_dirname = os.path.dirname(json_file)
    if f_dirname != "":
        os.makedirs(f_dirname, exist_ok=True)
    f = open(json_file, 'w')
    f.write(json_data)
    f.close()
    return json_file

def _encode_dataset(json_file, encoder, cache_path=None):
    """Encode dataset from json file."""
    data = []
    if not cache_path:
        cache_path = json_file.split('.')[0] + f'_cache.pkl'
    if os.path.exists(cache_path):
        print(f'Loading exists cache from {cache_path} begin ...')
        start_time = time.time()
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        end_time = time.time()
        print(f'Loading exists cache end, time cost: {end_time - start_time: .3f} s')
    else:
        print(f'Building dataset begin ...')
        start_time = time.time()
        try:
            jdict = jload(json_file)
        except BaseException:
            with open(json_file, 'r') as f:
                lines = f.readlines()
            jdict = [json.loads(line.strip()) for line in lines]
        for example in jdict:
            doc_ids = encoder.encode(example)
            if len(doc_ids) > 0:
                data.append(doc_ids)
        # save cache
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        end_time = time.time()
        print(f'Building dataset end, time cost: {end_time - start_time: .3f} s')
    return data

def _load_dataset(dataset_name, encoder, cache_path=None, root_folder='data', override=False):
    # Load and encode dataset
    json_file = _load_dataset_to_json(dataset_name, root_folder, override)
    data = _encode_dataset(json_file, encoder, cache_path)
    return data

def load_dataset(dataset_name, key, encoder_class=Encoder, vocab_file=None,
                 merge_file=None, cache_path=None, root_folder='data', override=False):
    # Load and encode dataset
    if vocab_file is None:
        vocab_file = f'{root_folder}/vocab.json'
    if merge_file is None:
        merge_file = f'{root_folder}/merges.txt'
    encoder_args = {
        'key': key,
        'rank': 0,
        'make_vocab_size_divisible_by': 128,
        'tensor_model_parallel_size': 1,
        'vocab_extra_ids': 0,
        'tokenizer_type': 'GPT2BPETokenizer',
        'vocab_file': vocab_file,
        'merge_file': merge_file
    }
    encoder_args = SimpleNamespace(**encoder_args)
    encoder = encoder_class(encoder_args)
    data = _load_dataset(dataset_name, encoder, cache_path, root_folder, override)
    return data

def load_padded_dataset(dataset_name, encoder, max_seq_len, cache_path=None, root_folder='data', override=False):
    # Load and encode dataset
    data = _load_dataset(dataset_name, encoder, cache_path, root_folder, override)
    for idx, doc_ids in enumerate(data):
        if len(doc_ids) > max_seq_len:
            data[idx] = doc_ids[:max_seq_len]
    # deal with max_seq_len + 1 (for tokens/labels seq_len = max_seq_len+1 - 1)
    # print(f'Cutting or padding data to max_seq_len + 1 = {max_seq_len + 1} begin ...')
    # start_time = time.time()
    # max_seq_len = max_seq_len + 1
    # for idx, doc_ids in enumerate(data):
    #     if len(doc_ids) > max_seq_len:
    #         data[idx] = doc_ids[:max_seq_len]
    #     elif len(doc_ids) < max_seq_len:
    #         data[idx] += [encoder.pad_id()] * (max_seq_len - len(doc_ids))
    # end_time = time.time()
    # print(f'Cutting or padding data end, time cost: {end_time - start_time: .3f} s')
    return data

def _get_length_num_distribution(data, min_seq_len=256, max_seq_len=8192, buckets=None):
    # Get length distribution
    seq_len_num_distribution = {}
    for doc_ids in data:
        sample_len = len(doc_ids)
        if buckets is None:
            padded_len = max(min(2 ** (sample_len.bit_length()), max_seq_len), min_seq_len)
        else:
            padded_len = buckets[bisect.bisect_left(buckets, sample_len)]
        seq_len_num_distribution[padded_len] = seq_len_num_distribution.get(padded_len, 0) + 1
    seq_len_num_distribution = dict(sorted(seq_len_num_distribution.items(), key=lambda x: x[0]))
    return seq_len_num_distribution

def get_length_distribution(data, min_seq_len=256, max_seq_len=8192, buckets=None):
    total_num = len(data)
    seq_len_distribution = _get_length_num_distribution(data, min_seq_len, max_seq_len, buckets)
    for seq_len, num in seq_len_distribution.items():
        seq_len_distribution[seq_len] = num / total_num
    return seq_len_distribution

def get_length_list(data):
    return [len(doc_ids) for doc_ids in data]

def get_length_cumulative_distribution(data, min_seq_len=256, max_seq_len=8192):
    total_num = len(data)
    seq_len_distribution = _get_length_num_distribution(data, min_seq_len, max_seq_len)
    cumulative_num = 0
    for s in seq_len_distribution.keys():
        num = seq_len_distribution[s]
        seq_len_distribution[s] = num + cumulative_num
        cumulative_num += num
    assert cumulative_num == total_num
    for seq_len, num in seq_len_distribution.items():
        seq_len_distribution[seq_len] = num / total_num
    return seq_len_distribution

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default='alpaca/alpaca'
    )
    parser.add_argument(
        "--key", type=str, default='text'
    )
    parser.add_argument(
        "--override", type=bool, default=False
    )
    parser.add_argument(
        "--cache_path", type=str, default=None
    )
    parser.add_argument(
        "--root_folder", type=str, default='data'
    )
    parser.add_argument(
        "--vocab_file", type=str, default=None
    )
    parser.add_argument(
        "--merge_file", type=str, default=None
    )

    args = parser.parse_args()
    data = load_dataset(args.dataset, args.key, vocab_file=args.vocab_file,
                        merge_file=args.merge_file, cache_path=args.cache_path,
                        root_folder=args.root_folder, override=args.override)
    print(f'dataset {args.dataset} seq_len distribution: {get_length_distribution(data)}')
    print(f'dataset {args.dataset} seq_len cumulative distribution: {get_length_cumulative_distribution(data)}')