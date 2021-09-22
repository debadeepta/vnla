from __future__ import division

import os
import sys
import re
import string
import json
import time
import math
from collections import Counter
import numpy as np
import networkx as nx
import base64
import csv

# padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>', '<EOS>']
padding_idx = base_vocab.index('<PAD>')
csv.field_size_limit(sys.maxsize)


def load_nav_graphs(scan, path=None):

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    if path is not None:
        DATA_DIR = path
    else:
        DATA_DIR = os.getenv('PT_DATA_DIR', '../../../data')
    with open(os.path.join(DATA_DIR, 'connectivity/%s_connectivity.json' % scan)) as f:
        G = nx.Graph()
        positions = {}
        data = json.load(f)
        for i, item in enumerate(data):
            if item['included']:
                for j, conn in enumerate(item['unobstructed']):
                    if conn and data[j]['included']:
                        positions[item['image_id']] = np.array([item['pose'][3],
                                item['pose'][7], item['pose'][11]]);
                        assert data[j]['unobstructed'][i], 'Graph should be undirected'
                        G.add_edge(item['image_id'],data[j]['image_id'],
                            weight=distance(item,data[j]))
        nx.set_node_attributes(G, values=positions, name='position')
        return G

def load_region_map(scan):
    DATA_DIR = os.getenv('PT_DATA_DIR', '../../../data')
    region_map = {}
    with open(os.path.join(DATA_DIR, 'view_to_region/%s.panorama_to_region.txt' % scan)) as f:
        for line in f:
            fields = line.rstrip().split()
            view = fields[1]
            region = fields[-1]
            region_map[view] = region
    return region_map

def load_datasets(splits, path, prefix=''):
    data = []
    for split in splits:
        with open(os.path.join(path, prefix + '_%s.json' % split)) as f:
            data += json.load(f)
    return data


class Tokenizer(object):
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # Split on any non-alphanumeric character

    def __init__(self, vocab, encoding_length, split_by_spaces=True):
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.word_to_index = {}
        if vocab:
            for i,word in enumerate(vocab):
                self.word_to_index[word] = i
        self.split_by_spaces = split_by_spaces

    def split_sentence(self, sentence):
        if self.split_by_spaces:
            return sentence.split()

        toks = []
        for word in [s.strip().lower() for s in self.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless
            # it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def encode_sentence(self, sentence, encoding_length=None, reverse=True, eos=True, to_numpy=True):
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')

        encoding = []

        if isinstance(sentence, list):
            sent_split = sentence
        else:
            sent_split = self.split_sentence(sentence)

        if reverse:
            sent_split = sent_split[::-1]

        for word in sent_split: # reverse input sentences
            if word in self.word_to_index:
                encoding.append(self.word_to_index[word])
            else:
                encoding.append(self.word_to_index['<UNK>'])
        if eos:
            encoding.append(self.word_to_index['<EOS>'])

        if encoding_length is None:
            encoding_length = self.encoding_length

        if len(encoding) < encoding_length:
            encoding.extend([self.word_to_index['<PAD>']] * (encoding_length-len(encoding)))

        encoding = encoding[:encoding_length]

        if to_numpy:
            encoding = np.array(encoding)

        return encoding

def build_vocab(path, splits, min_count, max_length, start_vocab=base_vocab,
    split_by_spaces=False, prefix=''):

    count = Counter()
    t = Tokenizer(None, max_length, split_by_spaces=split_by_spaces)
    data = load_datasets(splits, path, prefix=prefix)
    for item in data:
        for instr in item['instructions']:
            if split_by_spaces:
                count.update(instr.split())
            else:
                count.update(t.split_sentence(instr))
    vocab = list(start_vocab)
    for word,num in count.most_common():
        if num >= min_count:
            vocab.append(word)
        else:
            break
    return vocab


def write_vocab(vocab, path):
    print('Writing vocab of size %d to %s' % (len(vocab),path))
    with open(path, 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)


def read_vocab(paths):
    vocab = base_vocab
    added = set(vocab)
    for path in paths:
        with open(path) as f:
            words = [word.strip() for word in f.readlines()]
            for w in words:
                if w not in added:
                    added.add(w)
                    vocab.append(w)
    print('Read vocab of size', len(vocab))
    return vocab

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def read_subgoal_vocab(paths):
    vocab = ['<PAD>', '<UNK>', '<EOS>', '<BOS>', '<NO_OBJ>']
    added = set(vocab)
    for path in paths:
        with open(path) as f:
            words = [word.strip() for word in f.readlines()]
            for w in words:
                if w not in added:
                    added.add(w)
                    vocab.append(w)
    print('Read vocab of size', len(vocab))
    return vocab

def load_img_features(path):
    print('Loading image features from %s' % path)
    tsv_fieldnames = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
    features = {}
    with open(path, "rt") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
        for i, item in enumerate(reader):
            image_h = int(item['image_h'])
            image_w = int(item['image_w'])
            vfov = int(item['vfov'])
            long_id = item['scanId'] + '_' + item['viewpointId']
            features[long_id] = np.frombuffer(
                    base64.decodebytes(bytearray(item['features'], 'utf-8')),
                    dtype=np.float32).reshape((36, 2048))
    return image_h, image_w, vfov, features

def load_region_label_to_name():
    region_label_to_name = {}
    with open('../../../data/region_label.txt') as f:
        for line in f:
            line = line.rstrip()
            code = line[1]
            try:
                label = line[line.index('=') + 2:line.index('(') - 1]
            except ValueError:
                label = line[line.index('=') + 2:]
            region_label_to_name[code] = label
    return region_label_to_name

def load_panos_to_region(house_id, region_label_to_name, include_region_id = False):
    pano_file = '../../data/v1/scans/' + house_id + \
        '/house_segmentations/' + 'panorama_to_region.txt'
    panos_to_region = {}
    with open(pano_file) as f:
        for line in f:
            values = line.rstrip().split()
            if include_region_id:
                panos_to_region[values[1]] = [values[-2], values[-1]]
            else:
                panos_to_region[values[1]] = values[-1]
    return panos_to_region
