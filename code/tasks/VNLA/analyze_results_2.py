import os
import sys
import json
from collections import defaultdict, Counter

import numpy as np
from scipy.stats import sem

in_file = sys.argv[1]

if 'unseen' in in_file:
    data_file = '../../../data/intent_based_commands_v3/R2R_test_unseen.json'
elif 'seen' in in_file:
    data_file = '../../../data/intent_based_commands_v3/R2R_test_seen.json'

with open(data_file) as f:
    data = json.load(f)
    data_map = {}
    for item in data:
        data_map[str(item['path_id']) + '_0'] = item

with open(in_file) as f:
    results = json.load(f)

# Accuracy by object

"""
acc = defaultdict(list)
room  = defaultdict(set)
objs = []
for item in results:
    data_item = data_map[item['instr_id']]
    obj_name = data_item['object_name']
    acc[obj_name].append(item['is_success'] * 100)
    objs.append(obj_name)

    room[obj_name].add(data_item['end_region_name'])

res = {}
for k in acc:
    if len(acc[k]) >= 50:
        res[k] = (np.average(acc[k]), 1.96* sem(acc[k]), len(acc[k]))

tmp = []
for k, v in res.iteritems():
    tmp.append((k, v))

tmp = sorted(tmp, key=lambda x: -x[1][0])

objs = Counter(objs)

name_str = []
mean_str = []
se_str = []
for x in tmp[:5]:
    name_str.append('"%s (%d)"' % (x[0], x[1][2]))
    mean_str.append('%.2f' % x[1][0])
    se_str.append('%.2f'% x[1][1])
    print str(x[0]) + ' (' + str(objs[x[0]]) + ')' '\t' + str(x[1][0]) + ' ' + str(x[1][1])
    print room[x[0]]

print ','.join(name_str)
print ','.join(mean_str)
print ','.join(se_str)

"""

acc = defaultdict(list)


z = defaultdict(set)

objs = []
for item in results:
    data_item = data_map[item['instr_id']]
    try:
        obj_name = data_item['end_region_name'][:data_item['end_region_name'].index(' or ')]
    except:
        obj_name = data_item['end_region_name']
    acc[obj_name].append(item['is_success'] * 100)
    objs.append(obj_name)

    z[obj_name].add(data_item['object_name'])

total = 0
res = {}
for k in acc:
    if len(acc[k]) >= 50:
        res[k] = (np.average(acc[k]), 1.96*sem(acc[k]), len(acc[k]))
    total += len(acc[k])


for k in res:
    print k, res[k]
    print '     ', z[k]


tmp = []
for k, v in res.iteritems():
    tmp.append((k, v))

tmp = sorted(tmp, key=lambda x: x[1][0])

objs = Counter(objs)

name_str = []
mean_str = []
se_str = []
for x in reversed(tmp[:5]):
    name_str.append('"%s (%d)"' % (x[0], x[1][2]))
    mean_str.append('%.2f' % x[1][0])
    se_str.append('%.2f'% x[1][1])
    print str(x[0]) + ' (' + str(objs[x[0]]) + ')' '\t' + str(x[1][0]) + ' ' + str(x[1][1])

print ','.join(name_str)
print ','.join(mean_str)
print ','.join(se_str)

#for x in tmp[:20]:
#    print str(x[0] * 5) + '-' + str((x[0] + 1) * 5 - 1) + '\t' + str(x[1])

