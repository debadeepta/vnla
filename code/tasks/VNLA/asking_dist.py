from __future__ import division

import sys
import json
from collections import defaultdict
import numpy as np

in_file = sys.argv[1]

with open(in_file) as f:
    data = json.load(f)


ratios = []

total = 0
total_steps = 0
for item in data:
    for i, j in enumerate(item['agent_ask']):
        total_steps += 1
        if j == 1:
            ratios.append(i / len(item['agent_ask']))
            total += 1

print total / total_steps

bins = [0.2, 0.4, 0.6, 0.8, 1.0]

ratios_by_bins = defaultdict(list)

for r in ratios:
    for b in bins:
        if r <= b:
            ratios_by_bins[b].append(r)
            break

sorted_bins = []
for k, v in ratios_by_bins.iteritems():
    sorted_bins.append((k, len(v) / total))

sorted_bins = sorted(sorted_bins)

name_str = []
val_str = []
total = 0
for i, (k, v) in enumerate(sorted_bins):
    total += v
    name_str.append('"%s"' % str(k))
    val_str.append('%s' % str(int(total * 1000) / 10.))
    print '<=' + str(k) + '\t' + str(int(total * 1000) / 10.)

print ','.join(name_str)
print ','.join(val_str)


