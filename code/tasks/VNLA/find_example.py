import os
import sys
import json

with open('output/final/v3_verbal_hard_learn_to_ask_nav_sample_ask_teacher/snapshots/v3_verbal_hard_learn_to_ask_nav_sample_ask_teacher_val_unseen.test_unseen.json') as f:
    data = json.load(f)

for item in data:
    if 10 <= len(item['agent_ask']) <= 15 and not item['is_success']:
        print item['instr_id']


"""
longest = 1e9
res = []

for item in data:
    if not item['is_success']:
        this_len = len(item['agent_ask'])
        if this_len < longest:
            res = [item['instr_id']]
            longest = this_len
        elif this_len == longest:
            res.append(item['instr_id'])

print longest
for x in res:
    print x
"""

