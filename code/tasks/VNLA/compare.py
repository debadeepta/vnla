import sys
import os
import json

with open('output/v3_learn_to_ask_nav_sample_ask_teacher//snapshots/v3_learn_to_ask_nav_sample_ask_teacher_val_unseen.test_unseen.json') as f:
    data_direct = json.load(f)

with open('output/v3_verbal_learn_to_ask_nav_sample_ask_teacher/snapshots/v3_verbal_learn_to_ask_nav_sample_ask_teacher_val_unseen.test_unseen.json') as f:
    data_indirect = json.load(f)


def map_success(data):
    new_map = {}
    for item in data:
        new_map[item['instr_id']] = bool(item['is_success'])
    return new_map

indirect_success_map = map_success(data_indirect)
direct_success_map = map_success(data_direct)

assert len(indirect_success_map) == len(direct_success_map)

total = 0
common = 0
indirect = 0
direct = 0

for k in indirect_success_map:
    total    += 1
    common   += (indirect_success_map[k] == direct_success_map[k] == True)
    indirect += (indirect_success_map[k] == True != direct_success_map[k])
    direct   += (indirect_success_map[k] == False != direct_success_map[k])

    if indirect_success_map[k] == False and direct_success_map[k] == True:
        print k

print total, common, indirect, direct


