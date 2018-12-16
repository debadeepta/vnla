import os
import sys

sys.path.append('build')
import MatterSim
import time
import math
import cv2
import json

WIDTH = 640
HEIGHT = 480
VFOV = math.radians(60)
HFOV = VFOV * WIDTH / HEIGHT
TEXT_COLOR = [230, 40, 40]

cv2.namedWindow('displaywin')

sim = MatterSim.Simulator()
sim.setDiscretizedViewingAngles(True)
sim.setCameraResolution(WIDTH, HEIGHT)
sim.setCameraVFOV(VFOV)
sim.setNavGraphPath('../data/connectivity')
sim.setDatasetPath('data')
sim.init()

set_name = sys.argv[1]
result_file = sys.argv[2]
with open(result_file) as f:
    data = json.load(f)

with open('../data/intent_based_commands/R2R_' + set_name + '.json') as f:
    tmp = json.load(f)
    queries = {}
    for item in tmp:
        queries[item['path_id']] = item

while True:
    print('Type a random number from 0 to 9999: ')
    idx = int(sys.stdin.readline())
    item = data[idx]

    query = queries[int(item['instr_id'].split('_')[0])]
    print query['instructions'][0]

    start_point = item['trajectory'][0]
    print item['scan'], item['instr_id'], query['heading'], start_point[0], item['is_success']

    sim.newEpisode(item['scan'], start_point[0], start_point[1], start_point[2])

    step_idx = 0

    ended = False
    while True:
        state = sim.getState()

        locations = state.navigableLocations
        im = state.rgb
    	origin = locations[0].point
    	for idx, loc in enumerate(locations[1:]):
            # Draw actions on the screen
            fontScale = 3.0/loc.rel_distance
            x = int(WIDTH/2 + loc.rel_heading/HFOV*WIDTH)
            y = int(HEIGHT/2 - loc.rel_elevation/VFOV*HEIGHT)
            cv2.putText(im, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale, TEXT_COLOR, thickness=3)
        cv2.imshow('displaywin', im)

        k = cv2.waitKey(1)

        if k == ord('q'):
            break

        if k == ord('r'):
            sim.newEpisode(item['scan'], start_point[0], start_point[1], start_point[2])

        if k == ord('f'):
            action = item['agent_nav'][step_idx]
            if (action[0] == 0 and action[1] == 0 and action[2] == 0) or \
                step_idx == len(item['agent_nav']) - 1:
                ended = True
            if ended:
                action = (0, 0, 0)
            else:
                step_idx += 1

            print(step_idx, action, item['agent_ask'][step_idx])
            print(item['trajectory'][step_idx])
            sim.makeAction(action[0], action[1], action[2])




