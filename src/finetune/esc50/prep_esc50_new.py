# -*- coding: utf-8 -*-

from genericpath import exists
import numpy as np
import json
import os
import zipfile
import wget

# label = np.loadtxt('/data/sls/scratch/yuangong/aed-pc/src/utilities/esc50_label.csv', delimiter=',', dtype='str')
# f = open("/data/sls/scratch/yuangong/aed-pc/src/utilities/esc_class_labels_indices.csv", "w")
# f.write("index,mid,display_name\n")
#
# label_set = []
# idx = 0
# for j in range(0, 5):
#     for i in range(0, 10):
#         cur_label = label[i][j]
#         cur_label = cur_label.split(' ')
#         cur_label = "_".join(cur_label)
#         cur_label = cur_label.lower()
#         label_set.append(cur_label)
#         f.write(str(idx)+',/m/07rwj'+str(idx).zfill(2)+',\"'+cur_label+'\"\n')
#         idx += 1
# f.close()
#

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

# downlooad esc50
# dataset provided in https://github.com/karolpiczak/ESC-50
base_dir = '/git/datasets/esc50/ESC-50-master/'
if os.path.exists(base_dir) == False:
    esc50_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
    wget.download(esc50_url, out='/git/datasets/esc50')
    with zipfile.ZipFile('/git/datasets/esc50/ESC-50-master.zip', 'r') as zip_ref:
        zip_ref.extractall('/git/datasets/esc50/')
    os.remove('/git/datasets/esc50/ESC-50-master.zip')

    # convert the audio to 16kHz
if os.path.exists(os.path.join(base_dir,"audio_16k/*.wav")) == False:
    if not os.path.exists(os.path.join(base_dir,"audio_16k")):
        os.makedirs(os.path.join(base_dir,"audio_16k"))
    audio_list = get_immediate_files('/git/datasets/esc50/ESC-50-master/audio')
    for audio in audio_list:
        print('sox ' + base_dir + '/audio/' + audio + ' -r 16000 ' + base_dir + '/audio_16k/' + audio)
        os.system('sox ' + base_dir + '/audio/' + audio + ' -r 16000 ' + base_dir + '/audio_16k/' + audio)

label_set = np.loadtxt('/git/datasets/esc50/esc_class_labels_indices.csv', delimiter=',', dtype='str')
label_map = {}
for i in range(1, len(label_set)):
    label_map[eval(label_set[i][2])] = label_set[i][0]
print(label_map)

# fix bug: generate an empty directory to save json files
if os.path.exists('/git/datasets/esc50/datafiles') == False:
    os.mkdir('/git/datasets/esc50/datafiles')

for fold in [1,2,3,4,5]:
    base_path = "/git/datasets/esc50/ESC-50-master/audio_16k/"
    meta = np.loadtxt('/git/datasets/esc50/ESC-50-master/meta/esc50.csv', delimiter=',', dtype='str', skiprows=1)
    train_wav_list = []
    eval_wav_list = []
    for i in range(0, len(meta)):
        cur_label = label_map[meta[i][3]]
        cur_path = meta[i][0]
        cur_fold = int(meta[i][1])
        # /m/07rwj is just a dummy prefix
        cur_dict = {"wav": base_path + cur_path, "labels": '/m/07rwj'+cur_label.zfill(2)}
        if cur_fold == fold:
            eval_wav_list.append(cur_dict)
        else:
            train_wav_list.append(cur_dict)

    print('fold {:d}: {:d} training samples, {:d} test samples'.format(fold, len(train_wav_list), len(eval_wav_list)))

    with open('/git/datasets/esc50/datafiles/esc_train_data_'+ str(fold) +'.json', 'w') as f:
        json.dump({'data': train_wav_list}, f, indent=1)
    with open('/git/datasets/esc50/datafiles/esc_eval_data_'+ str(fold) +'.json', 'w') as f:
        json.dump({'data': eval_wav_list}, f, indent=1)

print('Finished ESC-50 Preparation')
