from gluoncv import data, utils
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np

train_dataset = data.VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
val_dataset = data.VOCDetection(splits=[(2007, 'test')])
print('Num of training images:', len(train_dataset))
print('Num of validation images:', len(val_dataset))

desired_labels = [(6.0, 14.0), (11.0, 14.0), (1.0, 14.0), (14.0, 17.0)]
heights = []
widths = []
RGBs = []
HWdict = dict()
HWlabeldict = dict()
with open('train_idxs.txt', 'w') as f:
    for idx, (x, y) in enumerate(train_dataset):
        heights.append(x.shape[0])
        widths.append(x.shape[1])
        RGBs.append(x.shape[2])
        hw = (x.shape[0],x.shape[1])
        curr_count = HWdict.get(hw, 0)
        HWdict[hw] = curr_count + 1
        # print(y)
        # print(type(y))
        if hw == (375, 500):
            class_ids = tuple(np.unique(y[:, 4:5]))
            if len(class_ids) > 1:
                curr_label_count = HWlabeldict.get(class_ids, 0)
                HWlabeldict[class_ids] = curr_label_count + 1
            if class_ids in desired_labels:
                f.write(str(idx) + '\n')

print(HWdict)

# keys = HWdict.keys()
# values = HWdict.values()
# [[keys, values]] = ((str(key), value) for key,value in HWdict.items())
keys = [str(key) for key in HWdict.keys()]
values = list(HWdict.values())
# plt.bar(keys, values)
# plt.show()
max_val = max(HWdict.values())
print(max_val)
max_key = max(HWdict, key=HWdict.get)
print(max_key)
print(HWdict.get(max_key))

HWlabeldict = dict(sorted(HWlabeldict.items(), key=lambda item: item[1]))
print(HWlabeldict)
max_val_label = max(HWlabeldict.values())
print(max_val_label)
max_key = max(HWlabeldict, key=HWlabeldict.get)
print(max_key)
print(HWlabeldict.get(max_key))
label_keys = [str(key) for key in HWlabeldict.keys()]
label_vals = list(HWlabeldict.values())
plt.bar(label_keys[-1:-15:-1], label_vals[-1:-15:-1])
plt.show()

plt.hist(heights)
plt.show()

plt.hist(widths)
plt.show()

plt.hist(RGBs)
plt.show()