import hashlib

sst_dev = open("data/ids-sst-dev.csv", "r").read().strip().split("\n")[1:]
sst_dev_label = {} # id to label
sst_sentence = {}
for i in sst_dev:
    id_ = i.split("\t")[1]
    sentence = i.split("\t")[2]
    label = int(i.split("\t")[3])
    sst_dev_label[id_] = label
    sst_sentence[id_] = sentence

pred = open("predictions/sst-dev-output.csv", "r").read().strip().split("\n")[1:]
sst_pred = {}
for i in pred:
	id_, label = i.split(" , ")
	sst_pred[id_] = int(label)
print(len(sst_dev_label), len(sst_pred))

diffs = {0:0, 1:0, 2:0, 3:0, 4:0}
wrong_classes = {0:0, 1:0, 2:0, 3:0, 4:0}
for id_ in sst_pred:
	if sst_pred[id_] != sst_dev_label[id_]:
		diff = abs(sst_pred[id_] - sst_dev_label[id_])
		diffs[diff] += 1
		wrong_classes[sst_dev_label[id_]] += 1
		print(sst_sentence[id_], sst_dev_label[id_], sst_pred[id_])
print(diffs)
print(wrong_classes)

para_dev = open("data/quora-dev.csv", "r").read().strip().split("\n")[1:]
para_dev_label = {} # id to label
para_sentence = {}
for i in para_dev:
    id_ = i.split("\t")[1]
    sentence = (i.split("\t")[2], i.split("\t")[3])
    print(i.split("\t"))
    label = float(i.split("\t")[4])
    para_dev_label[id_] = label
    para_sentence[id_] = sentence
print(para_dev_label)

pred = open("predictions/para-dev-output.csv", "r").read().strip().split("\n")[1:]
para_pred = {}
for i in pred:
	id_, label = i.split(" , ")
	para_pred[id_] = float(label)
print(len(para_dev_label), len(para_pred))

labels_wrong = {0.0:0, 1.0:0}
for id_ in para_pred:
	if para_pred[id_] != para_dev_label[id_]:
		print(para_sentence[id_], para_pred[id_], para_dev_label[id_])
		labels_wrong[para_dev_label[id_]] += 1
print(labels_wrong)

sts_dev = open("data/sts-dev.csv", "r").read().strip().split("\n")[1:]
sts_dev_label = {} # id to label
sts_sentence = {}
for i in sts_dev:
    id_ = i.split("\t")[1]
    sentence = (i.split("\t")[2], i.split("\t")[3])
    print(i.split("\t"))
    label = float(i.split("\t")[4])
    sts_dev_label[id_] = label
    sts_sentence[id_] = sentence
print(sts_dev_label)

pred = open("predictions/sts-dev-output.csv", "r").read().strip().split("\n")[1:]
sts_pred = {}
for i in pred:
	id_, label = i.split(" , ")
	sts_pred[id_] = float(label)
print(len(sts_dev_label), len(sts_pred))

def closest_multiple_of_0_1(value):
    rounded_value = round(value / 0.1) * 0.1
    return max(0, min(rounded_value, 5.0))

labels_wrong = {}
for i in range(0, 51):
	labels_wrong[i] = 0
for id_ in sts_pred:
	difference = abs(sts_pred[id_] - sts_dev_label[id_]) * 10
	labels_wrong[int(difference)] += 1
	if int(difference) > 20:
		print(sts_sentence[id_], sts_pred[id_], sts_dev_label[id_])
print(labels_wrong)

lin_dev = open("cola_public/raw/in_domain_dev.tsv", "r").read().strip().split("\n")[1:]
lin_dev_label = {} # id to label
lin_sentence = {}
c = 0
for i in lin_dev:
    sentence = i.split("\t")[3]
    id_ = hashlib.md5(sentence.encode()).hexdigest()
    label = int(i.split("\t")[1])
    lin_dev_label[id_] = label
    lin_sentence[id_] = sentence
    c += 1
print(lin_dev_label)

pred = open("predictions/lin-dev-output.csv", "r").read().strip().split("\n")[1:]
lin_pred = {}
for i in pred:
	id_, label = i.split(" , ")
	lin_pred[id_] = int(float(label))
print(len(lin_dev_label), len(lin_pred))

wrong = {0:0, 1:0}
for i in lin_dev_label:
	if lin_pred[i] != lin_dev_label[i]:
		print(lin_sentence[i], lin_pred[i], lin_dev_label[i])
		wrong[lin_dev_label[i]] += 1
print(wrong)