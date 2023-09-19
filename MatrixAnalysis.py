import os 
import numpy as np
import matplotlib.pyplot as plt

result1 = np.load("./cosinetest/output/young/Main4_Facenet_yolov8_cosinedist.npy")

enrol_path = './cosinetest/enrolment_young/'
probe_path = './cosinetest/probe/'

enrol_filename = os.listdir(enrol_path)
probe_filename = os.listdir(probe_path)
NRIC_enrol = []
for i in enrol_filename:
    NRIC_enrol.append(int(i.split('_')[0]))

NRIC_probe = []
for i in probe_filename:

    NRIC_probe.append(int(i.split('_')[0]))

enrolname = np.array(NRIC_enrol)
probename = np.array(NRIC_probe)

a = np.tile(enrolname,probename.shape[0])
b = np.repeat(probename,enrolname.shape[0])


ground_truth = a==b


c=1-result1.flatten() ########

num_true = np.count_nonzero(ground_truth== True)
p = np.where(c<0.2)


true_readings = []
false_positive = []
false_negative = []

#threshold = [0,0.2,0.4,0.6,0.8,2] #######
threshold = []
i=0
while i<1:
    threshold.append(i)
    i+=0.01

true_pos = [num_true, num_true , num_true, num_true, num_true,num_true]

for val in threshold:
    result = c>val ########
    readings = ground_truth.astype(int) - result.astype(int)
    false_positive.append(np.count_nonzero(readings<0)/ground_truth.shape[0]*100)
    print(val)
    print('False Positve = ', np.count_nonzero(readings<0)/ground_truth.shape[0]*100)
    print('False Negative = ', np.count_nonzero(readings>0)/num_true*100)
    false_negative.append(np.count_nonzero(readings>0)/num_true*100)


fig, ax = plt.subplots()

#ax.plot(false_negative[10:20], false_positive[10:20])
ax.plot(false_negative[3:72], false_positive[3:72])
print(len(false_positive),len(false_negative))
#fig.legend()
'''
ax.plot(threshold, false_positive,label='False Positive')
ax.plot(threshold, true_pos, label='True Positive')

#plt.rcParams["figure.figsize"] = [7.50, 3.50]
#plt.rcParams["figure.autolayout"] = True
line1, = plt.plot(false_negative, label="False Negative")
line2, = plt.plot(false_positive, label="False Positive")
line2, = plt.plot(true_pos, label="True Positive")
leg = plt.legend(loc='upper center')
'''
plt.title('Facenet Cosine Similarity')
plt.xlabel('False Negative %')
plt.ylabel('False Positive %')
plt.autoscale()
plt.show()