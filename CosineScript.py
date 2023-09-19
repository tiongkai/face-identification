import numpy as np
import time
import csv

import time
import os
from deepface.commons import functions, realtime, distance as dst
from deepface import DeepFace



run = 'Test11'
model_name = 'Facenet'
detector_backend = 'yolov8'

output_dir = './cosinetest/output/test/'

enrol_path = './cosinetest/enrolment_mini/'
probe_path = './cosinetest/probe_mini/'

enrol_filename = os.listdir(enrol_path)
probe_filename = os.listdir(probe_path)


field_names = ['Img1','Img2','Cosine Distance']

'''
outputfile = './cosinetest/output/test/{set}_results.csv'.format(set= run+"_"+model_name+"_"+detector_backend)
isExist = os.path.exists(outputfile)
if not isExist:

# Create a new directory because it does not exist
    with open(outputfile, 'w') as csv_file:
        dict_object = csv.writer(csv_file)
        dict_object.writerow(field_names)
'''

enrol_mat = []
probe_mat = []

start = time.time()

for i in probe_filename:
    #t = time.time()
    temp = []
    target_size = functions.find_target_size(model_name=model_name)
    # img pairs might have many faces
    img1_objs = functions.extract_faces(
        img=probe_path+i,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=False,
        align=True,)
    #print("Extract Face time - ", time.time()-t)
    for img1_content, img1_region, _ in img1_objs:
        #tt=time.time()
        img1_embedding_obj = DeepFace.represent(
        img_path=img1_content,
        model_name=model_name,
        enforce_detection=False,
        detector_backend="skip",
        align=True,
        normalization='base')
        #print(img1_embedding_obj[0]["embedding"])
        temp.append(img1_embedding_obj[0]["embedding"])
        #print("Embedding Face time - ", time.time()-tt)
        #np.hstack((enrol_mat,np.array(img1_embedding_obj[0]["embedding"],ndmin=2)))
    probe_mat.append(min(temp))

print('Probe time : ', time.time()-start)

enrol_start = time.time()

for j in enrol_filename:
    #t = time.time()
    temp = []
    target_size = functions.find_target_size(model_name=model_name)
    # img pairs might have many faces
    img1_objs = functions.extract_faces(
        img=enrol_path+j,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=False,
        align=True,)
    #print("Extract Face time - ", time.time()-t)
    for img1_content, img1_region, _ in img1_objs:
        #tt=time.time()
        img1_embedding_obj = DeepFace.represent(
        img_path=img1_content,
        model_name=model_name,
        enforce_detection=False,
        detector_backend="skip",
        align=True,
        normalization='base')
        temp.append(img1_embedding_obj[0]["embedding"])
        #np.hstack((enrol_mat,np.array(img1_embedding_obj[0]["embedding"],ndmin=2)))
    #if i == '000215954_11102018_5_001.jpg':
        #print(img1_embedding_obj[0]["embedding"])
        #print("Embedding Face time - ", time.time()-tt)
    enrol_mat.append(min(temp))
print('Enrol time : ', time.time() - enrol_start)

'''
for i in probe_filename:
    for j in enrol_filename:
        with open(outputfile, 'a') as csv_file:
            dict_object = csv.writer(csv_file) 
            output = [i,j]
            dict_object.writerow(output)
'''

#Calculation of the cosine distance 

calc_start = time.time()
probe = np.array(probe_mat)
print(probe.shape)
enrol = np.array(enrol_mat)
print(enrol.shape)

unit_probe = probe / np.linalg.norm(probe,axis = 1).reshape(probe.shape[0],1)
unit_enrol = enrol / np.linalg.norm(enrol,axis = 1).reshape(enrol.shape[0],1)
result_mat = np.dot(unit_probe,unit_enrol.T)
cos_diff = 1- result_mat
print('Calculation time : ', time.time() - calc_start)

print(cos_diff.flatten().shape)
'''
for p in list(cos_diff.flatten()):
    with open(outputfile, 'a') as csv_file:
        dict_object = csv.writer(csv_file) 
        dict_object.writerow([p])

'''

np.save(output_dir+"{run_no}_probe".format(run_no=run+"_"+model_name+"_"+detector_backend),probe)
np.save(output_dir+"{run_no}_enrol".format(run_no=run+"_"+model_name+"_"+detector_backend),enrol)
np.save(output_dir+"{run_no}_cosinedist".format(run_no=run+"_"+model_name+"_"+detector_backend),cos_diff)
print('Total Time : ', time.time() - start)
