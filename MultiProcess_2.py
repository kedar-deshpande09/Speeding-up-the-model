from sys import maxsize
import face_recognition
import docopt
from sklearn import svm
import os
import timeit
import pickle
import cv2
import multiprocessing 
import time
import numpy as np
from multiprocessing import Pool

dir = "C:\\Users\\Kedar\\Desktop\\Picture\\1"
test = "C:\\Users\\Kedar\\Desktop\\MassFace\\9people.jpeg"
groupId = str(os.path.basename(os.path.normpath(dir)))
print("GroupID=", groupId)
print("Dir=", dir)
userids = []
filename = "C:\\Users\\Kedar\\Desktop\\Picture\\1\\"+groupId + ".pkl"
if filename is None:
    print("No model found")
    print(userids)
clf = pickle.load(open(filename, 'rb'))
print('C1')
test_image = face_recognition.load_image_file(test)
test_image = cv2.resize(test_image,(500,500))
face_locations = face_recognition.face_locations(test_image)
no = len(face_locations)
print("Number of faces detected: ", no)
list_no = []
for i in range(no):
    list_no.append(i)
print("Found:")

global faceRecognizer_
def faceRecognizer_(j):
    test_image_enc = face_recognition.face_encodings(test_image)[j]
    name = clf.predict([test_image_enc])
    userids.append(*name)
    return userids
if __name__ == '__main__':
    start = timeit.default_timer()
    print('Pool Check-1')
    with Pool(10) as p:
            print(p.map(faceRecognizer_, list_no))
            print('Pool Check-2')
            p.close()
            p.join()
    stop = timeit.default_timer()
    print('Time: ', stop - start)