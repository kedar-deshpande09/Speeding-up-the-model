import face_recognition
import docopt
from sklearn import svm
import os
import timeit
import pickle

def face_train(dir):
    # Training the SVC classifier
    # The training data would be all the
    # face encodings from all the known
    # images and the labels are their names
    # print("Directory Name::", os.path.basename(os.path.normpath(dir)))
    groupId = str(os.path.basename(os.path.normpath(dir)))
    encodings = []
    names = []
    print("Directory::", dir)
    # Training directory
    if dir[-1]!='\\':
        dir += '\\'
    train_dir = os.listdir(dir)

    # Loop through each person in the training directory
    for person in train_dir:
        pix = os.listdir(dir + person)

        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file(
                dir + person + "\\" + person_img)
            face_bounding_boxes = face_recognition.face_locations(face)

            # If training image contains exactly one face
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image
                # with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)
            else:
                print(person + "\\" + person_img + " can't be used for training")

    # Create and train the SVC classifier
    clf = svm.SVC(gamma ='scale')
    clf.fit(encodings, names)
    modelPath = "C:\\Users\\Kedar\\Desktop\\Picture\\"+groupId + ".pkl"
    pickle.dump(clf, open(modelPath, 'wb'))

    # # Load the test image with unknown faces into a numpy array
    # test_image = face_recognition.load_image_file(test)

    # # Find all the faces in the test image using the default HOG-based model
    # face_locations = face_recognition.face_locations(test_image)
    # no = len(face_locations)
    # print("Number of faces detected: ", no)

    # # Predict all the faces in the test image using the trained classifier
    # print("Found:")
    # stop = timeit.default_timer()
    # # print('Time: ', stop - start)
    # for i in range(no):
    #     start = timeit.default_timer()
    #     test_image_enc = face_recognition.face_encodings(test_image)[i]
    #     name = clf.predict([test_image_enc])
    #     # print(*name)
    #     userids.append(*name)
    #     stop = timeit.default_timer()
    #     # print('Time: ', stop - start)  

    # return userids