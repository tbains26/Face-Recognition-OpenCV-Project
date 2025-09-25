import os
import numpy as np
import cv2
import face_recognition

path="ImagesAttendance"

# store loaded images with corresponding names
images=[]
names=[]


file_list=os.listdir(path)
print(file_list)


# Load images and extract names
for current_name in file_list:
    current_img=cv2.imread(path+"/"+current_name)
    images.append(current_img)

    names.append(os.path.splitext(current_name)[0])

print(names)

# Get encodings for all images
def get_Encoding(image):
    encode_list = []

    for people in images:
        people=cv2.cvtColor(people,cv2.COLOR_BGR2RGB)

        encode = face_recognition.face_encodings(people)[0]
        encode_list.append(encode)

    return encode_list

encode_list=get_Encoding(images)

# Encode known faces
known_faces_encode=get_Encoding(images)
print("completed encoding")

# Start the webcam
capture=cv2.VideoCapture(0)

while True:
    complete, people=capture.read()
    small_img=cv2.resize(people,(0,0),None,0.25,0.25)
    small_img=cv2.cvtColor(small_img,cv2.COLOR_BGR2RGB)

    # Detect/encode faces
    faceFrame=face_recognition.face_locations(small_img)
    encodeFrame=face_recognition.face_encodings(small_img,faceFrame)

    # Compare detected faces with known faces
    for encode_Face,face_Location in zip(encodeFrame,faceFrame):
        match=face_recognition.compare_faces(known_faces_encode,encode_Face)
        face_Dis=face_recognition.face_distance(known_faces_encode,encode_Face)
        print(face_Dis)
        matched_Face=np.argmin(face_Dis)

        if match[matched_Face]:
            person=names[matched_Face].upper()
            print(person)

    # Display webcam
    cv2.imshow("Webcam",people)
    cv2.waitKey(1)
