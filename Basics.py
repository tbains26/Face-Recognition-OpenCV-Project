import numpy as np
import cv2
import face_recognition

# Load and convert the main/test image
img_Mark=face_recognition.load_image_file("ImagesBasic/Mark_Zuckerberg.jpg")
img_Mark=cv2.cvtColor(img_Mark,cv2.COLOR_BGR2RGB)
img_Test=face_recognition.load_image_file("ImagesBasic/Mark_Zuckerberg_test.jpg")
img_Test=cv2.cvtColor(img_Test,cv2.COLOR_BGR2RGB)

# Find face and encode in main/test image
face_locations = face_recognition.face_locations(img_Mark)[0]
encode_Mark=face_recognition.face_encodings(img_Mark)[0]
cv2.rectangle(img_Mark,(face_locations[3],face_locations[0]),(face_locations[1],face_locations[2]),(255,0,0),2)


face_locations_Test = face_recognition.face_locations(img_Test)[0]
encode_Test=face_recognition.face_encodings(img_Test)[0]
cv2.rectangle(img_Test,(face_locations_Test[3],face_locations_Test[0]),(face_locations_Test[1],face_locations_Test[2]),(255,0,0),2)

# Compare both encodings to get distance score
outcome=face_recognition.compare_faces([encode_Mark],encode_Test)
face_Distance=face_recognition.face_distance([encode_Mark],encode_Test)

# Print results
print(outcome, face_Distance)

# Display the results on test image
cv2.putText(img_Test,f"{outcome} {round(face_Distance[0],2)}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)


# Display images
cv2.imshow("Mark_Zuckerberg",img_Mark)
cv2.imshow("Mark Test",img_Test)
cv2.waitKey(0)
