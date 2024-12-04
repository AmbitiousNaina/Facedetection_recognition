import os
import pickle
import cvzone
import cv2
import face_recognition
import numpy as np

# setting up the webcam
cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)
imgBg = cv2.imread('Resources/background.png')

# importing the mode imgs into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

print(len(imgModeList))

# Load the encoding file

print("Loading encode file")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print(studentIds)
print("Encoded file loaded")

studentInfo = {
    111111: "Sunayana",
    852741: "Elina",
    963852: "Elon Musk"
}


while True:

    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

# detecting faces in realtime
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBg[162:162+480, 55:55+640] = img
    imgBg[44:44 + 633, 808:808 + 414] = imgModeList[3]

# matching the deteced face with known face
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print("Matches", matches)
        # print("Face Dist", faceDist)
        matchIndex = np.argmin(faceDist)
        # print("Match Index", matchIndex)

# display the result
        if matches[matchIndex]:
            print("Known Face Detected!")

            student_id = studentIds[matchIndex]
            student_name = studentInfo.get(int(student_id), "Unknown Student")
            print(f"Student ID: {student_id}, Name: {student_name}")


        # print(studentIds[matchIndex])

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
        bbox = 55+x1, 162+y1, x2-x1, y2-y1
        imgBg = cvzone.cornerRect(imgBg, bbox, rt=0)



    # cv2.imshow("Webcam Frame", img)
    cv2.imshow("Face Attendance", imgBg)
    cv2.waitKey(1)


