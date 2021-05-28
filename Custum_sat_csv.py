import cv2
import face_recognition
import os
import pickle
import time
from datetime import datetime
#import mysql.connector
import json
import csv
import requests

#mydb = mysql.connector.connect(host="localhost",user="root",password="123456",database="mydb")
#mycursor = mydb.cursor()

#gender lib's
MODEL_DIR = "./model_dir"
FACE_CLASSIFIER = cv2.CascadeClassifier(MODEL_DIR + '/haarcascade_frontalface_default.xml')
#CLASSIFIER = load_model(MODEL_DIR + './Emotion_Detection.h5')
#CLASS_LABELS = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_locations = []

FACE_PROTO = MODEL_DIR + "/opencv_face_detector.pbtxt"
FACE_MODEL = MODEL_DIR + "/opencv_face_detector_uint8.pb"
GENDER_PROTO = MODEL_DIR + "/gender_deploy.prototxt"
GENDER_MODEL = MODEL_DIR + "/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
#genderList = ['Male', 'Female']
genderList = [ 1 , 0 ]
faceNet = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
genderNet = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

#data_base for staff
KNOWN_FACES_DIR = 'known_faces_SAP'
#KNOWN_FACES_DIR = 'known_faces_innovation'
#KNOWN_FACES_DIR = 'known_faces_naviostreet'

UNKNOWN_FACES_DIR = 'unknown_faces'
TOLERANCE = 0.55
MODEL = 'hog'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model

skip_frame=2
total_frame=0
#cap = cv2.VideoCapture('D19_20210520174406.mp4')
cap = cv2.VideoCapture("rtsp://user:parol12345@10.250.6.118:554/cam/realmonitor?channel=1&subtype=0") ##sap

location = 'Vatan 2nd floor'
print('Loading known faces...')
known_faces = []
known_names = []
unknown_faces = []
unknown_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

for name in os.listdir(UNKNOWN_FACES_DIR):
    for filename in os.listdir(f"{UNKNOWN_FACES_DIR}/{name}"):
        encoding = pickle.load(open(f"{UNKNOWN_FACES_DIR}/{name}/{filename}", "rb"))
        unknown_faces.append(encoding)
        unknown_names.append(int(name))

if len(unknown_names) > 0:
     next_id = max(unknown_names) + 1
else:
     next_id = 0
#next_id = 0

print('Processing unknown faces...')
# Now let's loop over a folder of faces we want to label
while True:
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (800, 600))
    #frame = cv2.resize(frame, (800 , 600), fx = 10, fy = 10, interpolation = cv2.INTER_LINEAR)
    total_frame = total_frame + 1
    #frame = frame[400:1240, 0:1200]
    if (total_frame % skip_frame) == 0:

        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CLASSIFIER.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")
            left, bottom, right, top = x, y, x + w, y + h
            face = frame[bottom:top, left:right]
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            #if np.sum([roi_gray]) != 0:
             #   matches = []
              #  roi = roi_gray.astype('float') / 255.0
               # roi = img_to_array(roi)
                #roi = np.expand_dims(roi, axis=0)

                #predictions = CLASSIFIER.predict(roi)[0]
                #label = CLASS_LABELS[predictions.argmax()]
            locations = face_recognition.face_locations(face, model=MODEL)

            encodings = face_recognition.face_encodings(face, locations)

            for face_encoding, face_location in zip(encodings, locations):
                results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
                match = None
                if True in results:  # If at least one is true, get a name of first of found labels
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    match = known_names[results.index(True)]
                    print(f' - {match} from {results}')
                    # sql = "INSERT INTO Entry (customerID, staffID, Date, Emotion, Age, Gender, PictureLink) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                    # val = ("", match, current_time, 'label', '20', gender, 'link')
                    who = 0
                    cols = ['people_ai_id','who','object_code','gender','age','emotion','img_link']
                    objects = [{"people_ai_id":match,
                    "who":who,
                    "object_code":1,
                    "gender":gender,
                    "age":35,
                    "emotion":5,
                    "img_link": "link"
                    }]
                    path = "28_05_sap.csv"
                    with open (path, 'a') as f:
                        wr = csv.DictWriter(f,fieldnames=cols)
                        #wr.writeheader()
                        for word in objects:
                            wr.writerow(word)
                        wr.writerows(objects)
                    f.close()
                    data = json.dumps(objects,indent=4)
                    print(data)
                    r = requests.post("http://13.58.70.203:8000/visits/",json=data)
                    print(r.status_code)
                    #mycursor.execute(sql, val)
                else:
                    for face_encoding, face_location in zip(encodings, locations):
                        results_un = face_recognition.compare_faces(unknown_faces, face_encoding, TOLERANCE)
                        match = None
                        if True in results_un:  # If at least one is true, get a name of first of found labels
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            match = unknown_names[results_un.index(True)]
                            print(f' - {match} from {results_un}')
                            #sql = "INSERT INTO Entry (customerID, staffID, Date, Emotion, Age, Gender, PictureLink) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                            #val = (match, "", current_time, 'label', '20', gender, 'link')
                            #mycursor.execute(sql, val)
                            who = 1
                            cols = ['people_ai_id', 'who', 'object_code', 'gender', 'age', 'emotion', 'img_link']
                            objects = [{"people_ai_id":match,
                            "who":who,
                            "object_code":1,
                            "gender":gender,
                            "age":30,
                            "emotion":1,
                            "img_link":"link"
                                        }]
                            path = "28_05_sap.csv"
                            with open(path, 'a') as f:
                                wr = csv.DictWriter(f, fieldnames=cols)
                                #wr.writeheader()
                                for word in objects:
                                    wr.writerow(word)
                                wr.writerows(objects)
                            f.close()
                            data = json.dumps(objects,indent=4)
                            print(data)
                            r = requests.post("http://13.58.70.203:8000/visits/", json=data)
                            print(r.status_code)
                        else:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            match = str(next_id)
                            next_id += 1
                            unknown_names.append(match)
                            unknown_faces.append(face_encoding)
                            os.mkdir(f"{UNKNOWN_FACES_DIR}/{match}")
                            pickle.dump(face_encoding, open(f"{UNKNOWN_FACES_DIR}/{match}/{match}-{int(time.time())}.pkl", "wb"))
                            # sql q= "INSERT INTO Entry (customerID, staffID, Date, Emotion, Age, Gender, PictureLink) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                            # val = (match, "", current_time, 'label', '20', gender, 'link')
                            # mycursor.execute(sql, val)
                            who = 1
                            cols = ['people_ai_id', 'who', 'object_code', 'gender', 'age', 'emotion', 'img_link']
                            objects = [{"people_ai_id": match,
                                       "who": who,
                                       "object_code": 1,
                                       "gender": gender,
                                       "age":25,
                                       "emotion": 1,
                                       "img_link":"link"
                                       }]
                            path = "28_05_sap.csv"
                            with open(path, 'a') as f:
                                wr = csv.DictWriter(f,fieldnames=cols)
                                #wr.writeheader()
                                for word in objects:
                                    wr.writerow(word)
                                wr.writerows(objects)
                            f.close()
                            data = json.dumps(objects,indent=4)
                            print(data)
                            r = requests.post("http://13.58.70.203:8000/visits/", json=data)
                            print(r.status_code)
                label_position = (x, y - 10)
                cv2.putText(frame, str(match) + " ", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # mydb.commit()
        frame = cv2.resize(frame,(800,600))
        cv2.imshow('Customer Satisfaction', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()