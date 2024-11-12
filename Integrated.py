import cv2
import numpy as np
from tensorflow import keras

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.426, 87.769, 114.896)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

new_model = keras.models.load_model("my_model.h5")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

padding = 20

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        
        eyes = eyeCascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:

            eye_roi = roi_color[ey:ey + eh, ex:ex + ew]

            if eye_roi.size > 0:
                final_image = cv2.resize(eye_roi, (224, 224))
                final_image = np.expand_dims(final_image, axis=0)
                final_image = final_image / 255.0
                predictions = new_model.predict(final_image)
                status = "Sober" if predictions > -3 else "Intoxicated"
                # cv2.putText(frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Draw intoxication status below the face
                cv2.putText(frame, status, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                
            blob = cv2.dnn.blobFromImage(roi_color, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            genderNet.setInput(blob)
            genderPred = genderNet.forward()
            gender = genderList[genderPred[0].argmax()]

            ageNet.setInput(blob)
            agePred = ageNet.forward()
            age_index = agePred[0].argmax()
            age_category = ageList[age_index]
            label = f"{gender}, {age_category}"

            if age_category.startswith('(0-2)') or age_category.startswith('(4-6)') or age_category.startswith('(8-12)') or age_category.startswith('(15-20)'):
                box_color = (0, 0, 255)  # Red color for warning (underage)
            else:
                box_color = (0, 255, 0)  # Green color for normal

            cv2.rectangle(frame, (x, y - 30), (x + w, y), box_color, -1)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Capture', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
