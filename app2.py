from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
from tensorflow import keras

app = Flask(__name__)

# Load the pre-trained intoxication detection model
new_model = keras.models.load_model("my_model.h5")

# Load the Haar cascades for face and eye detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load face, age, and gender detection models
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

# Placeholder for camera object and status
camera = None
is_camera_on = False

def generate_frames():
    global camera
    while is_camera_on:
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            
            # Detect eyes in the ROI
            eyes = eyeCascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                eye_roi = roi_color[ey:ey + eh, ex:ex + ew]
                
                # Process only if eye_roi is not empty
                if eye_roi.size > 0:
                    final_image = cv2.resize(eye_roi, (224, 224))
                    final_image = np.expand_dims(final_image, axis=0)
                    final_image = final_image / 255.0
                    
                    # Predict intoxication status
                    predictions = new_model.predict(final_image)
                    status = "Sober" if predictions > -3 else "Intoxicated"
                    
                    # Draw intoxication status below the face
                    cv2.putText(frame, status, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Create a blob from the region of interest for gender and age prediction
                blob = cv2.dnn.blobFromImage(roi_color, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                
                # Predict gender
                genderNet.setInput(blob)
                genderPred = genderNet.forward()
                gender = genderList[genderPred[0].argmax()]
                
                # Predict age
                ageNet.setInput(blob)
                agePred = ageNet.forward()
                age_index = agePred[0].argmax()
                age_category = ageList[age_index]
                label = f"{gender}, {age_category}"
                
                # Choose color for label box based on age
                if age_category.startswith('(0-2)') or age_category.startswith('(4-6)') or age_category.startswith('(8-12)') or age_category.startswith('(15-20)'):
                    box_color = (0, 0, 255)  # Red color for underage warning
                else:
                    box_color = (0, 255, 0)  # Green color for normal
                
                # Draw colored rectangle for age and gender label
                cv2.rectangle(frame, (x, y - 30), (x + w, y), box_color, -1)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Encode the frame in JPEG format and yield as a byte stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Redirect to home page after login
        return redirect(url_for('home'))
    else:
        return render_template('login.html')

@app.route('/home')
def home():
    # Render the home page
    return render_template('home.html')

@app.route('/signup')
def signup():
    # Render the sign-up page
    return render_template('signup.html')

@app.route('/index')
def index():
    # Render the index page
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control_camera():
    global camera, is_camera_on
    if request.form.get('action') == 'start':
        camera = cv2.VideoCapture(0)
        is_camera_on = True
    elif request.form.get('action') == 'stop':
        if camera is not None:
            camera.release()
        is_camera_on = False
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='127.0.0.1', port=5000)  uncomment if ip not found
    # print("http://127.0.0.1:5000/")
