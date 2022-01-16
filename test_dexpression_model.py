import cv2
import keras.models
import numpy as np

emotion_dict = {0: 'fear',
                1: 'anger',
                2: 'disgust',
                3: 'happiness',
                4: 'neutral',
                5: 'sadness',
                6: 'surprise'}
model = keras.models.load_model('./model_dexpression_based_2.h5')

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    frame = cv2.resize(frame, (1280, 720))

    if not ret:
        break

    face_detector = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
        roi_frame = frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_frame, (224, 224)), -1), 0)

        # predict the emotions
        emotion_prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
