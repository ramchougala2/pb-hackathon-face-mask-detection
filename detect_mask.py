import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('mask_detector_model.h5')

# Class order - MUST match train_generator.class_indices from Colab
# e.g. if it printed {'WithMask': 0, 'WithoutMask': 1}, keep this order
class_names = ['WithMask', 'WithoutMask']

# Load OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        # Crop the face region
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (224, 224))
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        # Predict
        prediction = model.predict(face_img, verbose=0)
        class_index = np.argmax(prediction)
        confidence = prediction[0][class_index]
        label = class_names[class_index]

        # Green box for mask, red for no mask
        color = (0, 255, 0) if label == 'WithMask' else (0, 0, 255)

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        text = f"{label}: {confidence*100:.1f}%"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Face Mask Detector - Press Q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
