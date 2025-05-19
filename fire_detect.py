import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('fire_detection_model.h5')

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to match model input size (128x128)
    img = cv2.resize(frame, (128, 128))
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize

    # Predict
    prediction = model.predict(img)[0][0]

    # Display Result
    if prediction > 0.5:
        cv2.putText(frame, "🔥 FIRE DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("🔥 ALERT! Fire detected!")

    cv2.imshow("Fire Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
