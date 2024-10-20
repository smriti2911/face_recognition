import cv2
import numpy as np
import os

# Path to Haar cascade XML file
haar_file = r"C:\Users\smriti\Downloads\30 DAYS MACHINE LEARNING\FaceRecognitionusingMLClassifier\FaceRecognitionusingMLClassifier\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

# Path to dataset
datasets = r"C:\Users\smriti\Downloads\steve"
print('Training...')

# Initialize variables
(images, labels, names, id) = ([], [], {}, 0)

# Read dataset and preprocess
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        print(f"Processing {subdir}")
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)  # Use os.path.join for cross-platform support
            print(f"Reading {path}")
            img = cv2.imread(path, 0)  # Read in grayscale
            if img is not None:  # Check if image was read correctly
                try:
                    images.append(cv2.resize(img, (130, 100)))  # Resize images
                    labels.append(int(id))
                except Exception as e:
                    print(f"Error resizing image {filename}: {e}")
            else:
                print(f"Failed to load image: {path}")
        id += 1

# Check if training data is loaded
if len(images) == 0:
    print("No training data found.")
else:
    (images, labels) = [np.array(lis) for lis in [images, labels]]
    print(f"Training data ready: {images.shape}, {labels.shape}")

    # Create the LBPH face recognizer model
    model = cv2.face.LBPHFaceRecognizer_create()

    # Train the model with loaded images and labels
    model.train(images, labels)

    # Start the webcam for face recognition
    webcam = cv2.VideoCapture(0)  # Use default camera (change 0 to 1 or 2 if using external camera)
    cnt = 0

    while True:
        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces in the frame
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 2)  # Draw rectangle around face
            face = gray[y:y + h, x:x + w]  # Extract the face region
            face_resize = cv2.resize(face, (130, 100))  # Resize face to match training data

            # Predict the person
            prediction = model.predict(face_resize)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

            if prediction[1] < 800:
                cv2.putText(im, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
                print(names[prediction[0]])
                cnt = 0
            else:
                cnt += 1
                cv2.putText(im, 'Unknown', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                if cnt > 100:
                    print("Unknown Person")
                    cv2.imwrite("unKnown.jpg", im)  # Save image of unknown person
                    cnt = 0

        cv2.imshow('FaceRecognition', im)  # Display the output in a window
        key = cv2.waitKey(10)
        if key == 27:  # Exit when 'ESC' is pressed
            break

    # Release resources
    webcam.release()
    cv2.destroyAllWindows()
