import cv2, os

haar_file = r"C:\Users\smriti\Downloads\30 DAYS MACHINE LEARNING\FaceRecognitionusingMLClassifier\FaceRecognitionusingMLClassifier\haarcascade_frontalface_default.xml"
datasets = 'Dataset'
sub_data = r'C:\Users\smriti\Downloads\steve\Nnamiii'

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

(width, height) = (130, 100)

face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)  # Use default camera (index 0)

# Check if the webcam is opened successfully
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

count = 1
while count < 31:
    print(count)
    (_, im) = webcam.read()
    
    if im is None:
        print("Error: Could not read frame from webcam.")
        continue
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('%s/%s.png' % (path, count), face_resize)
        count += 1

    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27:  # Exit on ESC key
        break

webcam.release()
cv2.destroyAllWindows()
