import sys
import numpy as np
import cv2
import dlib
from imutils import face_utils
from mtcnn import MTCNN

import time

face_classifier = cv2.CascadeClassifier('opencv_haar_models/haarcascade_frontalface_default.xml')
dlib_hog_detector = dlib.get_frontal_face_detector()
# dlib_predictor = dlib.shape_predictor('dlib_models/shape_predictor_5_face_landmarks.dat')
dlib_dnn_detector = dlib.cnn_face_detection_model_v1("dlib_models/mmod_human_face_detector.dat")
cv_dnn_detector = cv2.dnn.readNetFromCaffe("opencv_dnn_models/deploy.prototxt", "opencv_dnn_models/res10_300x300_ssd_iter_140000.caffemodel")
#cv_dnn_detector = cv2.dnn.readNetFromTensorflow("opencv_dnn_models/opencv_face_detector.pbtxt", "opencv_dnn_models/opencv_face_detector_uint8.pb")
mtcnn_detector = MTCNN()

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

tt = time.perf_counter()
counter = 0
fps_all = 0.0
fps_det = 0.0
num_faces = 0
method = 4

while(True):
  t0 = time.perf_counter()
  ret, img_raw = capture.read()

  t1 = time.perf_counter()
  if method == 1:
    img = cv2.resize(img_raw, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.05,
              minNeighbors=1, minSize=(20,20), flags=cv2.CASCADE_SCALE_IMAGE)
    num_faces = len(faces)
    for (x, y, w, h) in faces:
      cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

  elif method == 2:
    img = cv2.resize(img_raw, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = dlib_hog_detector(rgb, 1)
    num_faces = len(dets)
    for (i, d) in enumerate(dets):
      cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 255, 255), 3)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # shape = dlib_predictor(gray, rect)
    # shape = face_utils.shape_to_np(shape)
    # for (x, y) in shape:
    #   cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

  elif method == 3:
    img = cv2.resize(img_raw, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = dlib_dnn_detector(img, 0)
    num_faces = len(rects)
    for (i, rect) in enumerate(rects):
      x1 = rect.rect.left()
      y1 = rect.rect.top()
      x2 = rect.rect.right()
      y2 = rect.rect.bottom()
      cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 3)

  elif method == 4:
    img = cv2.resize(img_raw, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    blob = cv2.dnn.blobFromImage(cv2.resize(img_raw, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    cv_dnn_detector.setInput(blob)
    faces = cv_dnn_detector.forward()
    num_faces = len(faces)
    for i in range(0, faces.shape[2]):
      confidence = faces[0, 0, i, 2]
      if confidence > 0.3:
        h, w = img.shape[:2]
        box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

  elif method == 5:
    img = cv2.resize(img_raw, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    faces = mtcnn_detector.detect_faces(img)
    num_faces = len(faces)
    for result in faces:
      x, y, w, h = result['box']
      cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

  t2 = time.perf_counter()
  fps_det = 1/(t2-tt)
  fps_all = 1/(t2-t1)
  tt = t2
  counter = counter + 1

  result = f'#{method}: {num_faces} face(s), {fps_det:.2f} {fps_all:.2f} hz, {img.shape[1]} x {img.shape[0]}'
  cv2.putText(img, result, (2,10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
  print(result)

  cv2.imshow('face-detecctor', img)
  ch = cv2.waitKey(1) & 0xFF

  if ch == ord('1'):
    method = 1
  elif ch == ord('2'):
    method = 2
  elif ch == ord('3'):
    method = 3
  elif ch == ord('4'):
    method = 4
  elif ch == ord('5'):
    method = 5
  elif ch == ord('q'):
    break

capture.release()
cv2.destroyAllWindows()

