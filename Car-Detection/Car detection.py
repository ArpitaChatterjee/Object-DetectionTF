#step1-> get a lot of car images to train the model
#step2-> convert the img into grayscale for easier detection(makes algo faster n run ample of data)
#step3-> train the algo to detect cars

#comp train usinf haar features-> edge fea., line features, four-rectangle features. it converts the img into grayscale
#and compares the img with the features and one wiht highest probability is taken as true. 
#if the drk region in the img matches with the haar fea. along with the lighter ones than the probability of 
#recognizing the obj increases.  

import cv2
from random import randrange

#pre-trained opencv img
classifier_file= '/content/drive/MyDrive/Colab Notebooks/Car&Pedestrian detection /car_detector.xml'

#create opencv img
img= cv2.imread('/content/drive/MyDrive/Colab Notebooks/Car&Pedestrian detection /carimg.jpeg')

from google.colab.patches import cv2_imshow
#display img wiht the car spoted
cv2_imshow(img)
cv2.waitKey()

gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray_img)
cv2.waitKey()

#create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#detect car using detect multiscale
cars = car_tracker.detectMultiScale(gray_img)

print(cars)

for (x,y,w,h) in cars:
      cv2.rectangle(img, (x,y), (x+w, y+h), (randrange(255),randrange(255),randrange(255)), 2)

cv2_imshow(img)
cv2.waitKey()


img_f='/content/drive/MyDrive/Colab Notebooks/Car&Pedestrian detection /car.jpg'
imgg = cv2.imread(img_f)
g_imgg= cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
cars2= car_tracker.detectMultiScale(g_imgg, 1.1, 1)
print(cars2)

for (x,y,w,h) in cars2:
  cv2.rectangle(imgg, (x,y), (x+w, y+h),  (0,0,255), 2 )

cv2_imshow(imgg)
cv2.waitKey()



#VIDEO Version
vidsrc= cv2.VideoCapture('/content/drive/MyDrive/Colab Notebooks/Car&Pedestrian detection /video1.avi')

#run untill car stops
while True:
  (read_success, vid) = vidsrc.read()

  #SAFE CODE
  if read_success:
    gray = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
  else:
    break
   #detection test
  vidtest = car_tracker.detectMultiScale(gray, 1.1, 1) 
  print(vidtest)

  for (x,y, w,h) in vidtest:
    cv2.rectangle(vid, (x,y), (x+w, y+h), (randrange(255), randrange(255), randrange(255)), 2)

  cv2_imshow(vid)
  cv2.waitKey(1) 

cv2.destroyAllWindows()
