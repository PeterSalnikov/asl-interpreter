import os

import cv2


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

fps = 30

# We need to set resolutions. 
# so, convert them from float to integer. 

cap = cv2.VideoCapture(1)
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 

size = (frame_width, frame_height) 
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    result = cv2.VideoWriter(os.path.join(DATA_DIR, str(j), '{}.avi'.format(j)), 
                    cv2.VideoWriter_fourcc(*'MJPG'), 
                    fps, size) 
    
    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press " " ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord(' '):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        result.write(frame) 

        cv2.waitKey(12)
        # cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1
    

cap.release()
cv2.destroyAllWindows()
