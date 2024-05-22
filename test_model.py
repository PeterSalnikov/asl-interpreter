import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.01)

DATA_DIR = './ASL_Alphabet_Dataset/asl_alphabet_test'

data = []
# labels = []
# for dir_ in os.listdir(DATA_DIR):
#     if dir_ == '.DS_Store':
#         continue
frame = 'test'

for img_path in os.listdir(DATA_DIR):
    data_aux = []

    x_ = []
    y_ = []

    img = cv2.imread(os.path.join(DATA_DIR, img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        # visualize hand landmarks to manually confirm correctness
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        cv2.imshow(img, frame)
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        data.append(data_aux)
        # labels.append(dir_)

print(len(data))

f = open('test_data.pickle', 'wb')
pickle.dump({'data': data}, f)
f.close()
