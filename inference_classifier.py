import pickle

import torch

import cv2
import mediapipe as mp
import numpy as np

import tut_1

from torchvision.datasets import ImageFolder


# model_dict = pickle.load(open('./model.p', 'rb'))
# model_dict = pickle.load(open('./nn_model.pt', 'rb'))
# model_dict = torch.load("nn_model.pt")
model = tut_1.ASLClassifier()
# model = torch.load("./nn_model.pt")
model.load_state_dict(torch.load("./nn_model.pt"))
model.eval()
# model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data_dir = './ASL_Alphabet_Dataset/asl_alphabet_train'
target_to_class = {k: v for v, k in ImageFolder(data_dir).class_to_idx.items()}

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

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

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        if len(data_aux) != 42:
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
            continue

        # prediction = model.predict([np.asarray(data_aux)])
        # prediction = model(torch.from_numpy(data_aux).float())
        # X = torch.rand(1, 42)



        with torch.no_grad():
            output = model(torch.tensor(data_aux))
            probabilities = torch.nn.functional.softmax(output, dim=0)
            probabilities = probabilities.cpu().numpy().flatten()

        prediction = np.argmax(probabilities)

        char = target_to_class[prediction]

        print(char)

        # predicted_character = predictions[0]
        # print(predicted_character)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        # cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
        #             cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
