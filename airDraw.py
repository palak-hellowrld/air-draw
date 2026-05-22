import cv2
import mediapipe as mp


mp_hands=mp.solutions.hands
mp_draw=mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)

points=[]

cap = cv2.VideoCapture(0)
cv2.namedWindow("Hand FX", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand FX", 1000, 720)
color=(0,255,0)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame,1)
    rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results=hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[index].classification[0].label
            if label=="Right":
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                distance=((((hand_landmarks.landmark[20].x)-(hand_landmarks.landmark[4].x))**(2))+(((hand_landmarks.landmark[20].y)-(hand_landmarks.landmark[4].y))**(2)))**(0.5)
                h, w, _ = frame.shape
                px = int(hand_landmarks.landmark[8].x * w)
                py = int(hand_landmarks.landmark[8].y * h)
                if distance<=0.5:
                    if not points:
                        points.append((px, py))
                    elif (((px-points[-1][0])**2)+((py-points[-1][1])**2))**0.5 <= 50:
                        points.append((px, py))
                    for i in range(len(points) - 1):
                        cv2.line(frame,points[i], points[i+1], color, 8)
                else:
                    points.clear()
            elif label=="Left":
                color_value = int(255 * hand_landmarks.landmark[8].y)
                color=(color_value, 0, 255 - color_value)

    
    cv2.imshow("Hand FX", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()