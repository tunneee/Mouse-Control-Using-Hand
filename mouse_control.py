import cv2
import mediapipe as mp
import autopy
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

screen_width, screen_height = autopy.screen.size()

cv2.namedWindow("Finger Control", cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            thumb_x = int(thumb.x * screen_width)
            thumb_y = int(thumb.y * screen_height)
            index_finger_x = int(index_finger.x * screen_width)
            index_finger_y = int(index_finger.y * screen_height)

            autopy.mouse.move(index_finger_x, index_finger_y)

            if thumb_x - index_finger_x < 10 and thumb_y - index_finger_y < 10:
                autopy.mouse.click()

    cv2.imshow("Finger Control", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()