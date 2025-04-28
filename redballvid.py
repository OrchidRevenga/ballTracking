import cv2
import numpy as np
import sys

cap = cv2.VideoCapture("rgb_ball_720.mp4")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red1_low = np.array([0, 120, 70])
    red1_up = np.array([10, 255, 255])
    red2_low = np.array([170, 120, 70])
    red2_up = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, red1_low, red1_up)
    mask2 = cv2.inRange(hsv, red2_low, red2_up)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)

        cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)

        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(frame, "Red Ball", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Red Ball", frame)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()
