import cv2
import mediapipe as mp


class FingerCounter:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        self.tip_ids = [4, 8, 12, 16, 20]  # Landmark IDs

    def count_fingers(self, image):
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        image.flags.writeable = True

        fingers_count = 0
        hand_data = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                landmarks = hand_landmarks.landmark
                hand_fingers = []

                # Determine if it's a left or right hand
                if landmarks[17].x > landmarks[5].x:
                    hand_type = "Right"
                else:
                    hand_type = "Left"

                # Thumb
                if hand_type == "Right":
                    if landmarks[self.tip_ids[0]].x < landmarks[2].x:
                        fingers_count += 1
                        hand_fingers.append(1)
                    else:
                        hand_fingers.append(0)
                else:  # Left Hand
                    if landmarks[self.tip_ids[0]].x > landmarks[2].x:
                        fingers_count += 1
                        hand_fingers.append(1)
                    else:
                        hand_fingers.append(0)

                # Other fingers
                for id in range(1, 5):
                    if landmarks[self.tip_ids[id]].y < landmarks[self.tip_ids[id] - 2].y:
                        fingers_count += 1
                        hand_fingers.append(1)
                    else:
                        hand_fingers.append(0)

                hand_data.append({"hand_type": hand_type, "fingers": hand_fingers})

        return fingers_count, hand_data


def main():
    cap = cv2.VideoCapture(0)  #Default camera
    finger_counter = FingerCounter()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)

        fingers_count, hand_data = finger_counter.count_fingers(image)

        # Display
        cv2.putText(
            image, f"Fingers: {fingers_count}", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
        )


        y_offset = 80
        for hand_info in hand_data:
            cv2.putText(
                image, f"{hand_info['hand_type']}: {hand_info['fingers']}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
            )
            y_offset += 30

        cv2.imshow('Finger Counter', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()