import cv2
import mediapipe as mp
import numpy as np


class FingerCounter:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        self.fingers = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }

    def get_vector(self, p1, p2):
        return [p2.x - p1.x, p2.y - p1.y, p2.z - p1.z]

    def get_vector_angle(self, v1, v2):
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)

        # Handle potential numerical errors
        cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        return np.degrees(angle_rad)

    def is_finger_raised(self, landmarks, finger_indices, is_right_hand):
        # Get key points
        mcp = landmarks[finger_indices[0]]  # Base of finger
        pip = landmarks[finger_indices[1]]  # First joint
        dip = landmarks[finger_indices[2]]  # Second joint
        tip = landmarks[finger_indices[3]]  # Fingertip

        # Special handling for thumb
        if finger_indices[0] == 1:  # Thumb
            return self.is_thumb_raised(landmarks, is_right_hand)

        # For other fingers
        # Vector from base to tip
        base_to_tip = self.get_vector(mcp, tip)

        # Vector from base to middle joint (PIP)
        base_to_pip = self.get_vector(mcp, pip)

        # Check straightness
        bend_angle = self.get_vector_angle(
            self.get_vector(pip, mcp),
            self.get_vector(pip, dip)
        )

        is_straight = bend_angle > 120

        # Check if finger is extended
        palm_center = landmarks[0]  # Using wrist as palm reference
        tip_distance = np.linalg.norm(self.get_vector(palm_center, tip))
        pip_distance = np.linalg.norm(self.get_vector(palm_center, pip))
        is_extended = tip_distance > pip_distance

        return is_straight and is_extended

    def is_thumb_raised(self, landmarks, is_right_hand):
        wrist = landmarks[0]
        thumb_cmc = landmarks[1]  # Thumb base
        thumb_mcp = landmarks[2]  # Thumb first joint
        thumb_ip = landmarks[3]  # Thumb second joint
        thumb_tip = landmarks[4]  # Thumb tip
        index_mcp = landmarks[5]  # Index finger base

        # 1. Check the angle between palm and thumb
        palm_vector = self.get_vector(wrist, index_mcp)
        thumb_vector = self.get_vector(thumb_cmc, thumb_tip)
        thumb_angle = self.get_vector_angle(palm_vector, thumb_vector)

        # 2. Check if thumb tip is away from palm
        thumb_palm_distance = np.linalg.norm(self.get_vector(wrist, thumb_tip))
        thumb_base_distance = np.linalg.norm(self.get_vector(wrist, thumb_mcp))

        angle_threshold = 20
        distance_ratio = thumb_palm_distance / thumb_base_distance

        return (thumb_angle > angle_threshold) and (distance_ratio > 1.2)

    def count_fingers(self, image):
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        image.flags.writeable = True

        fingers_count = 0
        hand_data = []

        if results.multi_hand_landmarks and results.multi_handedness:
            # Zip together the landmarks and handedness information
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                landmarks = hand_landmarks.landmark

                # Get hand type from MediaPipe
                hand_type = handedness.classification[0].label
                is_right_hand = (hand_type == "Right")

                hand_fingers = []

                # Check each finger
                for finger_name, finger_indices in self.fingers.items():
                    if self.is_finger_raised(landmarks, finger_indices, is_right_hand):
                        fingers_count += 1
                        hand_fingers.append(1)
                    else:
                        hand_fingers.append(0)

                # Calculate and store angles for visualization
                finger_info = []
                for finger_name, finger_indices in self.fingers.items():
                    if finger_name == 'thumb':
                        # Special angle calculation for thumb
                        wrist = landmarks[0]
                        thumb_mcp = landmarks[2]
                        thumb_tip = landmarks[4]
                        index_mcp = landmarks[5]

                        palm_vector = self.get_vector(wrist, index_mcp)
                        thumb_vector = self.get_vector(thumb_mcp, thumb_tip)
                        angle = self.get_vector_angle(palm_vector, thumb_vector)
                    else:
                        mcp = landmarks[finger_indices[0]]
                        pip = landmarks[finger_indices[1]]
                        dip = landmarks[finger_indices[2]]

                        angle = self.get_vector_angle(
                            self.get_vector(pip, mcp),
                            self.get_vector(pip, dip)
                        )
                    finger_info.append(f"{finger_name}: {angle:.1f}°")

                hand_data.append({
                    "hand_type": hand_type,
                    "fingers": hand_fingers,
                    "finger_angles": finger_info
                })

        return fingers_count, hand_data


def main():
    cap = cv2.VideoCapture(0)
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
                image, f"{hand_info['hand_type']}: {hand_info['fingers']}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
            )
            y_offset += 30

            # Display finger angles
            for i, angle_info in enumerate(hand_info['finger_angles']):
                cv2.putText(
                    image, angle_info,
                    (10, y_offset + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
                )
            y_offset += len(hand_info['finger_angles']) * 20 + 10

        cv2.imshow('Finger Counter', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()