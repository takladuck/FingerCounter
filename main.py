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
        # Landmark indices for each finger
        self.fingers = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }

    def get_vector(self, p1, p2):
        """Calculate vector between two points."""
        return [p2.x - p1.x, p2.y - p1.y, p2.z - p1.z]

    def get_vector_angle(self, v1, v2):
        """Calculate angle between two vectors."""
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)

        # Handle potential numerical errors
        cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        return np.degrees(angle_rad)

    def is_finger_raised(self, landmarks, finger_indices):
        """Check if a finger is raised using both straightness and extension."""
        # Get key points
        mcp = landmarks[finger_indices[0]]  # Base of finger
        pip = landmarks[finger_indices[1]]  # First joint
        dip = landmarks[finger_indices[2]]  # Second joint
        tip = landmarks[finger_indices[3]]  # Fingertip

        # Vector from base to tip
        base_to_tip = self.get_vector(mcp, tip)

        # Vector from base to middle joint (PIP)
        base_to_pip = self.get_vector(mcp, pip)

        # Check straightness
        bend_angle = self.get_vector_angle(
            self.get_vector(pip, mcp),
            self.get_vector(pip, dip)
        )

        # More lenient straightness threshold
        is_straight = bend_angle > 120  # Reduced from 165

        # Check if finger is extended (tip is further from palm than PIP joint)
        # We'll use the y-coordinate primarily but consider z-coordinate too
        is_extended = False

        # Different logic for thumb vs other fingers
        if finger_indices[0] == 1:  # Thumb
            # For thumb, check if it's extended sideways
            wrist = landmarks[0]
            palm_vector = self.get_vector(wrist, landmarks[5])  # Vector to index MCP
            thumb_vector = self.get_vector(mcp, tip)
            thumb_angle = self.get_vector_angle(palm_vector, thumb_vector)
            is_extended = thumb_angle > 35  # Reduced from 45
        else:
            # For other fingers, check if tip is higher than PIP joint
            palm_center = landmarks[0]  # Using wrist as palm reference
            tip_distance = np.linalg.norm(self.get_vector(palm_center, tip))
            pip_distance = np.linalg.norm(self.get_vector(palm_center, pip))
            is_extended = tip_distance > pip_distance

        return is_straight and is_extended

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

                # Determine hand type
                hand_type = "Right" if landmarks[17].x > landmarks[5].x else "Left"

                # Check each finger
                for finger_name, finger_indices in self.fingers.items():
                    if self.is_finger_raised(landmarks, finger_indices):
                        fingers_count += 1
                        hand_fingers.append(1)
                    else:
                        hand_fingers.append(0)

                # Calculate and store angles for visualization
                finger_info = []
                for finger_name, finger_indices in self.fingers.items():
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