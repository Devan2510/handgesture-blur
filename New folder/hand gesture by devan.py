import cv2
import mediapipe as mp
import urllib.request
import os
import math
import random

# ==========================================
# DOWNLOAD MODEL
# ==========================================

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_FILE = "hand_landmarker.task"

if not os.path.exists(MODEL_FILE):
    print("Downloading hand model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print("Download selesai!")

# ==========================================
# MEDIAPIPE
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_FILE
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# ==========================================
# GESTURE
# ==========================================

def get_gesture(hand):

    index = hand[8].y < hand[6].y
    middle = hand[12].y < hand[10].y
    ring = hand[16].y < hand[14].y
    pinky = hand[20].y < hand[18].y

    total = sum([index, middle, ring, pinky])

    # ✌️
    if index and middle and not ring and not pinky:
        return "TWO FINGERS"

    # ☝️
    if index and not middle and not ring and not pinky:
        return "POINTING"

    # ✋
    if total == 4:
        return "OPEN PALM"

    # ✊
    if total == 0:
        return "FIST"

    # 🤘
    if index and not middle and not ring and pinky:
        return "ROCK"

    return "NORMAL"


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera tidak bisa dibuka!")
    exit()

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]

        # ==================================
        # DETEKSI
        # ==================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        gesture = "NORMAL"
        hand = None

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]
            gesture = get_gesture(hand)

        # ==================================
        # ✌️ TWO FINGERS = BLUR
        # ==================================

        if gesture == "TWO FINGERS":

            frame = cv2.GaussianBlur(
                frame,
                (71, 71),
                0
            )

            cv2.putText(
                frame,
                "BLUR MODE",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                3
            )

        # ==================================
        # ✊ FIST = SHAKE + RED FLASH
        # ==================================

        elif gesture == "FIST":

            shake_x = random.randint(-12, 12)
            shake_y = random.randint(-12, 12)

            M = cv2.getRotationMatrix2D(
                (width // 2, height // 2),
                0,
                1
            )

            M[0, 2] += shake_x
            M[1, 2] += shake_y

            frame = cv2.warpAffine(
                frame,
                M,
                (width, height)
            )

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (width, height),
                (0, 0, 255),
                -1
            )

            frame = cv2.addWeighted(
                frame,
                0.7,
                overlay,
                0.3,
                0
            )

            cv2.putText(
                frame,
                "POWER",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3
            )

        # ==================================
        # ☝️ POINTING = LASER
        # ==================================

        elif gesture == "POINTING" and hand:

            x = int(hand[8].x * width)
            y = int(hand[8].y * height)

            # Laser
            cv2.line(
                frame,
                (x, y),
                (width, y),
                (0, 0, 255),
                8
            )

            # Glow
            cv2.line(
                frame,
                (x, y),
                (width, y),
                (255, 255, 255),
                2
            )

            cv2.circle(
                frame,
                (x, y),
                15,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                "LASER",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3
            )

        # ==================================
        # ✋ OPEN PALM = ENERGY CIRCLE
        # ==================================

        elif gesture == "OPEN PALM" and hand:

            # Cari posisi tengah tangan
            x = int(hand[9].x * width)
            y = int(hand[9].y * height)

            # Lingkaran energi
            for radius in range(40, 180, 35):

                cv2.circle(
                    frame,
                    (x, y),
                    radius,
                    (255, 100, 0),
                    3
                )

            # Titik tengah
            cv2.circle(
                frame,
                (x, y),
                15,
                (255, 255, 255),
                -1
            )

            cv2.putText(
                frame,
                "ENERGY",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 150, 0),
                3
            )

        # ==================================
        # 🤘 ROCK = GLITCH
        # ==================================

        elif gesture == "ROCK":

            # Geser channel warna
            b, g, r = cv2.split(frame)

            frame = cv2.merge([
                g,
                r,
                b
            ])

            # Glitch lines
            for i in range(15):

                y = random.randint(
                    0,
                    height - 1
                )

                cv2.line(
                    frame,
                    (0, y),
                    (width, y),
                    (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    ),
                    random.randint(1, 4)
                )

            cv2.putText(
                frame,
                "GLITCH",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 255),
                3
            )

        # ==================================
        # TAMPILKAN GESTURE
        # ==================================

        cv2.putText(
            frame,
            gesture,
            (30, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "HAND GESTURE EFFECTS",
            frame
        )

        # Q = keluar
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()