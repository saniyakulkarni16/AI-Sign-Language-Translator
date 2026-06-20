"""
detect_realtime.py
==================
Real-time ISL detection - works with A-Z image folder trained model.

Keyboard shortcuts:
    SPACE     - add current letter to sentence
    BACKSPACE - remove last letter
    C         - clear sentence
    Q         - quit
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
import os
import time
import collections

# -- Load model ---------------------------------------------------------------
MODELS_DIR = "models"
print("Loading model...")
rf     = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
le     = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
with open(os.path.join(MODELS_DIR, "feature_cols.json")) as f:
    feature_cols = json.load(f)

classes = list(le.classes_)   # ['A','B','C',...,'Z']
print(f"Model loaded OK  ({len(classes)} classes: {classes})\n")

# -- MediaPipe ----------------------------------------------------------------
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# -- Feature extractor --------------------------------------------------------
def build_feature_vector(results):
    left_lm  = np.zeros(63, dtype=np.float32)
    right_lm = np.zeros(63, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lm, handedness in zip(results.multi_hand_landmarks,
                                       results.multi_handedness):
            side  = handedness.classification[0].label
            wrist = hand_lm.landmark[0]
            coords = []
            for lm in hand_lm.landmark:
                coords.extend([lm.x - wrist.x,
                                lm.y - wrist.y,
                                lm.z - wrist.z])
            arr = np.array(coords, dtype=np.float32)
            if side == "Left":
                left_lm = arr
            else:
                right_lm = arr

    return np.concatenate([left_lm, right_lm])


def predict(fv):
    X      = scaler.transform(fv.reshape(1, -1))
    probs  = rf.predict_proba(X)[0]
    cls_id = int(np.argmax(probs))
    conf   = float(probs[cls_id])
    label  = str(le.inverse_transform([cls_id])[0])
    return label, conf

# -- Top-3 predictions --------------------------------------------------------
def top3(fv):
    X     = scaler.transform(fv.reshape(1, -1))
    probs = rf.predict_proba(X)[0]
    top   = np.argsort(probs)[::-1][:3]
    return [(str(le.inverse_transform([i])[0]), float(probs[i])) for i in top]

# -- Drawing ------------------------------------------------------------------
def conf_color(conf):
    if conf >= 0.75: return (50, 220, 80)
    if conf >= 0.45: return (50, 180, 220)
    return (80, 80, 220)


def draw_ui(frame, label, conf, top3_preds, sentence, fps, hand_detected):
    h, w  = frame.shape[:2]
    dark  = (20, 20, 20)
    white = (255, 255, 255)
    gray  = (130, 130, 130)

    # Top bar background
    cv2.rectangle(frame, (0, 0), (w, 80), dark, -1)

    if hand_detected:
        color = conf_color(conf)

        # Big letter
        cv2.putText(frame, label, (15, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 5)

        # Confidence bar
        bx, by, bw, bh = 110, 15, 380, 22
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
        cv2.rectangle(frame, (bx, by), (bx + int(bw * conf), by + bh), color, -1)
        cv2.putText(frame, f"{conf*100:.1f}%",
                    (bx + bw + 8, by + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, white, 1)

        # Top-3 alternatives (small, top right)
        for i, (lbl, c) in enumerate(top3_preds):
            shade = (200, 200, 200) if i == 0 else (100, 100, 100)
            cv2.putText(frame, f"{lbl}: {c*100:.0f}%",
                        (w - 130, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, shade, 1)
    else:
        cv2.putText(frame, "No hand detected - show your hand", (15, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (80, 80, 200), 2)

    # FPS
    cv2.putText(frame, f"FPS {fps:.0f}", (w - 130, h - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, gray, 1)

    # Sentence bar
    cv2.rectangle(frame, (0, h - 65), (w, h), dark, -1)
    disp = " ".join(sentence) if sentence else "[ Press SPACE to add letter ]"
    color2 = (255, 200, 50) if sentence else gray
    cv2.putText(frame, disp, (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color2, 2)

    # Help
    cv2.putText(frame,
                "SPACE: add   BACKSPACE: delete   C: clear   Q: quit",
                (10, h - 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, gray, 1)
    return frame


# -- Main loop ----------------------------------------------------------------
def run(camera=0, smooth_window=10, conf_threshold=0.35):
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    pred_buffer = collections.deque(maxlen=smooth_window)
    sentence    = []
    prev_time   = time.time()
    t3          = [("-", 0.0)] * 3

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.5,
    ) as hands:

        print("Webcam open! Hold ISL signs steady for best results.")
        print("SPACE=add  BACKSPACE=delete  C=clear  Q=quit\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            label, conf   = "-", 0.0
            hand_detected = False

            if results.multi_hand_landmarks:
                hand_detected = True
                for hl in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hl, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                fv = build_feature_vector(results)
                raw_label, raw_conf = predict(fv)
                t3 = top3(fv)

                pred_buffer.append((raw_label, raw_conf))
                votes = collections.Counter(p[0] for p in pred_buffer)
                label = votes.most_common(1)[0][0]
                confs = [p[1] for p in pred_buffer if p[0] == label]
                conf  = float(np.mean(confs))
            else:
                pred_buffer.clear()
                t3 = [("-", 0.0)] * 3

            frame = draw_ui(frame, label, conf, t3, sentence, fps, hand_detected)
            cv2.imshow("ISL Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if hand_detected and conf >= conf_threshold:
                    sentence.append(label)
                    print(f"Added: {label}  ->  {''.join(sentence)}")
            elif key == 8:
                if sentence:
                    print(f"Removed: {sentence.pop()}")
            elif key == ord('c'):
                sentence.clear()
                print("Cleared.")

    cap.release()
    cv2.destroyAllWindows()
    if sentence:
        print(f"\nFinal: {''.join(sentence)}")


if __name__ == "__main__":
    run()
