import cv2
import mediapipe as mp
import time
import math

# Screen setup
W, H = 1536, 864
SHIFT = -120

cap = cv2.VideoCapture(0)
cap.set(3, W)
cap.set(4, H)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def shift(x): return x + SHIFT

# Drum zones definition
zones = [
    {"name": "Crash Cymbal", "type": "circle", "center": (shift(W//2 - 300), 150), "r": 100, "color": (0,140,255)},
    {"name": "Ride Cymbal",  "type": "circle", "center": (shift(W//2 + 300), 150), "r": 100, "color": (0,140,255)},
    {"name": "High Tom",     "type": "circle", "center": (shift(W//2 - 120), 300), "r": 110, "color": (160,0,255)},
    {"name": "Mid Tom",      "type": "circle", "center": (shift(W//2 + 120), 300), "r": 110, "color": (160,0,255)},
    {"name": "Hi-Hat",       "type": "circle", "center": (shift(W//2 - 450), 330), "r": 100, "color": (0,200,200)},
    {"name": "Floor Tom",    "type": "circle", "center": (shift(W//2 + 450), 330), "r": 110, "color": (180,0,200)},
    {"name": "Snare",        "type": "circle", "center": (shift(W//2 - 350), 520), "r": 100, "color": (255,0,100)},
    {"name": "Kick",         "type": "rect", "x1": shift(W//2 - 200), "y1": 530, "x2": shift(W//2 + 200), "y2": 720, "color": (255,50,150)}
]

highlight = {z["name"]: 0 for z in zones}
last_hit = {z["name"]: 0 for z in zones}
cooldown = 0.25
p_time = 0
hit_display = ""

def detect_zone(x, y):
    for z in zones:
        if z["type"] == "circle":
            cx, cy = z["center"]
            if math.dist((x, y), (cx, cy)) <= z["r"]:
                return z["name"]
        else:
            if z["x1"] <= x <= z["x2"] and z["y1"] <= y <= z["y2"]:
                return z["name"]
    return None

def draw_zones(frame):
    for z in zones:
        if z["type"] == "circle":
            cx, cy = z["center"]
            cv2.circle(frame, (cx, cy), z["r"], z["color"], 4)
            cv2.putText(frame, z["name"], (cx - len(z["name"])*7, cy + 5), cv2.FONT_HERSHEY_DUPLEX, 0.7, z["color"], 2)
        else:
            cv2.rectangle(frame, (z["x1"], z["y1"]), (z["x2"], z["y2"]), z["color"], 4)
            tx = z["x1"] + (z["x2"] - z["x1"])//2 - len(z["name"])*7
            ty = z["y1"] + (z["y2"] - z["y1"])//2
            cv2.putText(frame, z["name"], (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.8, z["color"], 2)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    now = time.time()

    draw_zones(frame)
    hit_display = ""

    if results.multi_hand_landmarks:
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 8:  # Index finger tip
                    cv2.circle(frame, (cx, cy), 12, (0, 255, 255), cv2.FILLED)
                    cv2.putText(frame, "Index Tip", (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Check for zone hit
                    zone = detect_zone(cx, cy)
                    if zone and (now - last_hit[zone] > cooldown):
                        last_hit[zone] = now
                        highlight[zone] = now + 0.25
                        hit_display = f"Hit: {zone} ({hand_label})"
                        print(hit_display)

    # Highlight hit zones
    for z in zones:
        if highlight[z["name"]] > now:
            overlay = frame.copy()
            if z["type"] == "circle":
                cx, cy = z["center"]
                cv2.circle(overlay, (cx, cy), z["r"], z["color"], -1)
            else:
                cv2.rectangle(overlay, (z["x1"], z["y1"]), (z["x2"], z["y2"]), z["color"], -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # Display hit message
    if hit_display:
        cv2.putText(frame, hit_display, (W//2 - 240, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), 3)

    # FPS counter
    c_time = time.time()
    fps = 1 / (c_time - p_time) if c_time != p_time else 0
    p_time = c_time
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    cv2.imshow("AI Drums - Index Fingertip Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
