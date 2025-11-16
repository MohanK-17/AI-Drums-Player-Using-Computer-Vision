import cv2

W, H = 1536, 864
SHIFT = -120   # negative = move left, positive = move right

cap = cv2.VideoCapture(0)
cap.set(3, W)
cap.set(4, H)

def shift(x):
    return x + SHIFT

zones = [
    # Top Cymbals
    {"name": "Crash Cymbal", "type": "circle", "center": (shift(W//2 - 300), 150), "r": 100, "color": (0,140,255)},
    {"name": "Ride Cymbal",  "type": "circle", "center": (shift(W//2 + 300), 150), "r": 100, "color": (0,140,255)},

    # Middle Toms
    {"name": "High Tom",     "type": "circle", "center": (shift(W//2 - 120), 300), "r": 110, "color": (160,0,255)},
    {"name": "Mid Tom",      "type": "circle", "center": (shift(W//2 + 120), 300), "r": 110, "color": (160,0,255)},

    # Side Drums
    {"name": "Hi-Hat",       "type": "circle", "center": (shift(W//2 - 450), 330), "r": 100, "color": (0,200,200)},
    {"name": "Floor Tom",    "type": "circle", "center": (shift(W//2 + 450), 330), "r": 110, "color": (180,0,200)},

    # Lower Drums
    {"name": "Snare",        "type": "circle", "center": (shift(W//2 - 350), 520), "r": 100, "color": (255,0,100)},
    {"name": "Kick",         "type": "rect",
                              "x1": shift(W//2 - 200), "y1": 500,
                              "x2": shift(W//2 + 200), "y2": 700,
                              "color": (255,50,150)}
]

while True:
    ok, frame = cap.read()
    if not ok:
        break

    for z in zones:
        if z["type"] == "circle":
            cx, cy = z["center"]
            cv2.circle(frame, (cx, cy), z["r"], z["color"], 4)
            cv2.putText(frame, z["name"], (cx - 70, cy + 5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, z["color"], 2)

        elif z["type"] == "rect":
            cv2.rectangle(frame, (z["x1"], z["y1"]), (z["x2"], z["y2"]), z["color"], 5)
            cv2.putText(frame, z["name"], (z["x1"] + 70, z["y1"] + 100),
                        cv2.FONT_HERSHEY_DUPLEX, 1, z["color"], 2)

    cv2.imshow("AI Drum Zone Mapping (Shifted Left)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
