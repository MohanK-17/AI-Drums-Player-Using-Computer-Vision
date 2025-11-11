import cv2
from datetime import datetime

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Try changing the camera index (e.g., 1 or 2).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INFO] Webcam test started.")
    print("[INFO] Press 'S' to save a snapshot or 'ESC' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame. Exiting...")
            break

        cv2.putText(frame, "Webcam Smoke Test", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("AI Drums - Webcam Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Snapshot saved as {filename}")
        elif key == 27:
            print("[INFO] Exiting webcam test...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
