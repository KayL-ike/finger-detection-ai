import cv2
import mediapipe as mp
import numpy as np

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]
FINGER_NAMES = ["Jempol", "Telunjuk", "Tengah", "Manis", "Kelingking"]

LABELS = ["NOL", "SATU", "DUA", "TIGA", "EMPAT", "LIMA",
          "ENAM", "TUJUH", "DELAPAN", "SEMBILAN", "SEPULUH"]


def hitung_jari(landmarks, handedness: str) -> tuple[int, list[bool]]:
    fingers = [False] * 5

    tip  = landmarks[FINGER_TIPS[0]]
    mcp  = landmarks[FINGER_PIPS[0]]
    is_right = (handedness == "Right")
    fingers[0] = tip.x < mcp.x if is_right else tip.x > mcp.x

    for i in range(1, 5):
        fingers[i] = landmarks[FINGER_TIPS[i]].y < landmarks[FINGER_PIPS[i]].y

    return sum(fingers), fingers


def gambar_info(frame, total: int, tangan_info: list[dict]) -> None:
    """Menggambar overlay teks dan kotak info pada frame."""
    h, w = frame.shape[:2]

    label = LABELS[min(total, 10)] if tangan_info else "—"
    cv2.rectangle(frame, (w // 2 - 90, 12), (w // 2 + 90, 80), (0, 0, 0), -1)
    cv2.rectangle(frame, (w // 2 - 90, 12), (w // 2 + 90, 80), (60, 60, 60), 1)
    cv2.putText(frame, label, (w // 2 - 80, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (74, 222, 128), 3, cv2.LINE_AA)

    panel_x, panel_y = 12, h - 10 - len(tangan_info) * 30 - 40
    if tangan_info:
        cv2.rectangle(frame,
                      (panel_x - 4, panel_y - 24),
                      (panel_x + 240, panel_y + len(tangan_info) * 30 + 4),
                      (0, 0, 0), -1)
        cv2.rectangle(frame,
                      (panel_x - 4, panel_y - 24),
                      (panel_x + 240, panel_y + len(tangan_info) * 30 + 4),
                      (60, 60, 60), 1)
        cv2.putText(frame, "TANGAN TERDETEKSI", (panel_x, panel_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

        for i, info in enumerate(tangan_info):
            y = panel_y + i * 30 + 20
            nama = "Kiri " if info["label"] == "Left" else "Kanan"
            teks = f"{nama}: {info['count']} jari  |  {info['fingers_str']}"
            cv2.putText(frame, teks, (panel_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    total_teks = f"Total: {total}" if tangan_info else "Total: 0"
    cv2.rectangle(frame, (w - 130, h - 50), (w - 10, h - 10), (0, 0, 0), -1)
    cv2.rectangle(frame, (w - 130, h - 50), (w - 10, h - 10), (60, 60, 60), 1)
    cv2.putText(frame, total_teks, (w - 122, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (74, 222, 128), 2, cv2.LINE_AA)

    cv2.putText(frame, "Q / ESC : keluar", (w - 180, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)


def gambar_angka_tangan(frame, landmarks_px, count: int) -> None:
    wrist = landmarks_px[0]
    x, y = int(wrist[0]), int(wrist[1])
    label = str(count)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    pad = 8
    cv2.rectangle(frame,
                  (x - tw // 2 - pad, y + 12 - th - pad),
                  (x + tw // 2 + pad, y + 12 + pad),
                  (0, 0, 0), -1)
    cv2.rectangle(frame,
                  (x - tw // 2 - pad, y + 12 - th - pad),
                  (x + tw // 2 + pad, y + 12 + pad),
                  (74, 222, 128), 1)
    cv2.putText(frame, label,
                (x - tw // 2, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (74, 222, 128), 2, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak dapat dibuka. Coba ganti indeks: VideoCapture(1)")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 50)
    print("  Deteksi Angka Jari Tangan — MediaPipe + OpenCV")
    print("=" * 50)
    print("  Tekan Q atau ESC untuk keluar")
    print("=" * 50)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame tidak terbaca, mencoba lagi...")
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            total_jari = 0
            tangan_info = []

            if results.multi_hand_landmarks:
                for lm, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    label = handedness.classification[0].label  

                    mp_drawing.draw_landmarks(
                        frame, lm, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                    count, fingers = hitung_jari(lm.landmark, label)
                    total_jari += count

                    lm_px = [
                        (lm.landmark[i].x * w, lm.landmark[i].y * h)
                        for i in range(21)
                    ]
                    gambar_angka_tangan(frame, lm_px, count)

                    fingers_str = "".join(
                        ["▮" if f else "▯" for f in fingers]
                    )
                    tangan_info.append({
                        "label": label,
                        "count": count,
                        "fingers_str": fingers_str,
                    })

            gambar_info(frame, total_jari, tangan_info)

            cv2.imshow("Deteksi Angka Jari Tangan", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # Q atau ESC
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Program selesai.")


if __name__ == "__main__":
    main()
