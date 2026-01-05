
import cv2
import socket
import json
import base64
import time
import threading
import queue
import argparse
from typing import Optional
class CameraServer:
    def __init__(
        self,
        server_host: str = "localhost",
        server_port: int = 6100,
        video_source: Optional[str] = None,
        fps_limit: int = 3
    ):
        self.server_host = server_host
        self.server_port = server_port
        self.video_source = video_source
        self.fps_limit = fps_limit
        self.frame_delay = 1.0 / fps_limit

        self.frame_queue = queue.Queue(maxsize=20)
        self.running = False

        self.sock = None
        self.frame_id = 0

    def connect_to_server(self) -> bool:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.server_host, self.server_port))
            print(f"[CAMERA] Connected to {self.server_host}:{self.server_port}")
            return True
        except Exception as e:
            print(f"[CAMERA] Connection error: {e}")
            return False

    # ================= CAPTURE THREAD =================
    def capture_frames(self):
        cap = cv2.VideoCapture(0 if self.video_source is None else self.video_source)

        if not cap.isOpened():
            print("[CAMERA] Cannot open video source")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                if self.video_source:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame = cv2.resize(frame, (320, 240))

            if not self.frame_queue.full():
                self.frame_queue.put(frame)

            cv2.imshow("Camera Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False
                break

        cap.release()
        cv2.destroyAllWindows()


    def send_frames(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)

                time.sleep(self.frame_delay)

                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_base64 = base64.b64encode(buffer).decode("utf-8")

                packet = {
                    "frame_id": self.frame_id,
                    "timestamp": time.time(),
                    "frame_data": frame_base64,
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "channels": frame.shape[2]
                }

                self.sock.send((json.dumps(packet) + "\n").encode("utf-8"))
                print(f"[CAMERA] Sent frame {self.frame_id}")

                self.frame_id += 1

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[CAMERA] Send error: {e}")
                self.running = False
                break

    def start(self):
        if not self.connect_to_server():
            return

        self.running = True

        threading.Thread(target=self.capture_frames, daemon=True).start()
        threading.Thread(target=self.send_frames, daemon=True).start()

        print("[CAMERA] Camera server started")

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()
        print("[CAMERA] Camera server stopped")


# ================= MAIN =================
def main():
    parser = argparse.ArgumentParser(description="Camera Server - Streaming frames to Spark")
    parser.add_argument("--host", default="localhost", help="Spark server IP")
    parser.add_argument("--port", type=int, default=6100, help="Spark server port")
    parser.add_argument("--video", default=None, help="Video file path (optional)")
    parser.add_argument("--fps", type=int, default=3, help="FPS limit")

    args = parser.parse_args()

    camera_server = CameraServer(
        server_host=args.host,
        server_port=args.port,
        video_source=args.video,
        fps_limit=args.fps
    )

    camera_server.start()


if __name__ == "__main__":
    main()
