#!/usr/bin/env python3

import socket
import json
import os
import time
import threading
import queue
from pyspark.sql import SparkSession

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "selfie_segmenter.tflite")
OUTPUT_DIR = os.path.join(BASE_DIR, "image_processed")


def process_frame_static(data):
    try:
        packet, model_path, output_dir = data

        import cv2
        import base64
        import numpy as np
        from datetime import datetime
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import os

        # Decode frame
        frame_bytes = base64.b64decode(packet["frame_data"])
        frame = cv2.imdecode(
            np.frombuffer(frame_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )
        if frame is None:
            return "ERROR: frame decode failed"

        # Load model
        options = vision.ImageSegmenterOptions(
            base_options=python.BaseOptions(
                model_asset_path=model_path
            ),
            output_category_mask=True
        )
        segmenter = vision.ImageSegmenter.create_from_options(options)

        # Segmentation
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )
        mask = segmenter.segment(mp_image).category_mask.numpy_view()

        if mask.ndim == 2:
            mask = np.stack([mask] * 3, axis=-1)

        bg = np.full(frame.shape, (192, 192, 192), dtype=np.uint8)
        result = np.where(mask > 0.2, bg, frame)

        # Ensure output dir exists (TRONG WORKER)
        os.makedirs(output_dir, exist_ok=True)

        ts = datetime.fromtimestamp(packet["timestamp"]).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            output_dir,
            f"frame_{packet['frame_id']}_{ts}.jpg"
        )

        cv2.imwrite(path, result)
        return path

    except Exception as e:
        return f"ERROR: {str(e)}"

class SparkProcessingServer:
    def __init__(
        self,
        host="localhost",
        port=6100,
        batch_size=5
    ):
        self.host = host
        self.port = port
        self.batch_size = batch_size

        self.queue = queue.Queue()
        self.running = True

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        self._init_spark()

    def _init_spark(self):
        import sys
        python_path = sys.executable

        self.spark = (
            SparkSession.builder
            .appName("BigDataBackgroundRemoval")
            .master("local[*]")
            .config("spark.pyspark.python", python_path)
            .config("spark.pyspark.driver.python", python_path)
            .getOrCreate()
        )

        self.sc = self.spark.sparkContext
        self.sc.setLogLevel("WARN")
        print("[SPARK] Ready")

    def _setup_server(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        self.conn, addr = self.sock.accept()
        print(f"[SERVER] Connected from {addr}")

    def _receive_frames(self):
        buffer = ""
        while self.running:
            data = self.conn.recv(4096).decode()
            if not data:
                break

            buffer += data
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line.strip():
                    self.queue.put(json.loads(line))

    def _process_batch(self, batch):
        rdd = self.sc.parallelize(
            [(pkt, MODEL_PATH, OUTPUT_DIR) for pkt in batch],
            numSlices=min(len(batch), self.sc.defaultParallelism)
        )

        results = rdd.map(process_frame_static).collect()

        errors = [
            r for r in results
            if isinstance(r, str) and r.startswith("ERROR")
        ]
        if errors:
            print("[SPARK] Errors:", errors[:2])
        else:
            print(f"[SPARK] Saved {len(results)} frames")

    def _worker(self):
        batch = []
        last = time.time()

        while self.running:
            try:
                batch.append(self.queue.get(timeout=1))
            except queue.Empty:
                pass

            if (
                len(batch) >= self.batch_size
                or (batch and time.time() - last > 3)
            ):
                print(f"[SPARK] Processing {len(batch)} frames")
                self._process_batch(batch)
                batch.clear()
                last = time.time()

    def start(self):
        self._setup_server()

        threading.Thread(
            target=self._receive_frames,
            daemon=True
        ).start()

        threading.Thread(
            target=self._worker,
            daemon=True
        ).start()

        try:
            while True:
                time.sleep(1)
                if self.queue.qsize():
                    print(f"[QUEUE] {self.queue.qsize()} frames waiting")
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.running = False
        self.conn.close()
        self.sock.close()
        self.spark.stop()
        print("[SERVER] Stopped")


if __name__ == "__main__":
    SparkProcessingServer().start()
