import cv2
import numpy as np
from threading import Thread
import time
from collections import deque
import queue
import os

class Config:
    def __init__(self):
        # YOLO configuration
        self.yolo_config = "yolov3.cfg"
        self.yolo_weights = "yolov3.weights"
        self.yolo_names = "coco.names"
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.input_size = (416, 416)
        
        # Camera parameters
        self.camera_width = 426
        self.camera_height = 240
        self.max_cameras = 10
        
        # Buffer and threading parameters
        self.frame_buffer_size = 5
        self.detection_interval = 0.1

class FrameBuffer:
    def __init__(self, maxsize=5):
        self.buffer = queue.Queue(maxsize=maxsize)
    
    def put(self, frame):
        if self.buffer.full():
            try:
                self.buffer.get_nowait()
            except queue.Empty:
                pass
        self.buffer.put(frame)
    
    def get(self):
        try:
            return self.buffer.get_nowait()
        except queue.Empty:
            return None

class DetectionThread(Thread):
    def __init__(self, frame_buffer, net, output_layers, config):
        super().__init__()
        self.frame_buffer = frame_buffer
        self.net = net
        self.output_layers = output_layers
        self.config = config
        self.running = True
        self.daemon = True
        self.results = {}
        self.detection_active = False
        
    def run(self):
        while self.running:
            if self.detection_active:
                frame = self.frame_buffer.get()
                if frame is not None:
                    small_frame = cv2.resize(frame, (320, 240))
                    detected_objects = self.detect_objects(small_frame)
                    self.results[id(frame)] = detected_objects
            time.sleep(self.config.detection_interval)

    def detect_objects(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 
            1/255.0, 
            self.config.input_size, 
            swapRB=True, 
            crop=False
        )
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.config.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indices = cv2.dnn.NMSBoxes(
            boxes, 
            confidences, 
            self.config.confidence_threshold, 
            self.config.nms_threshold
        )
        
        return [(boxes[i], confidences[i], class_ids[i]) for i in indices.flatten()] if len(indices) > 0 else []

class CameraThread(Thread):
    def __init__(self, camera_id, frame_buffer, config):
        super().__init__()
        self.camera_id = camera_id
        self.frame_buffer = frame_buffer
        self.config = config
        self.capture = None
        self.running = True
        self.daemon = True
        self.initialize_camera()
        
    def initialize_camera(self):
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_ANY, "Default"),
            (0, "Legacy")
        ]
        
        for backend, name in backends:
            try:
                if backend == 0:
                    self.capture = cv2.VideoCapture(self.camera_id)
                else:
                    self.capture = cv2.VideoCapture(self.camera_id, backend)
                
                if self.capture.isOpened():
                    print(f"Successfully opened camera {self.camera_id} using {name} backend")
                    break
            except Exception as e:
                print(f"Failed to open camera {self.camera_id} with {name} backend: {str(e)}")
                continue
        
        if self.capture is None or not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
        
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
    def run(self):
        while self.running:
            if self.capture.isOpened():
                ret, frame = self.capture.read()
                if ret:
                    frame = cv2.resize(frame, (self.config.camera_width, self.config.camera_height))
                    self.frame_buffer.put(frame)
            time.sleep(0.01)
    
    def release(self):
        self.running = False
        if self.capture is not None:
            self.capture.release()

def draw_detections(frame, detections, classes):
    for box, confidence, class_id in detections:
        x, y, w, h = box
        label = f"{classes[class_id]}: {confidence:.2f}"
        color = (0, 255, 0)  # Green
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

def scan_cameras(config):
    available_cameras = {}
    print("Scanning for available cameras...")
    
    for i in range(config.max_cameras):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    camera_name = f"Camera {i}"
                    available_cameras[i] = camera_name
                    print(f"Found: {camera_name}")
                cap.release()
        except Exception as e:
            continue
    
    return available_cameras

def load_yolo(config):
    # Check if YOLO files exist
    if not all(os.path.exists(f) for f in [config.yolo_config, config.yolo_weights, config.yolo_names]):
        raise FileNotFoundError("YOLO files not found. Please ensure yolov3.cfg, yolov3.weights, and coco.names are in the current directory.")
    
    # Load class names
    with open(config.yolo_names, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Load network
    net = cv2.dnn.readNet(config.yolo_weights, config.yolo_config)
    layer_names = net.getLayerNames()
    output_layers = net.getUnconnectedOutLayersNames()
    
    return net, output_layers, classes

def main():
    # Initialize configuration
    config = Config()
    
    try:
        # Load YOLO
        net, output_layers, classes = load_yolo(config)
        
        # Scan for available cameras
        available_cameras = scan_cameras(config)
        if not available_cameras:
            print("No cameras found!")
            return
        
        # Initialize frame buffers and threads
        frame_buffers = []
        camera_threads = []
        detection_threads = []
        
        # Create threads for each camera
        for camera_id in available_cameras:
            frame_buffer = FrameBuffer(maxsize=config.frame_buffer_size)
            frame_buffers.append(frame_buffer)
            
            camera_thread = CameraThread(camera_id, frame_buffer, config)
            camera_threads.append(camera_thread)
            
            detection_thread = DetectionThread(frame_buffer, net, output_layers, config)
            detection_threads.append(detection_thread)
        
        # Start all threads
        for thread in camera_threads + detection_threads:
            thread.start()
        
        # Create windows for each camera
        for i in range(len(available_cameras)):
            cv2.namedWindow(f"Camera {i}")
        
        # Main loop
        while True:
            for i, frame_buffer in enumerate(frame_buffers):
                frame = frame_buffer.get()
                if frame is not None:
                    # Get and draw detections
                    detections = detection_threads[i].results.get(id(frame), [])
                    draw_detections(frame, detections, classes)
                    
                    # Show camera status
                    status = "Detection: ON" if detection_threads[i].detection_active else "Detection: OFF"
                    cv2.putText(
                        frame,
                        status,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )
                    
                    cv2.imshow(f"Camera {i}", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                # Toggle detection for all cameras
                for thread in detection_threads:
                    thread.detection_active = not thread.detection_active
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        # Cleanup
        for thread in camera_threads:
            thread.release()
        for thread in detection_threads:
            thread.running = False
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()