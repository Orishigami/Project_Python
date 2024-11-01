import cv2
import numpy as np
from threading import Thread
import time

# Initialize global variables for ROI drawing
roi_start = None
roi_end = None
rois = {}
selected_camera = None
car_count_per_roi = {}  # ตัวแปรเก็บจำนวนรถสำหรับแต่ละ ROI ของกล้องแต่ละตัว

# กำหนดสีสำหรับ ROI ในแต่ละกล้อง
roi_colors = [(0, 0, 255), (0, 255, 0)]  # สีแดงและสีเขียว

# โหลด YOLOv3
yolo_config = "yolov3.cfg"
yolo_weights = "yolov3.weights"
yolo_names = "coco.names"

# โหลดชื่อคลาส
with open(yolo_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# กำหนดค่าของ YOLO
net = cv2.dnn.readNet(yolo_weights, yolo_config)

# ใช้ CUDA กับ YOLOv3
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()

def scan_cameras(max_cameras=10):
    available_cameras = {}
    
    print("กำลังสแกนหากล้องที่ใช้งานได้...")
    for i in range(max_cameras):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    camera_name = f"Camera {i}"
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    camera_name += f" - {width}x{height}"
                    
                    available_cameras[i] = camera_name
                    print(f"พบกล้อง: {camera_name}")
                cap.release()
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการสแกนกล้อง {i}: {str(e)}")
            continue
    
    if not available_cameras:
        print("ไม่พบกล้องที่ใช้งานได้")
    else:
        print(f"\nพบกล้องที่ใช้งานได้ทั้งหมด {len(available_cameras)} ตัว")
    
    return available_cameras

def select_cameras(available_cameras):
    if not available_cameras:
        return []
    
    selected_cameras = []
    print("\nเลือกกล้องที่ต้องการใช้งาน (สูงสุด 4 ตัว):")
    print("กรุณาป้อนหมายเลขกล้องที่ต้องการ (คั่นด้วยช่องว่าง) หรือกด Enter เพื่อใช้ทุกกล้อง")
    print("ตัวอย่าง: 0 1 2")
    
    while True:
        user_input = input("\nเลือกกล้อง: ").strip()
        
        if not user_input:
            selected_cameras = list(available_cameras.keys())[:4]
            break
            
        try:
            selected = [int(x) for x in user_input.split()]
            if not all(x in available_cameras for x in selected):
                print("กรุณาเลือกเฉพาะกล้องที่มีอยู่")
                continue
                
            if len(selected) > 4:
                print("สามารถเลือกกล้องได้สูงสุด 4 ตัว")
                continue
                
            if len(selected) < 1:
                print("กรุณาเลือกอย่างน้อย 1 กล้อง")
                continue
                
            selected_cameras = selected
            break
            
        except ValueError:
            print("กรุณาป้อนเฉพาะตัวเลขคั่นด้วยช่องว่าง")
    
    print("\nกล้องที่ถูกเลือก:")
    for cam_id in selected_cameras:
        print(f"- {available_cameras[cam_id]}")
    
    return selected_cameras

class CameraThread:
    def __init__(self, camera_id, width=426, height=240):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.capture = None
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_ANY, "Default"),
            (0, "Legacy")
        ]
        
        for backend, name in backends:
            try:
                if backend == 0:
                    self.capture = cv2.VideoCapture(camera_id)
                else:
                    self.capture = cv2.VideoCapture(camera_id, backend)
                
                if self.capture.isOpened():
                    print(f"Successfully opened camera {camera_id} using {name} backend")
                    break
            except Exception as e:
                print(f"Failed to open camera {camera_id} with {name} backend: {str(e)}")
                continue
        
        if self.capture is None or not self.capture.isOpened():
            raise RuntimeError(f"ไม่สามารถเปิดกล้อง {camera_id} ได้ด้วย backend ใดๆ")
        
        try:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        except Exception as e:
            print(f"Warning: Could not set camera properties: {str(e)}")
        
        self.frame = None
        self.ret = False
        self.running = True
        self.thread = Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.detection_active = False  # เพิ่มตัวแปรควบคุมการตรวจจับ

    def _update(self):
        while self.running:
            if self.capture.isOpened():
                self.ret, frame = self.capture.read()
                if self.ret:
                    self.frame = cv2.resize(frame, (self.width, self.height))
            time.sleep(0.01)

    def get_frame(self):
        if self.frame is None:
            return False, np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.capture is not None:
            self.capture.release()

def draw_roi(event, x, y, flags, param):
    global roi_start, roi_end, rois, selected_camera

    cam_id = None
    for i, pos in enumerate(param):
        cam_x, cam_y = pos
        if cam_x <= x < cam_x + 426 and cam_y <= y < cam_y + 240:
            cam_id = i
            break

    if cam_id is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x - pos[0], y - pos[1])
        selected_camera = cam_id
    elif event == cv2.EVENT_MOUSEMOVE and roi_start:
        roi_end = (x - pos[0], y - pos[1])
    elif event == cv2.EVENT_LBUTTONUP and roi_start:
        roi_end = (x - pos[0], y - pos[1])

        if cam_id not in rois:
            rois[cam_id] = []
        
        if len(rois[cam_id]) < 2:
            rois[cam_id].append((roi_start, roi_end))
            print(f"ROI added for Camera {cam_id}: {rois[cam_id][-1]}")
            cameras[cam_id].detection_active = True  # เปิดใช้งานการตรวจจับเมื่อวาด ROI เสร็จ
        else:
            print(f"Camera {cam_id} reached the maximum ROI limit of 2")

        roi_start, roi_end = None, None

def is_point_in_roi(x, y, roi_start, roi_end):
    x1, y1 = min(roi_start[0], roi_end[0]), min(roi_start[1], roi_end[1])
    x2, y2 = max(roi_start[0], roi_end[0]), max(roi_start[1], roi_end[1])
    return x1 <= x <= x2 and y1 <= y <= y2

def is_box_in_roi(box, roi_start, roi_end):
    center_x = box[0] + box[2] // 2
    center_y = box[1] + box[3] // 2
    return is_point_in_roi(center_x, center_y, roi_start, roi_end)

def detect_objects(frame, camera_rois=None, detection_active=False, cam_id=None):

    if not detection_active or not camera_rois:
        return []

    # เลือก GPU สำหรับการประมวลผล
    gpu_id = gpu_manager.get_next_gpu()
    if gpu_id is not None:
        cv2.cuda.setDevice(gpu_id)
        net = gpu_manager.models[gpu_id]
    else:
        net = gpu_manager.models['cpu']

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []
    
    # รีเซ็ตตัวนับจำนวนรถของกล้องแต่ละตัวสำหรับแต่ละ ROI ก่อนการตรวจจับ
    car_count_per_roi[cam_id] = [0] * len(camera_rois)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "car":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                box = [x, y, w, h]
                
                for idx, (roi_start, roi_end) in enumerate(camera_rois):
                    if is_box_in_roi(box, roi_start, roi_end):
                        boxes.append(box)
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
                        car_count_per_roi[cam_id][idx] += 1  # เพิ่มจำนวนรถใน ROI นี้
                        break

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    result_boxes = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            result_boxes.append((x, y, w, h, label, confidence))
    
    return result_boxes

def print_gpu_stats():
    """
    แสดงสถิติการใช้งาน GPU
    """
    for gpu in gpu_manager.gpus:
        cv2.cuda.setDevice(gpu['device_id'])
        free_mem, total_mem = cv2.cuda.getMemInfo()
        used_mem = total_mem - free_mem
        usage_percent = (used_mem / total_mem) * 100
        
        print(f"\nGPU {gpu['device_id']}: {gpu['name']}")
        print(f"Memory Usage: {used_mem/1024**3:.2f}GB / {total_mem/1024**3:.2f}GB ({usage_percent:.1f}%)")

def draw_detected_objects(frame, boxes, cam_id):
    for (x, y, w, h, label, confidence) in boxes:
        color = (255, 0, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # แสดงจำนวนรถในแต่ละ ROI บนหน้าจอ
    if cam_id in car_count_per_roi:
        for idx, count in enumerate(car_count_per_roi[cam_id]):
            start, end = rois[cam_id][idx]
            color = roi_colors[idx % len(roi_colors)]
            cv2.putText(
                frame,
                f"Cars: {count}",
                (start[0], start[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

def check_cuda_availability():
    """
    ตรวจสอบว่าระบบสามารถใช้งาน CUDA ได้หรือไม่
    returns: (bool, str) - (CUDA available?, message)
    """
    try:
        # ตรวจสอบว่า OpenCV build มี CUDA support หรือไม่
        cv2_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        if not cv2_cuda:
            return False, "OpenCV was not built with CUDA support"
        
        # ตรวจสอบว่ามี CUDA device พร้อมใช้งานหรือไม่
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        if device_count == 0:
            return False, "No CUDA devices available"
            
        # ตรวจสอบข้อมูล CUDA device
        device = cv2.cuda.getDevice()
        device_name = cv2.cuda.getDeviceName(device)
        compute_capability = cv2.cuda.computeCapability(device)
        
        return True, f"CUDA available on {device_name} (Compute {compute_capability[0]}.{compute_capability[1]})"
        
    except Exception as e:
        return False, f"Error checking CUDA: {str(e)}"

# แก้ไขส่วนการโหลด YOLO model
def initialize_yolo(use_cuda=True):
    """
    โหลดและกำหนดค่า YOLO model
    params:
        use_cuda: bool - เปิดใช้งาน CUDA หรือไม่
    returns: (net, message)
    """
    try:
        net = cv2.dnn.readNet(yolo_weights, yolo_config)
        
        if use_cuda:
            cuda_available, cuda_message = check_cuda_availability()
            if cuda_available:
                print(f"Enabling CUDA: {cuda_message}")
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                print(f"CUDA not available: {cuda_message}")
                print("Falling back to CPU")
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            print("Using CPU (CUDA disabled)")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
        return net, "Model loaded successfully"
        
    except Exception as e:
        return None, f"Error initializing YOLO: {str(e)}"    

def get_available_gpus():
    """
    ตรวจสอบและคืนค่า GPU ที่สามารถใช้งานได้ทั้งหมด
    returns: list of (device_id, name, memory)
    """
    available_gpus = []
    try:
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
        for device_id in range(device_count):
            cv2.cuda.setDevice(device_id)
            device_name = cv2.cuda.getDeviceName(device_id)
            free_mem, total_mem = cv2.cuda.getMemInfo()
            compute_capability = cv2.cuda.computeCapability(device_id)
            
            gpu_info = {
                'device_id': device_id,
                'name': device_name,
                'free_memory': free_mem,
                'total_memory': total_mem,
                'compute_capability': compute_capability
            }
            available_gpus.append(gpu_info)
            
    except Exception as e:
        print(f"Error getting GPU info: {str(e)}")
    
    return available_gpus

class MultiGPUManager:
    def __init__(self):
        self.gpus = get_available_gpus()
        self.models = {}
        self.current_gpu_index = 0
    
    def initialize_models(self):
        """
        โหลด YOLO model สำหรับแต่ละ GPU ที่ใช้งานได้
        """
        if not self.gpus:
            print("No GPUs available, falling back to CPU")
            net = cv2.dnn.readNet(yolo_weights, yolo_config)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.models['cpu'] = net
            return
        
        for gpu in self.gpus:
            try:
                cv2.cuda.setDevice(gpu['device_id'])
                net = cv2.dnn.readNet(yolo_weights, yolo_config)
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.models[gpu['device_id']] = net
                print(f"Initialized model on GPU {gpu['device_id']}: {gpu['name']}")
            except Exception as e:
                print(f"Failed to initialize model on GPU {gpu['device_id']}: {str(e)}")
    
    def get_next_gpu(self):
        """
        เลือก GPU ถัดไปแบบ round-robin
        """
        if not self.gpus:
            return None
        
        gpu = self.gpus[self.current_gpu_index]
        self.current_gpu_index = (self.current_gpu_index + 1) % len(self.gpus)
        return gpu['device_id']

def main():

    global gpu_manager
    gpu_manager = MultiGPUManager()
    
    # แสดงข้อมูล GPU ที่พบ
    print("\nDetected GPUs:")
    for gpu in gpu_manager.gpus:
        print(f"GPU {gpu['device_id']}: {gpu['name']}")
        print(f"Memory: {gpu['free_memory']/1024**3:.2f}GB free / {gpu['total_memory']/1024**3:.2f}GB total")
        print(f"Compute Capability: {gpu['compute_capability'][0]}.{gpu['compute_capability'][1]}")
        print()
    
    # โหลด model สำหรับแต่ละ GPU
    gpu_manager.initialize_models()
    # ตรวจสอบ CUDA ก่อนเริ่มโปรแกรม
    cuda_available, cuda_message = check_cuda_availability()
    print(f"\nCUDA Status: {cuda_message}")

    # โหลด YOLO model
    global net
    net, model_message = initialize_yolo(use_cuda=cuda_available)
    if net is None:
        print(f"Failed to initialize YOLO: {model_message}")
        return
    
    print(f"YOLO Status: {model_message}")
    
    # แสดงข้อมูลการใช้งาน GPU/CPU
    if cuda_available:
        device = cv2.cuda.getDevice()
        print(f"Using GPU: {cv2.cuda.getDeviceName(device)}")
        free_mem, total_mem = cv2.cuda.getMemInfo()
        print(f"GPU Memory: {free_mem/1024**3:.2f}GB free / {total_mem/1024**3:.2f}GB total")
    else:
        print("Using CPU for inference")

    available_cameras = scan_cameras()
    if not available_cameras:
        print("ไม่สามารถเริ่มโปรแกรมได้เนื่องจากไม่พบกล้อง")
        return

    selected_cameras = select_cameras(available_cameras)
    if not selected_cameras:
        print("ไม่ได้เลือกกล้อง จบการทำงาน")
        return

    global cameras  # เพิ่มตัวแปร global
    cameras = []
    for camera_id in selected_cameras:
        try:
            camera = CameraThread(camera_id, width=426, height=240)
            cameras.append(camera)
        except Exception as e:
            print(f"ไม่สามารถเริ่มการทำงานของกล้อง {camera_id}: {str(e)}")

    cv2.namedWindow("Multi Camera View")
    
    camera_positions = [(i % 2 * 426, i // 2 * 240) for i in range(len(cameras))]
    cv2.setMouseCallback("Multi Camera View", draw_roi, camera_positions)

    grid = np.zeros((480, 852, 3), dtype=np.uint8)

    while True:
        for i, camera in enumerate(cameras):
            ret, frame = camera.get_frame()
            if ret:
                resized_frame = cv2.resize(frame, (426, 240))
                cam_x, cam_y = camera_positions[i]

                detected_boxes = detect_objects(
                    resized_frame,
                    rois.get(i, None),
                    camera.detection_active,
                    cam_id=i  # ส่ง cam_id เพื่อนับจำนวนรถในแต่ละกล้อง
                )
                draw_detected_objects(resized_frame, detected_boxes, cam_id=i)

                status = "Detection: ON" if camera.detection_active else "Detection: OFF"
                cv2.putText(
                    resized_frame,
                    f"Camera {selected_cameras[i]} - {status}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

                grid[cam_y:cam_y+240, cam_x:cam_x+426] = resized_frame

                if i in rois:
                    for j, (start, end) in enumerate(rois[i]):
                        color = roi_colors[j % len(roi_colors)]
                        cv2.rectangle(
                            grid[cam_y:cam_y+240, cam_x:cam_x+426],
                            start,
                            end,
                            color,
                            2
                        )

        if roi_start and roi_end and selected_camera is not None:
            cam_x, cam_y = camera_positions[selected_camera]
            start_pt = (roi_start[0] + cam_x, roi_start[1] + cam_y)
            end_pt = (roi_end[0] + cam_x, roi_end[1] + cam_y)
            cv2.rectangle(grid, start_pt, end_pt, (0, 0, 255), 1)

        cv2.imshow("Multi Camera View", grid)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    for camera in cameras:
        camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
