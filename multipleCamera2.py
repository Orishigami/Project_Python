import cv2
import numpy as np
import os
from threading import Thread
import time

# Initialize global variables for ROI drawing
roi_start = None
roi_end = None
rois = {}
selected_camera = None

def scan_cameras(max_cameras=10):
    """
    สแกนหากล้องที่สามารถใช้งานได้
    returns: dict ของกล้องที่ใช้งานได้ในรูปแบบ {camera_id: camera_name}
    """
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
                    try:
                        backend_name = cap.getBackendName()
                        if backend_name:
                            camera_name = f"Camera {i} ({backend_name})"
                    except:
                        pass
                    
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
    """
    ให้ผู้ใช้เลือกกล้องที่ต้องการใช้งาน
    returns: list ของ camera_ids ที่ถูกเลือก
    """
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
            actual_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if actual_width != width or actual_height != height:
                print(f"Warning: Requested resolution {width}x{height} but got {actual_width}x{actual_height}")
        except Exception as e:
            print(f"Warning: Could not set camera properties: {str(e)}")
        
        self.frame = None
        self.ret = False
        self.running = True
        self.thread = Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()

    def _update(self):
        consecutive_failures = 0
        while self.running:
            if not self.capture.isOpened():
                consecutive_failures += 1
                print(f"กล้อง {self.camera_id} ถูกตัดการเชื่อมต่อ (ความล้มเหลวครั้งที่ {consecutive_failures})")
                if consecutive_failures >= 5:
                    break
                time.sleep(1)
                continue
                
            try:
                self.ret, frame = self.capture.read()
                if self.ret:
                    self.frame = cv2.resize(frame, (self.width, self.height))
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    print(f"ไม่สามารถอ่านเฟรมจากกล้อง {self.camera_id} (ความล้มเหลวครั้งที่ {consecutive_failures})")
            except Exception as e:
                consecutive_failures += 1
                print(f"เกิดข้อผิดพลาดในการอ่านกล้อง {self.camera_id}: {str(e)}")
            
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

    # คำนวณว่าเมาส์อยู่ในตำแหน่งของกล้องใด
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
        if selected_camera not in rois:
            rois[selected_camera] = []
        rois[selected_camera].append((roi_start, roi_end))
        roi_start, roi_end = None, None
        print(f"ROI added for Camera {selected_camera}: {rois[selected_camera][-1]}")

def main():
    available_cameras = scan_cameras()
    if not available_cameras:
        print("ไม่สามารถเริ่มโปรแกรมได้เนื่องจากไม่พบกล้อง")
        return

    selected_cameras = select_cameras(available_cameras)
    if not selected_cameras:
        print("ไม่ได้เลือกกล้อง จบการทำงาน")
        return

    cameras = []
    for camera_id in selected_cameras:
        try:
            camera = CameraThread(camera_id, width=426, height=240)
            cameras.append(camera)
        except Exception as e:
            print(f"ไม่สามารถเริ่มการทำงานของกล้อง {camera_id}: {str(e)}")

    cv2.namedWindow("Multi Camera View")
    
    # ตำแหน่งของกล้องแต่ละตัวใน grid
    camera_positions = [(i % 2 * 426, i // 2 * 240) for i in range(len(cameras))]
    cv2.setMouseCallback("Multi Camera View", draw_roi, camera_positions)

    grid = np.zeros((480, 852, 3), dtype=np.uint8)

    while True:
        for i, camera in enumerate(cameras):
            ret, frame = camera.get_frame()
            if ret:
                resized_frame = cv2.resize(frame, (426, 240))
                cam_x, cam_y = camera_positions[i]

                cv2.putText(
                    resized_frame, 
                    f"Camera {selected_cameras[i]}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255, 255, 255), 
                    2
                )

                grid[cam_y:cam_y+240, cam_x:cam_x+426] = resized_frame

                if i in rois:
                    for start, end in rois[i]:
                        cv2.rectangle(
                            grid[cam_y:cam_y+240, cam_x:cam_x+426], start, end, (0, 255, 0), 2
                        )

        if roi_start and roi_end:
            cam_idx = selected_camera
            cam_x, cam_y = camera_positions[cam_idx]
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
