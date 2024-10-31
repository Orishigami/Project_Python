import cv2
import numpy as np
import os
from threading import Thread
import time

def scan_cameras(max_cameras=10):
    """
    สแกนหากล้องที่สามารถใช้งานได้
    returns: dict ของกล้องที่ใช้งานได้ในรูปแบบ {camera_id: camera_name}
    """
    available_cameras = {}
    
    print("กำลังสแกนหากล้องที่ใช้งานได้...")
    for i in range(max_cameras):
        try:
            # ทดลองเปิดด้วย DirectShow ก่อน
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                # ถ้าไม่สำเร็จ ลองใช้ default backend
                cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    # พยายามอ่านชื่อกล้อง
                    camera_name = f"Camera {i}"
                    try:
                        backend_name = cap.getBackendName()
                        if backend_name:
                            camera_name = f"Camera {i} ({backend_name})"
                    except:
                        pass
                    
                    # เก็บข้อมูลความละเอียด
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
        
        # ถ้าไม่ได้ป้อนอะไร ใช้ทุกกล้องที่มี (สูงสุด 4 ตัว)
        if not user_input:
            selected_cameras = list(available_cameras.keys())[:4]
            break
            
        try:
            # แปลงข้อมูลที่ป้อนเป็น list ของตัวเลข
            selected = [int(x) for x in user_input.split()]
            
            # ตรวจสอบความถูกต้อง
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
        
        # ลองเปิดกล้องด้วยวิธีต่างๆ
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
        
        # ตั้งค่า properties
        try:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            # ตรวจสอบว่าการตั้งค่าสำเร็จหรือไม่
            actual_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if actual_width != width or actual_height != height:
                print(f"Warning: Requested resolution {width}x{height} but got {actual_width}x{actual_height}")
        except Exception as e:
            print(f"Warning: Could not set camera properties: {str(e)}")
        
        self.frame = None
        self.ret = False
        self.running = True
        
        # เริ่ม thread
        self.thread = Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()

    def _update(self):
        consecutive_failures = 0
        while self.running:
            if not self.capture.isOpened():
                consecutive_failures += 1
                print(f"กล้อง {self.camera_id} ถูกตัดการเชื่อมต่อ (ความล้มเหลวครั้งที่ {consecutive_failures})")
                if consecutive_failures >= 5:  # ถ้าล้มเหลว 5 ครั้งติดกัน ให้หยุด thread
                    break
                time.sleep(1)  # รอสักครู่ก่อนลองใหม่
                continue
                
            try:
                self.ret, frame = self.capture.read()
                if self.ret:
                    self.frame = cv2.resize(frame, (self.width, self.height))
                    consecutive_failures = 0  # reset ตัวนับเมื่ออ่านภาพสำเร็จ
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
        return self.ret, self.frame.copy()  # ส่ง copy ของเฟรมเพื่อป้องกันการแก้ไขข้าม thread

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)  # รอ thread สูงสุด 1 วินาที
        if self.capture is not None:
            self.capture.release()

def main():
    # สแกนหากล้องที่ใช้งานได้
    available_cameras = scan_cameras()
    if not available_cameras:
        print("ไม่สามารถเริ่มโปรแกรมได้เนื่องจากไม่พบกล้อง")
        return

    # ให้ผู้ใช้เลือกกล้อง
    selected_cameras = select_cameras(available_cameras)
    if not selected_cameras:
        print("ไม่ได้เลือกกล้อง จบการทำงาน")
        return

    # Initialize cameras
    cameras = []
    for camera_id in selected_cameras:
        try:
            camera = CameraThread(camera_id, width=426, height=240)
            cameras.append(camera)
        except Exception as e:
            print(f"ไม่สามารถเริ่มการทำงานของกล้อง {camera_id}: {str(e)}")

    # Display cameras
    cv2.namedWindow("Multi Camera View")

    # ปรับขนาด grid เป็น 2x2 โดยใช้ขนาด 426x240
    grid = np.zeros((480, 852, 3), dtype=np.uint8)  # ขนาดรวมคือ 2 x (426x240)

    while True:
        for i, camera in enumerate(cameras):
            ret, frame = camera.get_frame()
            if ret:
                # ปรับขนาดเฟรม
                resized_frame = cv2.resize(frame, (426, 240))

                # เพิ่มข้อความหมายเลขกล้อง
                cv2.putText(
                    resized_frame, 
                    f"Camera {selected_cameras[i]}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255, 255, 255), 
                    2
                )

                # วางเฟรมในตำแหน่งที่เหมาะสมใน grid
                grid_x = (i % 2) * 426
                grid_y = (i // 2) * 240
                grid[grid_y:grid_y+240, grid_x:grid_x+426] = resized_frame

        cv2.imshow("Multi Camera View", grid)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Cleanup
    for camera in cameras:
        camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()