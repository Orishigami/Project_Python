import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

class MultiCamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Camera Display")
        
        self.cams = self.detect_cameras()
        self.labels = []

        for i in range(len(self.cams)):
            label = Label(root)
            label.grid(row=i//2, column=i%2)  # แสดงกล้อง 2 ตัวต่อแถว
            self.labels.append(label)
        
        self.update_frames()

    def detect_cameras(self, max_cameras=10):
        # ฟังก์ชันนี้ใช้ในการสแกนหากล้องทั้งหมด
        cams = []
        for cam_id in range(max_cameras):
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                cams.append(cap)  # เพิ่มกล้องที่เชื่อมต่อได้ลงในรายการ
            else:
                cap.release()  # ปล่อยการเชื่อมต่อหากไม่มีการตอบสนองจากกล้อง
        return cams

    def update_frames(self):
        for i, cam in enumerate(self.cams):
            ret, frame = cam.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.labels[i].imgtk = imgtk
                self.labels[i].configure(image=imgtk)
        
        self.root.after(10, self.update_frames)

    def close_cams(self):
        for cam in self.cams:
            cam.release()
        self.root.destroy()

def start_multi_cam():
    root = tk.Tk()
    app = MultiCamApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close_cams)
    root.mainloop()

if __name__ == "__main__":
    start_multi_cam()
