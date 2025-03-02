import numpy as np
import win32gui, win32ui, win32con, win32api
from PIL import Image
import cv2 as cv
import os
import time
import keyboard  # For global hotkey and simulating key presses

# Global automation flag, toggled with F8.
automation_enabled = True

def toggle_automation():
    global automation_enabled
    automation_enabled = not automation_enabled
    print("Automation", "enabled" if automation_enabled else "disabled")

# Toggle automation with F8.
keyboard.add_hotkey('F8', toggle_automation)

class WindowCapture:
    w = 0
    h = 0
    hwnd = None
    left = 0
    top = 0

    def __init__(self, window_name):
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(window_name))
        window_rect = win32gui.GetWindowRect(self.hwnd)
        border_pixels = 8
        titlebar_pixels = 30
        # Calculate the absolute coordinates of the client area.
        self.left = window_rect[0] + border_pixels
        self.top = window_rect[1] + titlebar_pixels
        self.w = (window_rect[2] - window_rect[0]) - (border_pixels * 2)
        self.h = (window_rect[3] - window_rect[1]) - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

    def get_screenshot(self):
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        # Remove alpha channel and ensure contiguous array.
        img = img[..., :3]
        img = np.ascontiguousarray(img)
        return img

    def get_window_size(self):
        return (self.w, self.h)

class ImageProcessor:
    W = 0
    H = 0
    net = None
    ln = None
    classes = {}
    colors = []

    def __init__(self, img_size, cfg_file, weights_file):
        np.random.seed(42)
        self.net = cv.dnn.readNetFromDarknet(cfg_file, weights_file)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.W = img_size[0]
        self.H = img_size[1]
        with open('obj.names', 'r') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            self.classes[i] = line.strip()
        # Colors for drawing boxes.
        self.colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255)
        ]

    def proccess_image(self, img):
        blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.ln)
        outputs = np.vstack(outputs)
        coordinates = self.get_coordinates(outputs, 0.5)
        self.draw_identified_objects(img, coordinates)
        return coordinates

    def get_coordinates(self, outputs, conf):
        boxes = []
        confidences = []
        classIDs = []
        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf:
                x, y, w, h = output[:4] * np.array([self.W, self.H, self.W, self.H])
                p0 = int(x - w // 2), int(y - h // 2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)
        indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf - 0.1)
        if len(indices) == 0:
            return []
        coordinates = []
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            coordinates.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'class': classIDs[i],
                'class_name': self.classes[classIDs[i]]
            })
        return coordinates

    def draw_identified_objects(self, img, coordinates):
        for coordinate in coordinates:
            x, y, w, h = coordinate['x'], coordinate['y'], coordinate['w'], coordinate['h']
            classID = coordinate['class']
            color = self.colors[classID]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.putText(img, self.classes[classID], (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv.imshow('window', img)

# Helper: click on an object (clicks at the center of its bounding box).
def click_object(bbox, wincap):
    cx = bbox['x'] + bbox['w'] // 2
    cy = bbox['y'] + bbox['h'] // 2
    abs_x = wincap.left + cx
    abs_y = wincap.top + cy
    win32api.SetCursorPos((abs_x, abs_y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
    print(f"Clicked at ({abs_x}, {abs_y})")

# Helper: perform attack by pointing at the given object and pressing key "5".
def attack_object(bbox, wincap):
    cx = bbox['x'] + bbox['w'] // 2
    cy = bbox['y'] + bbox['h'] // 2
    abs_x = wincap.left + cx
    abs_y = wincap.top + cy
    win32api.SetCursorPos((abs_x, abs_y))
    print(f"Attacking at ({abs_x}, {abs_y}) with key '5'")
    keyboard.press_and_release('5')

# --- State variables ---
# current_mode: either "left" or "right" based on the first detected side.
# attack_done: ensures the attack (key "5") is only triggered once per cycle.
current_mode = None
attack_done = False
last_click_time = time.time()

# Setup window capture and image processor.
window_name = "Drakensang Online | Онлайн фентъзи играта за твоя браузър - DSO"  # Replace with your game's window title.
cfg_file_name = "yolov4-tiny-custom.cfg"
weights_file_name = "yolov4-tiny-custom_last.weights"
wincap = WindowCapture(window_name)
improc = ImageProcessor(wincap.get_window_size(), cfg_file_name, weights_file_name)

print("Press F8 to toggle automation on/off. Press 'q' in the OpenCV window to quit.")

while True:
    ss = wincap.get_screenshot()
    coordinates = improc.proccess_image(ss)
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

    # Build a dictionary with the first occurrence of each detected object.
    objects = {}
    for coord in coordinates:
        name = coord['class_name']
        if name not in objects:
            objects[name] = coord

    current_time = time.time()

    if automation_enabled:
        # If no cycle has been selected, choose based on which side is detected.
        if current_mode is None:
            if 'left' in objects:
                current_mode = 'left'
                attack_done = False
                print("Left cycle started")
            elif 'right' in objects:
                current_mode = 'right'
                attack_done = False
                print("Right cycle started")
        else:
            if current_mode == 'left':
                # If "center" is not visible, click "left" repeatedly.
                if 'center' not in objects:
                    if 'left' in objects and current_time - last_click_time >= 0.5:
                        click_object(objects['left'], wincap)
                        last_click_time = current_time
                else:
                    # When "center" is visible, click "center" repeatedly.
                    if current_time - last_click_time >= 0.5:
                        click_object(objects['center'], wincap)
                        last_click_time = current_time
                # Once the "right" object is detected, attack immediately.
                if 'right' in objects and not attack_done:
                    if current_time - last_click_time >= 0.5:
                        attack_object(objects['right'], wincap)
                        attack_done = True
                        last_click_time = current_time
                # After attack, wait until "center" is gone and "left" reappears to reset.
                if attack_done and ('center' not in objects) and ('left' in objects):
                    print("Resetting left cycle")
                    current_mode = None
                    attack_done = False

            elif current_mode == 'right':
                # If "center" is not visible, click "right" repeatedly.
                if 'center' not in objects:
                    if 'right' in objects and current_time - last_click_time >= 0.5:
                        click_object(objects['right'], wincap)
                        last_click_time = current_time
                else:
                    # When "center" is visible, click "center" repeatedly.
                    if current_time - last_click_time >= 0.5:
                        click_object(objects['center'], wincap)
                        last_click_time = current_time
                # Once the "left" object is detected, attack immediately.
                if 'left' in objects and not attack_done:
                    if current_time - last_click_time >= 0.5:
                        attack_object(objects['left'], wincap)
                        attack_done = True
                        last_click_time = current_time
                # After attack, wait until "center" is gone and "right" reappears to reset.
                if attack_done and ('center' not in objects) and ('right' in objects):
                    print("Resetting right cycle")
                    current_mode = None
                    attack_done = False

    time.sleep(0.05)

print("Finished.")
