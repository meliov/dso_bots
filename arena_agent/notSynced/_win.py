import numpy as np
import win32gui, win32ui, win32con, win32api
from PIL import Image
import cv2 as cv
import os
import time
import keyboard  # For global hotkey and simulating key presses

# Global automation flag, toggled by F8
automation_enabled = True

def toggle_automation():
    global automation_enabled
    automation_enabled = not automation_enabled
    print("Automation", "enabled" if automation_enabled else "disabled")

# Register F8 as the toggle hotkey.
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
        # Remove the alpha channel and ensure contiguous array.
        img = img[..., :3]
        img = np.ascontiguousarray(img)
        return img

    def generate_image_dataset(self):
        if not os.path.exists("images"):
            os.mkdir("images")
        while True:
            img = self.get_screenshot()
            im = Image.fromarray(img[..., [2, 1, 0]])
            im.save(f"./images/img_{len(os.listdir('images'))}.jpeg")
            time.sleep(1)

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
        with open('../obj.names', 'r') as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            self.classes[i] = line.strip()
        # Colors for drawing boxes (expand if more than six classes).
        self.colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255)
        ]

    def proccess_image(self, img):
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
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
            x = coordinate['x']
            y = coordinate['y']
            w = coordinate['w']
            h = coordinate['h']
            classID = coordinate['class']
            color = self.colors[classID]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.putText(img, self.classes[classID], (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv.imshow('window', img)

# --- Helper functions for clicking and enemy processing ---
def click_object(bbox, wincap):
    # Calculate bottom center point of the rectangle
    cx = bbox['x'] + bbox['w'] // 2
    cy = bbox['y'] + bbox['h']  # Changed to use the bottom (down side)
    abs_x = wincap.left + cx
    abs_y = wincap.top + cy
    win32api.SetCursorPos((abs_x, abs_y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
    print(f"Clicked at bottom center ({abs_x}, {abs_y})")

def process_enemy(bbox, wincap):
    cx = bbox['x'] + bbox['w'] // 2
    cy = bbox['y'] + bbox['h'] // 2
    abs_x = wincap.left + cx
    abs_y = wincap.top + cy
    win32api.SetCursorPos((abs_x, abs_y))
    print(f"Pointed at enemy_skeleton at ({abs_x}, {abs_y})")
    time.sleep(0.05)
    keyboard.press_and_release('5')
    time.sleep(0.05)
    keyboard.press_and_release('1')
    print("Pressed keys: 5 then 1")

# --- State machine variables for arena algorithm ---
# States: "init" -> "phase1" -> "phase2" -> "phase3" -> "reset"
initial_side = None    # "left" or "right"
state = "init"
first_round = True     # Only on first round will 'm' be double-clicked
last_click_time = time.time()

# Setup window capture and image processor.
window_name = "Drakensang Online | Онлайн фентъзи играта за твоя браузър - DSO"  # Replace with your game's window title.
cfg_file_name = "../yolov4-tiny-custom.cfg"
weights_file_name = "../new.weights"
wincap = WindowCapture(window_name)
improc = ImageProcessor(wincap.get_window_size(), cfg_file_name, weights_file_name)

print("Press F8 to toggle automation on/off. Press 'q' (in the OpenCV window) to quit.")

while True:

    ss = wincap.get_screenshot()
    coordinates = improc.proccess_image(ss)
    key = cv.waitKey(1)
    if key == ord('q'):
        cv.destroyAllWindows()
        break

    # Build a dictionary of detected objects (first occurrence only).
    objects = {}
    for coord in coordinates:
        name = coord['class_name']
        if name not in objects:
            objects[name] = coord

    current_time = time.time()

    # --- If automation is disabled, skip arena processing (but still allow start/rematch) ---
    if not automation_enabled:
        time.sleep(0.05)
        continue

    # --- Global reset via start/rematch (keep original behavior) ---
    if 'start' in objects or 'rematch' in objects:
        if current_time - last_click_time >= 0.5:
            if 'start' in objects:
                click_object(objects['start'], wincap)
            if 'rematch' in objects:
                click_object(objects['rematch'], wincap)
            last_click_time = current_time
        # Reset arena state.
        initial_side = None
        state = "init"
        first_round = True
        print("Resetting state after start/rematch.")
        time.sleep(0.05)
        continue



    # --- Priority: if enemy_skeleton is visible, process ONLY that.
    if 'enemy_skeleton' in objects:
        if current_time - last_click_time >= 0.5:
            process_enemy(objects['enemy_skeleton'], wincap)
            last_click_time = current_time
        # Force round to "reset" once enemy is handled.
        state = "reset"
        time.sleep(0.05)
        continue

    # --- Arena state machine ---
    if state == "init":
        # Determine the initial side based on which one is visible.
        if 'left' in objects and 'right' not in objects:
            initial_side = 'left'
        elif 'right' in objects and 'left' not in objects:
            initial_side = 'right'
        elif 'left' in objects and 'right' in objects:
            # If both appear, default to left.
            initial_side = 'left'
        if initial_side is not None:
            if first_round and 'enemy_skeleton' not in objects:
                keyboard.press_and_release('m')
                keyboard.press_and_release('m')
                first_round = False
                print("Double 'm' pressed for first round initialization.")
            state = "phase1"
            print("Initial side set to", initial_side, "- moving to phase1.")

    elif state == "phase1":
        # Only click the initial side until center appears.
        if initial_side in objects and 'center' not in objects:
            if current_time - last_click_time >= 0.5:
                click_object(objects[initial_side], wincap)
                last_click_time = current_time
        if 'center' in objects:
            state = "phase2"
            print("Center detected; moving to phase2.")

    elif state == "phase2":
        # Only click the center until the opposite side appears.
        if 'center' in objects:
            if current_time - last_click_time >= 0.5:
                click_object(objects['center'], wincap)
                last_click_time = current_time
        opposite_side = 'left' if initial_side == 'right' else 'right'
        if opposite_side in objects:
            state = "phase3"
            print("Opposite side (" + opposite_side + ") detected; moving to phase3.")

    elif state == "phase3":
        # Only click the opposite side.
        opposite_side = 'left' if initial_side == 'right' else 'right'
        if opposite_side in objects:
            if current_time - last_click_time >= 0.5:
                click_object(objects[opposite_side], wincap)
                last_click_time = current_time
        # In phase3, we wait for the enemy_skeleton to appear (handled above).

    elif state == "reset":
        # In reset, wait until ONLY the initial side is visible (no center, no opposite side).
        disallowed = ['center']
        opposite_side = 'left' if initial_side == 'right' else 'right'
        disallowed.append(opposite_side)
        if initial_side in objects and all(obj not in objects for obj in disallowed):
            state = "phase1"
            print("Reset complete (only initial side visible); new round starting.")

    time.sleep(0.05)
    #securing new attempt to reconnect is clicked
    win32api.SetCursorPos((950, 980))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)

    if initial_side != 'left' and initial_side != 'init' and initial_side is not None:
        win32api.SetCursorPos((581, 545))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        time.sleep(0.05)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
        time.sleep(0.3)


print('Finished.')
