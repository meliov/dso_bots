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

        with open('obj.names', 'r') as file:
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
    # Calculate center of bounding box.
    cx = bbox['x'] + bbox['w'] // 2
    cy = bbox['y'] + bbox['h'] // 2
    # Convert to absolute screen coordinates.
    abs_x = wincap.left + cx
    abs_y = wincap.top + cy
    win32api.SetCursorPos((abs_x, abs_y))
    # Simulate mouse click.
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
    print(f"Clicked at ({abs_x}, {abs_y})")


def process_enemy(bbox, wincap):
    # Move mouse to enemy_skeleton's center.
    cx = bbox['x'] + bbox['w'] // 2
    cy = bbox['y'] + bbox['h'] // 2
    abs_x = wincap.left + cx
    abs_y = wincap.top + cy
    win32api.SetCursorPos((abs_x, abs_y))
    print(f"Pointed at enemy_skeleton at ({abs_x}, {abs_y})")
    time.sleep(0.05)
    # Press keys in order: first '5' then '1'
    keyboard.press_and_release('5')
    time.sleep(0.05)
    keyboard.press_and_release('1')
    print("Pressed keys: 5 then 1")


# --- State machine variables ---
initial_side = None         # Current initial side for the battle round ('left' or 'right')
movement_state = "init"     # Can be: init, phase1, phase2, phase3, enemy, reset
last_click_time = time.time()
first_round = True          # Flag to trigger double 'm' click only on the first round
round_completed = False     # Flag to indicate enemy processing is done
prev_initial_side = None    # To store the initial side from the previous round

# Setup window capture and image processor.
window_name = "Drakensang Online | Онлайн фентъзи играта за твоя браузър - DSO"  # Replace with your game's window title.
cfg_file_name = "yolov4-tiny-custom.cfg"
weights_file_name = "new.weights"

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

    if automation_enabled:
        # --- Start/Rematch button: reset state for a new battle round.
        if 'start' in objects or 'rematch' in objects:
            if current_time - last_click_time >= 0.5:
                if 'start' in objects:
                    click_object(objects['start'], wincap)
                if 'rematch' in objects:
                    click_object(objects['rematch'], wincap)
                last_click_time = current_time
            # Reset state variables for a new round.
            first_round = True
            round_completed = False
            prev_initial_side = None
            initial_side = None
            movement_state = "init"
            time.sleep(0.05)
            continue

        # --- Enemy Phase: When enemy_skeleton is detected, process it.
        if 'enemy_skeleton' in objects and movement_state != "enemy":
            movement_state = "enemy"
            if current_time - last_click_time >= 0.5:
                process_enemy(objects['enemy_skeleton'], wincap)
                last_click_time = current_time
            # Save the current initial side for the next round and mark round as complete.
            prev_initial_side = initial_side
            round_completed = True
            movement_state = "reset"
            time.sleep(0.05)
            continue

        # --- Reset Phase: Wait until the expected initial side reappears to start a new cycle.
        if round_completed:
            # Prefer to restart with the previous initial side if it is still visible.
            if prev_initial_side and prev_initial_side in objects:
                print("Resetting for next battle round. (Retaining side:", prev_initial_side,")")
                initial_side = prev_initial_side
                movement_state = "phase1"
                round_completed = False
            # Else, do nothing until the expected side appears.
            time.sleep(0.05)
            continue

        # --- Detection Phase: Determine the initial side if not yet set.
        if initial_side is None:
            # If we have a previous side (from a completed round), try to use it.
            if prev_initial_side and prev_initial_side in objects:
                initial_side = prev_initial_side
                movement_state = "phase1"
                print("Initial side re-detected as", initial_side)
            else:
                # If only one side is visible, choose it.
                # (If both are visible, we now prioritize based on which one appears first.)
                if 'right' in objects and 'left' not in objects:
                    initial_side = 'right'
                    movement_state = "phase1"
                    print("Initial side set to right")
                elif 'left' in objects and 'right' not in objects:
                    initial_side = 'left'
                    movement_state = "phase1"
                    print("Initial side set to left")
                elif 'left' in objects and 'right' in objects:
                    # If both appear and no prior info exists, choose based on previous round preference;
                    # if first round, you might choose the one that’s more reliable. Here we default to left.
                    initial_side = 'left'
                    movement_state = "phase1"
                    print("Both sides detected; defaulting initial side to left")
            # On the very first round, press 'm' twice (if enemy not already detected).
            if first_round and 'enemy_skeleton' not in objects:
                keyboard.press_and_release('m')
                keyboard.press_and_release('m')
                first_round = False
                print("Double 'm' pressed for first round initialization")

        # --- Movement Phases ---
        if movement_state == "phase1":
            # Click the initial side repeatedly until center appears.
            if initial_side == 'left':
                if 'left' in objects and 'center' not in objects:
                    if current_time - last_click_time >= 0.5:
                        click_object(objects['left'], wincap)
                        last_click_time = current_time
                if 'center' in objects:
                    movement_state = "phase2"
                    print("Transition to phase2 (center) from left")
            elif initial_side == 'right':
                if 'right' in objects and 'center' not in objects:
                    if current_time - last_click_time >= 0.5:
                        click_object(objects['right'], wincap)
                        last_click_time = current_time
                if 'center' in objects:
                    movement_state = "phase2"
                    print("Transition to phase2 (center) from right")

        elif movement_state == "phase2":
            # Click center repeatedly until the opposite side appears.
            if 'center' in objects:
                if current_time - last_click_time >= 0.5:
                    click_object(objects['center'], wincap)
                    last_click_time = current_time
            # Transition when the opposite side is detected.
            if initial_side == 'left' and 'right' in objects:
                movement_state = "phase3"
                print("Transition to phase3 (right) from center")
            elif initial_side == 'right' and 'left' in objects:
                movement_state = "phase3"
                print("Transition to phase3 (left) from center")

        elif movement_state == "phase3":
            # Click the opposite side repeatedly until enemy_skeleton appears.
            if initial_side == 'left':
                if 'right' in objects:
                    if current_time - last_click_time >= 0.5:
                        click_object(objects['right'], wincap)
                        last_click_time = current_time
            elif initial_side == 'right':
                if 'left' in objects:
                    if current_time - last_click_time >= 0.5:
                        click_object(objects['left'], wincap)
                        last_click_time = current_time

    time.sleep(0.05)

print('Finished.')
