import os
import time
import threading
import pyautogui
import keyboard

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def take_screenshots():
    global running
    counter = 1
    ensure_directory("images")
    while True:
        if running:
            screenshot = pyautogui.screenshot()
            screenshot.save(f"images/img_{counter}.png")
            counter = counter + 1
            print("took a screenshot")
        time.sleep(1)

def toggle_screenshot():
    global running
    running = not running
    status = "Started" if running else "Stopped"
    print(f"Screenshot capture {status}")

running = False
keyboard.add_hotkey("F8", toggle_screenshot)

thread = threading.Thread(target=take_screenshots, daemon=True)
thread.start()

print("Press F8 to start/stop screenshot capture.")
keyboard.wait()