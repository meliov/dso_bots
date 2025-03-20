import pyautogui
import keyboard
import time
import json
import win32gui

# Load recorded events
with open("recorded_events.txt", "r") as f:
    events = json.load(f)

print("Press F8 to start playback...")
keyboard.wait("F8")

# Get the window handle
window_name = "Drakensang Online | Онлайн фентъзи играта за твоя браузър - DSO"
hwnd = win32gui.FindWindow(None, window_name)
if hwnd:
    win32gui.SetForegroundWindow(hwnd)
else:
    print(f"Window '{window_name}' not found!")
    exit()

print("Replaying actions... Press F9 to stop.")

start_time = time.time()
for event in events:
    if keyboard.is_pressed("F9"):  # Stop playback if F9 is pressed
        print("Playback stopped!")
        break

    elapsed_time = event["time"]
    current_time = time.time() - start_time

    # Wait for the right timing
    if elapsed_time > current_time:
        time.sleep(elapsed_time - current_time)

    # Execute the recorded event
    if event["type"] == "click":
        pyautogui.click(event["data"]["x"], event["data"]["y"])
    elif event["type"] == "move":
        pyautogui.moveTo(event["data"]["x"], event["data"]["y"])
    elif event["type"] == "key":
        pyautogui.press(event["data"]["key"])

print("Playback finished!")
