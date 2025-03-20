import pyautogui
import keyboard
import time
import json
import win32gui

# Function to get window handle by name
def get_window_handle(window_name):
    def callback(hwnd, hwnd_list):
        if win32gui.IsWindowVisible(hwnd) and window_name in win32gui.GetWindowText(hwnd):
            hwnd_list.append(hwnd)
    hwnds = []
    win32gui.EnumWindows(callback, hwnds)
    return hwnds[0] if hwnds else None

# Wait for F8 to start recording
print("Press F8 to start recording...")
keyboard.wait('F8')

# Get the target window handle
window_name = "Drakensang Online | Онлайн фентъзи играта за твоя браузър - DSO"
hwnd = get_window_handle(window_name)
if hwnd:
    win32gui.SetForegroundWindow(hwnd)
else:
    print(f"Window '{window_name}' not found!")
    exit()

print("Recording started. Press F9 to stop.")

events = []
start_time = time.time()

# Function to record events
def record_event(event_type, data):
    timestamp = time.time() - start_time
    events.append({"type": event_type, "data": data, "time": timestamp})

# Mouse events
def on_click(x, y, button, pressed):
    if pressed:
        record_event("click", {"x": x, "y": y, "button": str(button)})

def on_move(x, y):
    record_event("move", {"x": x, "y": y})

# Keyboard events
def on_key_event(event):
    if event.event_type == "down":
        record_event("key", {"key": event.name})

# Start listeners
from pynput import mouse

mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move)
mouse_listener.start()

keyboard.hook(on_key_event)

# Stop recording on F9
keyboard.wait("F9")
mouse_listener.stop()

# Save to file
with open("recorded_events.txt", "w") as f:
    json.dump(events, f, indent=4)

print("Recording saved!")
