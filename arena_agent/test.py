import pyautogui
import time

def is_blue(x, y):
    r, g, b = pyautogui.pixel(x, y)
    return b > 150

def is_red(color):
    r, g, b = color
    return r > 150

while True:  # Run the loop 10 times

    print(pyautogui.pixel(1327, 790))

    if is_blue(pyautogui.pixel(1327, 790)):
        print("Blue")
    elif is_red(pyautogui.pixel(1327, 790)):
        print("Red")

    time.sleep(1)  # Wait for 1 second before checking again