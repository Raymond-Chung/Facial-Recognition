import mediapipe as mp
import os


print(f"MediaPipe Version: {mp.__version__}")
print(f"MediaPipe Path: {os.path.dirname(mp.__file__)}")

try:
    print(f"Solutions available: {mp.solutions}")
except AttributeError:
    print("Solutions NOT found!")

# import mediapipe as mp
# from mediapipe.python.solutions import hands as mp_hands

# print(f"Direct import success: {mp_hands is not None}")