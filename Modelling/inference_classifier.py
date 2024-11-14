import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import google.generativeai as genai
from speak import speak 
from dotenv import load_dotenv
from os import getenv
import threading  # For running speak function without blocking
import sys
import tkinter as tk
from tkinter import ttk 
from PIL import Image, ImageTk
from ttkthemes import ThemedTk  

load_dotenv()

KEY1 = getenv("KEY1AM")  
if not KEY1:
    print("Please set your Gemini API key in the .env file.")
    sys.exit(1)

genai.configure(api_key=KEY1)
try:
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    sys.exit(1)

conversation_history = []
def get_response(user_message):
    global conversation_history
    conversation_history.append({"role": "user", "content": user_message})
    prompt = ""
    
    for turn in conversation_history:
        prompt += f"{turn['role']}: {turn['content']}\n"
    
    prompt += (
        "assistant: (If the question is related to 'flight delay', simulate a response assuming the flight is delayed by a random "
        "amount between 1 and 5 hours. Apologize for the delay, and offer food coupons or assistance with rebooking without asking "
        "for more details. For all other inquiries, respond as a typical airport assistant would.)"
    )
    response = model.generate_content(prompt)
    conversation_history.append({"role": "assistant", "content": response.text})
    return response.text

print("Welcome to the Airport Virtual Navigation Assistant!")

MODEL_PATH = 'model.p'
NUM_LANDMARKS = 21

try:
    with open(MODEL_PATH, 'rb') as f:
        model_dict = pickle.load(f)
    gesture_model = model_dict['model']
except FileNotFoundError:
    print("Model file not found. Ensure 'model.p' exists in the current directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

labels_dict = {
    0: 'I', 1: 'restaurant',2:'yes', 4: 'ticket',
    6: 'want', 7: 'is',  9: 'begin', 10: 'delayed',
    11: 'flight', 12: 'gate', 13: 'help', 14: 'where' , 15:'stop'
}

cap = cv2.VideoCapture(0)  # Use 0 for the default camera
if not cap.isOpened():
    print("Camera not initialized properly. Check camera index or permissions.")
    sys.exit(1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

gesture_counts = {}
sentence = ""
api_response = ""

begin_count = 0
stop_count = 0
sentence_started = False

DETECTION_DELAY = 1.0
last_detection_time = 0

def extract_landmarks(hand_landmarks):
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append(landmark.x)
        landmarks.append(landmark.y)
    return landmarks

def speak_response(text):
    threading.Thread(target=speak, args=(text,)).start()

root = ThemedTk(theme="equilux")  
root.title("Airport Virtual Navigation Assistant")
root.geometry("900x700")

style = ttk.Style(root)
style.configure('TLabel', font=('Helvetica', 14))
style.configure('TButton', font=('Helvetica', 12))
style.configure('TFrame', background=style.lookup('TLabel', 'background'))

video_frame = ttk.Frame(root)
video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

info_frame = ttk.Frame(root)
info_frame.pack(side=tk.BOTTOM, fill=tk.X)

video_label = ttk.Label(video_frame)
video_label.pack()

gesture_label = ttk.Label(info_frame, text="Gesture: ")
gesture_label.pack(pady=5)

detected_word_label = ttk.Label(info_frame, text="Detected Word: ")
detected_word_label.pack(pady=5)

sentence_label = ttk.Label(info_frame, text="Sentence: ")
sentence_label.pack(pady=5)

response_label = ttk.Label(info_frame, text="Response: ", wraplength=880, justify="left")
response_label.pack(pady=5)

def update_frame():
    global gesture_counts, sentence, api_response
    global begin_count, stop_count, sentence_started
    global last_detection_time
    current_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        root.after(10, update_frame)
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    left_hand_landmarks = [0.0] * (NUM_LANDMARKS * 2)
    right_hand_landmarks = [0.0] * (NUM_LANDMARKS * 2)

    if results.multi_hand_landmarks:
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            if handedness.classification[0].label == "Left":
                left_hand_landmarks = extract_landmarks(hand_landmarks)
            elif handedness.classification[0].label == "Right":
                right_hand_landmarks = extract_landmarks(hand_landmarks)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        try:
            features = np.concatenate([left_hand_landmarks, right_hand_landmarks])
            prediction = gesture_model.predict([features])[0]
            predicted_character = labels_dict.get(prediction, "Unknown")
        except Exception as e:
            print(f"Error during prediction: {e}")
            predicted_character = "Error"

    else:
        predicted_character = "No Detected"

    if current_time - last_detection_time >= DETECTION_DELAY:
        if predicted_character not in ["Unknown", "Error", "No Detected"]:
            gesture_label.config(text=f"Gesture: {predicted_character}")
            detected_word_label.config(text=f"Detected Word: {predicted_character}")
            last_detection_time = current_time

            if predicted_character == "begin":
                begin_count += 1
                stop_count = 0  
                if begin_count >= 3 and not sentence_started:
                    sentence_started = True
                    begin_count = 0
                    gesture_counts = {}
                    sentence = ""
                    sentence_label.config(text="Sentence: ")
                    response_label.config(text="Response: ")
                    speak_response("Sentence formation started.")
            elif predicted_character in ["stop"]:
                stop_count += 1
                begin_count = 0
                if stop_count >= 3 and sentence_started:
                    sentence_started = False
                    stop_count = 0
                    if sentence.strip() != "":
                        full_message = sentence.strip() + " (You are an assistant at Chennai airport. Interpret the user's intent even if the sentence is incomplete or contains only a few words. Provide concise, formal, and informative responses as an airport assistant would. If there is any question which requires more details, just assume sample details. NEVER ASK ANY QUESTION , JUST ASSUME AND ANSWER IT.)"
                        assistant_response = get_response(full_message)
                        response_label.config(text=f"Response: {assistant_response}")
                        speak_response(assistant_response)
                        sentence = ""
                        sentence_label.config(text="Sentence: ")
                        speak_response("Sentence formation ended.")
                    gesture_counts = {}
            else:
                if sentence_started and predicted_character not in ["begin", "stop"]:
                    gesture_counts[predicted_character] = gesture_counts.get(predicted_character, 0) + 1
                    if gesture_counts[predicted_character] >= 5:
                        sentence += predicted_character + " "
                        sentence_label.config(text=f"Sentence: {sentence}")
                        gesture_counts[predicted_character] = 0

        else:
            gesture_label.config(text=f"Gesture: {predicted_character}")
            detected_word_label.config(text=f"Detected Word: {predicted_character}")
    else:
        gesture_label.config(text=f"Gesture: {predicted_character} (Ignored)")
        detected_word_label.config(text=f"Detected Word: {predicted_character} (Ignored)")

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

update_frame()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
