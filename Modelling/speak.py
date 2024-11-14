import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 175)
    engine.setProperty('volume', 1.0)
    voices = engine.getProperty('voices')
    if len(voices) > 1:
        engine.setProperty('voice', voices[0].id)
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    print("Testing speak function...")
    speak("This is a test of the text-to-speech functionality.")
    print("Test complete.")