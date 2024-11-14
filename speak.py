import pyttsx3

def speak(text):
    engine = pyttsx3.init()

    """RATE"""
    rate = engine.getProperty('rate')  
    engine.setProperty('rate', 175)    # Setting slower rate (adjust as needed)

    """VOLUME"""
    volume = engine.getProperty('volume')  
    engine.setProperty('volume', 1.0)   # Setting full volume (adjust as needed)

    """VOICE"""
    voices = engine.getProperty('voices')
    if len(voices) > 1:
        engine.setProperty('voice', voices[0].id)  # Change to the second voice (female, if available)

    engine.say(text)
    engine.runAndWait()