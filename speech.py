import speech_recognition as sr
import pyttsx3

r = sr.Recognizer()
while True:
    try:
        with sr.Microphone() as mic:
            r.adjust_for_ambient_noise(mic, duration=0.2)

            print("Say something:")
            audio = r.listen(mic)
            text = r.recognize_google(audio)
            text = text.lower()
            print(f"recognized text: {text}")

    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        r = sr.Recognizer()
        continue
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
        break
