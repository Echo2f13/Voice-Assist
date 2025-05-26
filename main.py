import tkinter as tk
from tkinter import ttk, filedialog, PhotoImage, Label
import speech_recognition as sr
import pyttsx3
import webbrowser
import os
import subprocess
from threading import Thread
from PIL import Image, ImageTk
import pandas as pd
import pyautogui
import time
import yt_dlp
import urllib.request
import urllib.parse
import re
from pathlib import Path
import sys
import json
from datetime import datetime
import csv
import random
from pathlib import Path
import math
import string
import pyperclip 

ydl = yt_dlp.YoutubeDL()

def extract_youtube_urls(search_query):
    """Extract YouTube video URLs from search results."""
    encoded_query = urllib.parse.urlencode({"search_query": search_query})
    url = "https://www.youtube.com/results?" + encoded_query

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(req)
        html_content = response.read().decode()

        # Extract video IDs using regex
        video_id_pattern = r'"videoId":"([^"]{11})"'
        video_ids = re.findall(video_id_pattern, html_content)

        # Remove duplicates while preserving order
        unique_ids = []
        seen = set()
        for vid_id in video_ids:
            if vid_id not in seen:
                seen.add(vid_id)
                unique_ids.append(vid_id)

        # Convert to full YouTube URLs
        youtube_urls = [f"https://music.youtube.com/watch?v={vid_id}" for vid_id in unique_ids]

        return youtube_urls

    except Exception as e:
        print(f"[Error extracting YouTube URLs] {e}")
        return []
    
class VoiceAssistant:
    def __init__(self, root):
        self.root = root
        self.recording = False
        self.setup_ui()
        self.setup_voice_engine()
        self.recognizer = sr.Recognizer()
        self.calendar_file = "local_calendar.json"
        self._init_calendar_file()
        
    def setup_voice_engine(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self):
        with sr.Microphone() as source:
            self.update_status("Listening...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source, timeout=10)
            
        try:
            command = self.recognizer.recognize_google(audio)
            self.update_status(f"You said: {command}")
            return command.lower()
        except sr.UnknownValueError:
            self.update_status("Sorry, I didn't catch that.")
            return ""
        except sr.RequestError:
            self.update_status("Network error.")
            return ""
        except sr.WaitTimeoutError:
            self.update_status("Listening timed out.")
            return ""


    def handle_command(self, command):
        if not command:
            return
            
        self.update_status(f"Executing: {command}")
        
        if "search" in command:
            query = command.replace("search", "").strip()
            url = f"https://www.google.com/search?q={query}"
            webbrowser.open(url)
            self.speak(f"Searching for {query} on Google")
        elif "play" in command and ("song" in command or "music" in command):
            self.play_youtube_music(command)
        elif "read" in command and "pdf" in command:
            self.read_file(command)
        elif "open" in command:
            self.open_website(command)
        elif "add event" in command:
            self.add_event()
        elif ("csv" in command.lower() and "resume" not in command.lower()):
            self.analysis_csv()
        elif "resume" in command.lower():
            self.analysis_resume()
        elif "joke" in command:
            self.crack_a_joke()
        elif "create" in command:
            self.instant_meet(command)
        # elif "analyze CSV" or "CSV" in command:
        #     self.analysis_csv()
        # elif "resume" in command:
        #     self.analysis_resume()
        elif "exit" in command or "quit" in command:
            self.speak("Goodbye!")
            self.root.after(1000, self.root.destroy)
        else:
            self.speak("Command not recognized yet.")


# Part of a class, assumes `self.speak()` and `self.listen()` exist
    def play_youtube_music(self, command):
        print("Playing music on youtube music")
        query = command.replace("play", "").replace("song", "").replace("music", "").strip()
        if not query:
            self.speak("What song would you like me to play?")
            query = self.listen()

        if query:
            try:
                print(f"[youtube] Extracting URLs for: {query}")
                urls = extract_youtube_urls(query)
                print("got the song")
                for i, url in enumerate(urls[:5], 1):
                    print(f"[youtube] URL {i}: {url}")

                if urls:
                    first_url = urls[0]
                    print(f"\n[youtube] First video URL: {first_url}")

                    chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s --profile-directory="Profile 1"'
                    print("1")
                    webbrowser.open(first_url)

                    time.sleep(2)  # Wait for page to load
                    pyautogui.press('space')           # Start playback

                    self.speak(f"Now playing {query}")
                else:
                    print("[youtube] No URLs found")
                    self.speak("I couldn't find that song on YouTube.")
            except Exception as e:
                print(f"[Error playing music] {e}")
                self.speak(f"Opening YouTube Music with {query}")

    def read_file(self, command):
        print("Reading a PDF")
        file_path = filedialog.askopenfilename(
            title="Select a PDF file", 
            filetypes=[("PDF Files", "*.pdf")]
        )
        
        if not file_path:
            self.speak("No file selected")
            return
            
        # Open in Edge with Read Aloud
        try:
            # First open the PDF in Edge
            os.startfile(file_path, 'open')
            
            # Wait for Edge to open
            time.sleep(3)
            
            # Activate Read Aloud (Ctrl+Shift+U)
            pyautogui.hotkey('ctrl', 'shift', 'u')
            self.speak("Reading the PDF file")
            
        except Exception as e:
            self.speak("Sorry, I couldn't read that PDF file")
            print(f"Error: {e}")

    def open_website(self, command):
        # Load the CSV file properly as a DataFrame
        
        sites = pd.read_csv('data/sites.csv')  # file name as a string

        # Assuming CSV has columns 'name' and 'url'
        sites_dict = dict(zip(sites['name'].str.lower(), sites['url']))

        website = command.replace("open", "").strip().lower()  # normalize to lowercase

        if not website:
            self.speak("What website would you like to open?")
            website = self.listen().lower()

        if website:
            if website in sites_dict:
                webbrowser.open(sites_dict[website])
                print(f"Opening {website}")
                self.speak(f"Opening {website}")
            else:
                # If website looks like a URL or domain
                if "." in website:
                    url = website
                    if not website.startswith(('http://', 'https://')):
                        url = f"https://{website}"
                    webbrowser.open(url)
                    self.speak(f"Opening {website}")
                else:
                    self.speak(f"I don't know how to open {website}")


    def instant_meet(self, command):
        # Generate a random meeting code (similar to Google's format)
        try:
            # Google Meet's new meeting URL
            meet_url = "https://meet.google.com/new"
            
            # Open in browser
            webbrowser.open(meet_url)
            print("Google Meet Created")
            self.speak("Creating an instant google ieet in your browser...")
        except Exception as e:
            self.speak(f"Error opening meeting: {str(e)}")

    def add_event(self):
        """Add event to local calendar storage"""
        try:
            # Get event details
            self.speak("Let me add an event to your calendar")
            print("Adding Event")
            
            # Get title
            title = ""
            while len(title) < 3:
                self.speak("What should I name the event?")
                title = self.take_command()
                if not title:
                    self.speak("Please say the event name again")

            # Get time
            time_str = ""
            while len(time_str) < 5:
                self.speak("When is the event? Example: Tomorrow 2 PM or December 25 3 PM")
                time_str = self.take_command()
                if not time_str:
                    self.speak("Please say the time again")

            # Get duration
            duration_str = ""
            while not any(x in duration_str for x in ['hour', 'minute', 'hr', 'min']):
                self.speak("How long will it last? Example: 1 hour or 30 minutes")
                duration_str = self.take_command()
                if not duration_str:
                    self.speak("Please say the duration again")

            # Parse datetime
            from dateparser import parse
            import re
            from datetime import timedelta
            
            event_time = parse(time_str)
            if not event_time:
                raise ValueError("Couldn't understand time")
            
            # Parse duration
            numbers = re.findall(r'\d+', duration_str)
            mins = int(numbers[0]) if numbers else 60
            if 'hour' in duration_str or 'hr' in duration_str:
                mins *= 60
            end_time = event_time + timedelta(minutes=mins)
            
            # Add to local calendar
            with open(self.calendar_file, 'r+') as f:
                calendar = json.load(f)
                calendar["events"].append({
                    "title": title,
                    "start": event_time.isoformat(),
                    "end": end_time.isoformat(),
                    "created": datetime.now().isoformat()
                })
                f.seek(0)
                json.dump(calendar, f, indent=2)
            
            self.speak(f"Added event: {title} at {event_time.strftime('%A, %B %d at %I:%M %p')}")
            print(f"Event saved to {self.calendar_file}")

        except Exception as e:
            print(f"Error: {e}")
            self.speak("Sorry, I couldn't add that event")

    def view_events(self):
        """View upcoming events"""
        try:
            with open(self.calendar_file, 'r') as f:
                calendar = json.load(f)
            
            now = datetime.now()
            upcoming = [
                e for e in calendar["events"]
                if datetime.fromisoformat(e["start"]) > now
            ]
            
            if not upcoming:
                self.speak("You have no upcoming events")
                return
            
            self.speak(f"You have {len(upcoming)} upcoming events")
            for event in sorted(upcoming, key=lambda x: x["start"]):
                start = datetime.fromisoformat(event["start"])
                self.speak(f"{event['title']} on {start.strftime('%A %B %d at %I:%M %p')}")
                
        except Exception as e:
            print(f"Error viewing events: {e}")
            self.speak("Sorry, I couldn't check your events")


    def analysis_csv(self):
        print("reading CSV")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_analysis_dir = os.path.join(current_dir, "csv_analysis")

        if not os.path.exists(csv_analysis_dir):
            raise FileNotFoundError(f"The directory '{csv_analysis_dir}' does not exist")
        
        # Check if app.py exists in the directory
        app_path = os.path.join(csv_analysis_dir, "app.py")
        if not os.path.exists(app_path):
            raise FileNotFoundError(f"The file 'app.py' does not exist in '{csv_analysis_dir}'")
        
        # Check for venv and activate it
        venv_path = os.path.join(csv_analysis_dir, "venv")
        activate_script = ""
        
        if os.path.exists(venv_path):
            if sys.platform == "win32":
                activate_script = os.path.join(venv_path, "Scripts", "activate.bat")
            else:
                activate_script = os.path.join(venv_path, "bin", "activate")
        
        try:
            # Change to the csv_analysis directory
            os.chdir(csv_analysis_dir)
            
            # Run the streamlit command in a new window
            if activate_script and os.path.exists(activate_script):
                if sys.platform == "win32":
                    subprocess.Popen(f'start cmd /K "{activate_script} && streamlit run app.py"', shell=True)
                    self.speak("Analysing CSV, please attach your CSV file")
                else:
                    subprocess.Popen(f"gnome-terminal -- bash -c 'source {activate_script} && streamlit run app.py; exec bash'", 
                                shell=True)
                    self.speak("Analysing CSV, please attach your CSV file")
            else:
                if sys.platform == "win32":
                    subprocess.Popen('start cmd /K "streamlit run app.py"', shell=True)
                else:
                    subprocess.Popen("gnome-terminal -- bash -c 'streamlit run app.py; exec bash'", 
                                shell=True)
                self.speak("Analysing CSV, please attach your CSV file")
                
        except Exception as e:
            print(f"Error running streamlit app: {e}")
            self.speak("Sorry, I encountered an error. Please try again")


    def analysis_resume(self):
        print("reading resume")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        resume_analysis_dir = os.path.join(current_dir, "resume_analysis")
        
        # Check if the directory exists
        if not os.path.exists(resume_analysis_dir):
            raise FileNotFoundError(f"The directory '{resume_analysis_dir}' does not exist")
        
        # Check if app.py exists in the directory
        app_path = os.path.join(resume_analysis_dir, "app.py")
        if not os.path.exists(app_path):
            raise FileNotFoundError(f"The file 'app.py' does not exist in '{resume_analysis_dir}'")
        
        # Check for venv and activate it
        venv_path = os.path.join(resume_analysis_dir, "venv")
        activate_script = ""
        
        if os.path.exists(venv_path):
            if sys.platform == "win32":
                activate_script = os.path.join(venv_path, "Scripts", "activate.bat")
            else:
                activate_script = os.path.join(venv_path, "bin", "activate")
        
        try:
            # Change to the resume_analysis directory
            os.chdir(resume_analysis_dir)
            
            # Run the streamlit command in a new window
            if activate_script and os.path.exists(activate_script):
                if sys.platform == "win32":
                    subprocess.Popen(f'start cmd /K "{activate_script} && streamlit run app.py"', shell=True)
                    self.speak("Analysing Resume, please attach your Resume file")
                else:
                    subprocess.Popen(f"gnome-terminal -- bash -c 'source {activate_script} && streamlit run app.py; exec bash'", 
                                shell=True)
                    self.speak("Analysing Resume, please attach your Resume file")
            else:
                if sys.platform == "win32":
                    subprocess.Popen('start cmd /K "streamlit run app.py"', shell=True)
                else:
                    subprocess.Popen("gnome-terminal -- bash -c 'streamlit run app.py; exec bash'", 
                                shell=True)
                self.speak("Analysing Resume, please attach your Resume file")
                
        except Exception as e:
            print(f"Error running streamlit app: {e}")
            self.speak("Sorry, I encountered an error. Please try again")
            raise
                

    def crack_a_joke(self):
        """Read a random joke from jokes.csv and speak it aloud"""
        try:
            # Path to jokes CSV (assuming it's in same directory)
            jokes_file = Path(__file__).parent / "data/jokes.csv"
            
            # Check if file exists
            if not jokes_file.exists():
                self.speak("Sorry, I couldn't find my joke book!")
                print(f"Error: Jokes file not found at {jokes_file}")
                return

            # Read all jokes from CSV
            with open(jokes_file, 'r', encoding='utf-8') as file:
                jokes = list(csv.reader(file))
            
            # Check if file has content
            if not jokes:
                self.speak("My joke book is empty!")
                return

            # Select and speak random joke
            random_joke = random.choice(jokes)[0]  # [0] gets first column if CSV has multiple
            self.speak(random_joke)
            print(f"Told joke: {random_joke}")

        except Exception as e:
            print(f"Error telling joke: {e}")
            self.speak("Sorry, I forgot the punchline!")

            

    def take_command(self, timeout=5):
        """Listen for voice command and return as text"""
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("Listening...")
                r.pause_threshold = 1
                audio = r.listen(source, timeout=timeout)
            
            try:
                print("Recognizing...")
                query = r.recognize_google(audio, language='en-in')
                print(f"User said: {query}")
                return query.lower()
            except Exception as e:
                print(f"Recognition error: {e}")
                return ""
        except Exception as e:
            print(f"Listening error: {e}")
            return ""

    def _init_calendar_file(self):
        """Initialize empty calendar file if it doesn't exist"""
        if not os.path.exists(self.calendar_file):
            with open(self.calendar_file, 'w') as f:
                json.dump({"events": []}, f)

    def on_press(self):
        if not self.recording:
            self.recording = True
            self.update_button_state()
            Thread(target=self.execute_command, daemon=True).start()
            
    def execute_command(self):
        command = self.listen()
        if command:
            self.handle_command(command)
        self.recording = False
        self.root.after(0, self.update_button_state)

    def update_button_state(self):
        if self.recording:
            # Recording state - red glow effect
            self.canvas.itemconfig(self.glow_circle, fill='#ff4444', outline='#ff6666')
            self.canvas.itemconfig(self.main_circle, fill='#ff2222', outline='#ff4444')
            self.pulse_animation()
        else:
            # Normal state - blue gradient effect
            self.canvas.itemconfig(self.glow_circle, fill='#6a8caf', outline='#4a6baf')
            self.canvas.itemconfig(self.main_circle, fill="#ffffff", outline='#2a4b6f')
            if hasattr(self, 'pulse_job'):
                self.root.after_cancel(self.pulse_job)

    def pulse_animation(self, size=120, step=0):
        if not self.recording:
            return
            
        # Smooth pulsing animation
        pulse_offset = 8 * abs((step % 20) - 10) / 10
        new_size = size + pulse_offset
        center = 200
        
        # Update glow circle
        self.canvas.coords(self.glow_circle, 
                          center - new_size/2, center - new_size/2, 
                          center + new_size/2, center + new_size/2)
        
        self.pulse_job = self.root.after(50, self.pulse_animation, size, step + 1)

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update()

    def setup_ui(self):
        self.root.title("Voice Assistant")
        self.root.geometry("500x700")
        self.root.resizable(False, False)
        
        # Set background image
        try:
            bg_image = Image.open("assets/images/bg.png")
            bg_image = bg_image.resize((500, 700), Image.Resampling.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(bg_image)
            
            # Create background label
            bg_label = tk.Label(self.root, image=self.bg_photo)
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception as e:
            print(f"Could not load background image: {e}")
            self.root.configure(bg='#1a1a1a')

        # Create main frame with dark semi-transparent background
        main_frame = tk.Frame(self.root, bg='#1a1a1a', bd=0)
        main_frame.place(relx=0.5, rely=0.5, anchor='center')

        # Title label with modern styling
        title_label = tk.Label(
            main_frame,
            text="Voice Assistant",
            font=('Segoe UI', 24, 'bold'),
            fg='white',
            bg='#1a1a1a',
            pady=20
        )
        title_label.pack()

        # Status label with enhanced styling
        self.status_var = tk.StringVar(value="Ready to listen...")
        self.status_label = tk.Label(
            main_frame,
            textvariable=self.status_var,
            font=('Segoe UI', 14),
            fg='#e0e0e0',
            bg='#1a1a1a',
            pady=10
        )
        self.status_label.pack()

        # Create canvas for circular button with enhanced design
        self.canvas = tk.Canvas(main_frame, width=400, height=400, 
                               highlightthickness=0, bd=0, bg='#1a1a1a')
        self.canvas.pack(pady=30)

        # Create layered circles for depth effect
        center = 200
        
        # Outer glow circle
        self.glow_circle = self.canvas.create_oval(
            center - 60, center - 60, center + 60, center + 60,
            fill='#6a8caf', outline='#4a6baf', width=2
        )
        
        # Main button circle
        self.main_circle = self.canvas.create_oval(
            center - 50, center - 50, center + 50, center + 50,
            fill='#4a6baf', outline='#2a4b6f', width=3
        )

        # Load and add button icon
        try:
            button_img = Image.open("assets/images/button_icon.png")
            button_img = button_img.resize((60, 60), Image.Resampling.LANCZOS)
            self.button_icon = ImageTk.PhotoImage(button_img)
            self.icon_item = self.canvas.create_image(center, center, image=self.button_icon)
        except Exception as e:
            print(f"Could not load button icon: {e}")
            # Fallback to microphone emoji
            self.icon_item = self.canvas.create_text(
                center, center, text="🎤", 
                font=("Segoe UI Emoji", 32), fill='white'
            )

        # Enhanced interaction effects
        self.canvas.bind("<Button-1>", lambda e: self.on_press())
        self.canvas.bind("<Enter>", self.on_hover_enter)
        self.canvas.bind("<Leave>", self.on_hover_leave)
        self.canvas.config(cursor="hand2")

        # Instructions label
        instructions = tk.Label(
            main_frame,
            text="Click the button and speak your command",
            font=('Segoe UI', 11),
            fg='#b0b0b0',
            bg='#1a1a1a',
            pady=20
        )
        instructions.pack()

    def on_hover_enter(self, event):
        if not self.recording:
            # Hover effect - slightly brighter
            self.canvas.itemconfig(self.glow_circle, fill="#D3D3D3", outline='#5a7bbf')
            self.canvas.itemconfig(self.main_circle, fill="#ffffff", outline='#3a5b8f')

    def on_hover_leave(self, event):
        if not self.recording:
            # Return to normal state
            self.canvas.itemconfig(self.glow_circle, fill='#D3D3D3', outline='#4a6baf')
            self.canvas.itemconfig(self.main_circle, fill='#ffffff', outline='#2a4b6f')


if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceAssistant(root)
    root.mainloop()