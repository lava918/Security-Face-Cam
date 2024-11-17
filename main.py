import cv2
import datetime
import time
import pygame
import os
from threading import Thread

class SecurityCamera:
    def __init__(self, output_folder, alert_sound_path):
        self.cap = cv2.VideoCapture(0)
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Generate output filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_folder, f'security_footage_{timestamp}.mp4')
        
        # Initialize video writer with 4x speed (4 * fps)
        self.out = cv2.VideoWriter(output_path, 
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps * 4,
                                 (width, height))
        
        # Load face detection classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load full body detection classifier
        self.body_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_fullbody.xml'
        )
        
        # Initialize pygame for audio
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound(alert_sound_path)
        
        # Initialize variables
        self.last_message_time = time.time()
        self.running = True

    def play_alert(self):
        """Play alert sound in a separate thread"""
        self.alert_sound.play()

    def log_message(self, detection=False):
        """Log message with timestamp"""
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if detection:
            message = f"⚠️ ALERT! Motion detected at {current_time}"
        else:
            message = f"System active - No motion detected at {current_time}"
        print(message)

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert frame to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces and bodies
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            bodies = self.body_cascade.detectMultiScale(gray, 1.3, 5)

            current_time = time.time()
            
            if len(faces) > 0 or len(bodies) > 0:
                # Motion detected
                self.log_message(detection=True)
                Thread(target=self.play_alert).start()
                
                # Draw rectangles around detected objects
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                for (x, y, w, h) in bodies:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            elif current_time - self.last_message_time >= 6:
                # No motion detected, log message every 6 seconds
                self.log_message(detection=False)
                self.last_message_time = current_time

            # Write frame to video
            self.out.write(frame)

            # Display the frame
            cv2.imshow('Security Camera', frame)

            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

def main():
    # Set up paths
    output_folder = "security_footage"
    alert_sound_path = "alert.mp3"  # Make sure this file exists
    
    # Initialize and run security camera
    camera = SecurityCamera(output_folder, alert_sound_path)
    try:
        camera.run()
    finally:
        camera.cleanup()

if __name__ == "__main__":
    main()