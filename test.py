import numpy as np
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class DetectionModule:
    def __init__(self):
        self.identification_logged = False  # Flag to indicate logging identification
        self.frame_counter = 0  # Counter for frames

    def process_frame(self, frame):
        self.frame_counter += 1
        try:
            # Convert image format from RGB888i to BGR888p (planar)
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = frame.reshape((frame.shape[0] // 3, 3, -1)).astype(np.uint8)  # Planar format

            # Further processing can be added here
            logging.info(f'Processing frame: {self.frame_counter}')  # Log current processing frame

        except Exception as e:
            logging.error(f'Error processing frame: {e}')  # Better error handling

    def stop(self):
        # Add stop method logic
        logging.info('DetectionModule stopped.')

    def run(self, input_source):
        while True:
            ret, frame = input_source.read()
            if not ret:
                break
            self.process_frame(frame)
            # Check detection warnings (e.g. if a condition is met)
            if self.frame_counter % 10 == 0:
                logging.warning('Detection warning: Check the last processed frame')  # Detection warnings

# Usage example
if __name__ == '__main__':
    input_source = cv2.VideoCapture(0)  # Replace with desired video source
    detection_module = DetectionModule()
    detection_module.run(input_source)