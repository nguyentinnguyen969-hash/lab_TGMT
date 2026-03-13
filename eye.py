import pygame, random, time
from pygame.locals import *
import cv2
import numpy as np
from threading import Thread
from queue import Queue

#VARIABLES
SCREEN_WIDHT = 400
SCREEN_HEIGHT = 600
LIFT_SPEED = -8  # Constant speed up when eyes are open
GRAVITY = 1.5    # Reduced gravity for smoother falling
SPEED = 18       # Normal jump speed for keyboard backup
GAME_SPEED = 4

GROUND_WIDHT = 2 * SCREEN_WIDHT
GROUND_HEIGHT= 100

PIPE_WIDHT = 80
PIPE_HEIGHT = 500

PIPE_GAP = 150

wing = 'assets/audio/wing.wav'
hit = 'assets/audio/hit.wav'

pygame.mixer.init()


class Bird(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)

        self.images =  [pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
                        pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
                        pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha()]

        self.speed = SPEED

        self.current_image = 0
        self.image = pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = SCREEN_WIDHT / 6
        self.rect[1] = SCREEN_HEIGHT / 2

    def update(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.speed += GRAVITY

        #UPDATE HEIGHT
        self.rect[1] += self.speed

    def bump(self):
        self.speed = -SPEED

    def begin(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]




class Pipe(pygame.sprite.Sprite):

    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)

        self. image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDHT, PIPE_HEIGHT))


        self.rect = self.image.get_rect()
        self.rect[0] = xpos

        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = - (self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize


        self.mask = pygame.mask.from_surface(self.image)


    def update(self):
        self.rect[0] -= GAME_SPEED

        

class Ground(pygame.sprite.Sprite):
    
    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDHT, GROUND_HEIGHT))

        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT
    def update(self):
        self.rect[0] -= GAME_SPEED

def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])

def get_random_pipes(xpos):
    size = random.randint(100, 300)
    pipe = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted


# OpenCV Haar Cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
alt_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

def detect_eye_status(frame):
    """
    Detect if eyes are OPEN or CLOSED using Haar Cascade with improved parameters
    Returns 'open' or 'closed', eye_rectangles, face_rect
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Improve contrast and brightness
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Detect faces with optimized parameters
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.05,      # Smaller scale for finer detection
        minNeighbors=5,        # Stricter neighbors requirement
        minSize=(80, 80),      # Minimum face size
        maxSize=(400, 400)     # Maximum face size
    )
    
    if len(faces) == 0:
        return 'unknown', [], None
    
    # Get largest face
    face = max(faces, key=lambda f: f[2] * f[3])
    x, y, fw, fh = face
    
    # Add padding to face region for better eye detection
    pad = int(fw * 0.1)
    x_min = max(0, x - pad)
    y_min = max(0, y - pad)
    x_max = min(frame.shape[1], x + fw + pad)
    y_max = min(frame.shape[0], y + fh + pad)
    
    roi_gray = gray[y_min:y_max, x_min:x_max]
    
    # Split face ROI into left and right halves for more accurate eye detection
    roi_h, roi_w = roi_gray.shape
    mid_x = roi_w // 2
    
    left_roi = roi_gray[:, :mid_x]
    right_roi = roi_gray[:, mid_x:]
    
    def get_best_eye(roi, offset_x):
        eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.05, minNeighbors=4, minSize=(15, 15), maxSize=(80, 80))
        if len(eyes) == 0:
            eyes = alt_eye_cascade.detectMultiScale(roi, scaleFactor=1.05, minNeighbors=4, minSize=(15, 15), maxSize=(80, 80))
        
        if len(eyes) > 0:
            # Pick the largest/best eye in this half
            ex, ey, ew, eh = max(eyes, key=lambda e: e[2] * e[3])
            return (x_min + offset_x + ex, y_min + ey, ew, eh)
        return None

    left_eye = get_best_eye(left_roi, 0)
    right_eye = get_best_eye(right_roi, mid_x)
    
    eye_rects = []
    if left_eye: eye_rects.append(left_eye)
    if right_eye: eye_rects.append(right_eye)
    
    # Status: Open only if at least one clear eye is found
    # We now limit to exactly 1 per side (max 2).
    if len(eye_rects) >= 1:
        return 'open', eye_rects, (x, y, fw, fh)
    else:
        return 'closed', eye_rects, (x, y, fw, fh)

# Store previous eye status
prev_eye_info = {'status': 'closed'}

hand_gesture_queue = Queue()
camera_running = True

def camera_thread_func():
    """Thread function to capture and process camera frames"""
    global camera_running, prev_eye_info
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Give camera time to warm up
    for _ in range(10):
        cap.read()
    
    prev_eye_status = 'closed'
    last_stable_status = 'closed'
    stable_counter = 0
    
    while camera_running:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Detect eye status
        eye_status, eye_rects, face_rect = detect_eye_status(frame)
        
        if eye_status != 'unknown':
            if eye_status == prev_eye_status:
                stable_counter += 1
            else:
                stable_counter = 1
            
            # Update stable status after 2 consistent frames
            if stable_counter >= 2:
                last_stable_status = eye_status
            
            # Send CURRENT state for continuous control
            # Clear queue and put latest state
            while not hand_gesture_queue.empty():
                try: hand_gesture_queue.get_nowait()
                except: break
            hand_gesture_queue.put(last_stable_status)
            
            prev_eye_status = eye_status
        
        # Draw face detection
        if face_rect:
            x, y, fw, fh = face_rect
            cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 255), 2)
            cv2.putText(frame, "FACE", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw detected eyes
        for idx, (ex, ey, ew, eh) in enumerate(eye_rects):
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.circle(frame, (ex+ew//2, ey+eh//2), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"E{idx+1}", (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Display main status
        status = f"Eyes: {prev_eye_status.upper()}"
        color = (0, 255, 0) if prev_eye_status == 'open' else (0, 0, 255) if prev_eye_status == 'closed' else (127, 127, 127)
        cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 4)
        
        eye_count = len(eye_rects)
        cv2.putText(frame, f"Eyes: {eye_count}/2", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"Stable: {min(stable_counter, 5)}/2", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
        
        cv2.putText(frame, "CLOSE eyes = RISE | OPEN eyes = FALL", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, "Yellow=Face | Green=Eye | Closed cam to exit", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        cv2.imshow('Flappy Bird - Eye Tracking 👁️ (Optimized)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird')

BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))
BEGIN_IMAGE = pygame.image.load('assets/sprites/message.png').convert_alpha()

clock = pygame.time.Clock()

def initialize_game():
    """Initialize or reset the game"""
    bird_group = pygame.sprite.Group()
    bird = Bird()
    bird_group.add(bird)

    ground_group = pygame.sprite.Group()
    for i in range(2):
        ground = Ground(GROUND_WIDHT * i)
        ground_group.add(ground)

    pipe_group = pygame.sprite.Group()
    for i in range(2):
        pipes = get_random_pipes(SCREEN_WIDHT * i + 800)
        pipe_group.add(pipes[0])
        pipe_group.add(pipes[1])
    
    return bird_group, ground_group, pipe_group

# Main game loop
game_running = True
camera_thread = None

try:
    # Start camera thread
    camera_running = True
    camera_thread = Thread(target=camera_thread_func, daemon=True)
    camera_thread.start()
    print("Camera started. Close the camera window to stop.")
    time.sleep(1)  # Give camera time to initialize
except Exception as e:
    print(f"Camera initialization failed: {e}")
    camera_running = False

while game_running:
    bird_group, ground_group, pipe_group = initialize_game()
    
    # Begin screen
    begin = True
    while begin:
        clock.tick(15)

        for event in pygame.event.get():
            if event.type == QUIT:
                game_running = False
                camera_running = False
                begin = False
            if event.type == KEYDOWN:
                if event.key == K_SPACE or event.key == K_UP:
                    bird = bird_group.sprites()[0]
                    bird.bump()
                    pygame.mixer.music.load(wing)
                    pygame.mixer.music.play()
                    begin = False
        
        # Continuous eye control
        try:
            status = 'closed'
            while not hand_gesture_queue.empty():
                status = hand_gesture_queue.get_nowait()
            
            if status == 'closed':
                bird = bird_group.sprites()[0]
                bird.speed = LIFT_SPEED
                # Optional: reduce wing sound frequency for smooth flight
                if random.random() < 0.2:
                    pygame.mixer.music.load(wing)
                    pygame.mixer.music.play()
                begin = False
        except:
            pass

        screen.blit(BACKGROUND, (0, 0))
        screen.blit(BEGIN_IMAGE, (120, 150))

        if is_off_screen(ground_group.sprites()[0]):
            ground_group.remove(ground_group.sprites()[0])
            new_ground = Ground(GROUND_WIDHT - 20)
            ground_group.add(new_ground)

        bird = bird_group.sprites()[0]
        bird.begin()
        ground_group.update()

        bird_group.draw(screen)
        ground_group.draw(screen)

        pygame.display.update()

    # Game loop
    if not game_running:
        break
    
    while game_running:
        clock.tick(15)

        for event in pygame.event.get():
            if event.type == QUIT:
                game_running = False
                camera_running = False
            if event.type == KEYDOWN:
                if event.key == K_SPACE or event.key == K_UP:
                    bird = bird_group.sprites()[0]
                    bird.bump()
                    pygame.mixer.music.load(wing)
                    pygame.mixer.music.play()
        
        # Continuous eye control
        try:
            status = 'closed'
            while not hand_gesture_queue.empty():
                status = hand_gesture_queue.get_nowait()
            
            if status == 'closed':
                bird = bird_group.sprites()[0]
                bird.speed = LIFT_SPEED
        except:
            pass

        screen.blit(BACKGROUND, (0, 0))

        if is_off_screen(ground_group.sprites()[0]):
            ground_group.remove(ground_group.sprites()[0])
            new_ground = Ground(GROUND_WIDHT - 20)
            ground_group.add(new_ground)

        if is_off_screen(pipe_group.sprites()[0]):
            pipe_group.remove(pipe_group.sprites()[0])
            pipe_group.remove(pipe_group.sprites()[0])
            pipes = get_random_pipes(SCREEN_WIDHT * 2)
            pipe_group.add(pipes[0])
            pipe_group.add(pipes[1])

        bird_group.update()
        ground_group.update()
        pipe_group.update()

        bird_group.draw(screen)
        pipe_group.draw(screen)
        ground_group.draw(screen)

        pygame.display.update()

        if (pygame.sprite.groupcollide(bird_group, ground_group, False, False, pygame.sprite.collide_mask) or
                pygame.sprite.groupcollide(bird_group, pipe_group, False, False, pygame.sprite.collide_mask)):
            pygame.mixer.music.load(hit)
            pygame.mixer.music.play()
            time.sleep(1)
            break

# Cleanup
camera_running = False
pygame.quit()
print("Game ended!")

