import pygame, random, time
from pygame.locals import *
import cv2
import numpy as np
from threading import Thread
from queue import Queue

#VARIABLES
SCREEN_WIDHT = 400
SCREEN_HEIGHT = 600
SPEED = 20
GRAVITY = 2.5
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


# Thumb detection - using simple hand contour analysis
def detect_thumb_gesture(frame):
    """
    Detect thumb position: UP (like) or DOWN (fist)
    Returns 'like' or 'fist'
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Detect skin color
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    mask = cv2.bitwise_or(mask, mask2)
    
    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find hand contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 'fist'
    
    hand_contour = max(contours, key=cv2.contourArea)
    hand_area = cv2.contourArea(hand_contour)
    
    # Need minimum hand size
    if hand_area < 2000:
        return 'fist'
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(hand_contour)
    
    # Get extreme points
    top_point = tuple(hand_contour[hand_contour[:, :, 1].argmin()][0])
    bottom_point = tuple(hand_contour[hand_contour[:, :, 1].argmax()][0])
    left_point = tuple(hand_contour[hand_contour[:, :, 0].argmin()][0])
    right_point = tuple(hand_contour[hand_contour[:, :, 0].argmax()][0])
    
    # Calculate aspect ratio (height / width)
    aspect_ratio = float(h) / (w + 1)
    
    # Count peaks (fingers extended)
    # Use convex hull to count fingers
    hull = cv2.convexHull(hand_contour, returnPoints=False)
    
    # Simple logic:
    # - If hand has tall aspect ratio (> 1.2) = LIKE (thumb up)
    # - If hand has low aspect ratio (< 1.2) = FIST (thumb down)
    if aspect_ratio > 0.8:
        return 'like'
    else:
        return 'fist'

hand_gesture_queue = Queue()
camera_running = True

def camera_thread_func():
    """Thread function to capture and process camera frames"""
    global camera_running
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Give camera time to warm up
    for _ in range(10):
        cap.read()
    
    prev_gesture = 'fist'
    gesture_stable = 'fist'
    stable_count = 0
    
    while camera_running:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        
        # Get thumb gesture
        gesture = detect_thumb_gesture(frame)
        
        # Stabilize gesture detection (require 2 consecutive frames)
        if gesture == gesture_stable:
            stable_count += 1
            if stable_count > 2:
                current_gesture = gesture
            else:
                current_gesture = gesture_stable
        else:
            gesture_stable = gesture
            stable_count = 0
            current_gesture = gesture_stable
        
        # Send signal when gesture changes from FIST to LIKE
        if current_gesture == 'like' and prev_gesture == 'fist':
            hand_gesture_queue.put("jump")
            print("🎯 JUMP SIGNAL SENT!")
        
        prev_gesture = current_gesture
        
        # Draw on frame - show detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        mask = cv2.bitwise_or(mask, mask2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            hand_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(hand_contour)
            aspect_ratio = float(h) / (w + 1)
            
            # Draw rectangle and info
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Ratio: {aspect_ratio:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display main status
        status = f"Thumb: {current_gesture.upper()}"
        color = (0, 255, 0) if current_gesture == 'like' else (0, 165, 255)
        cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, "LIKE (Tall) = Jump | FIST (Round) = Skip", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Flappy Bird - Hand Gesture', frame)
        
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
        
        # Check for hand gesture
        try:
            while not hand_gesture_queue.empty():
                gesture = hand_gesture_queue.get_nowait()
                if gesture == "jump" and begin:
                    bird = bird_group.sprites()[0]
                    bird.bump()
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
        
        # Check for hand gesture
        try:
            while not hand_gesture_queue.empty():
                gesture = hand_gesture_queue.get_nowait()
                if gesture == "jump":
                    bird = bird_group.sprites()[0]
                    bird.bump()
                    pygame.mixer.music.load(wing)
                    pygame.mixer.music.play()
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

