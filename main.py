"""
Flappy Bird game controlled by smiling.
Uses blaze_face_short_range.tflite for face detection and emotion_set.pth (CNN) for expression.
Smile (class=4) makes the bird jump. Press R to reset. Face bounding boxes shown on camera feed.
"""

import cv2 as cv
import numpy as np
import time
import mediapipe as mp
import torch
from torchvision import transforms
import PIL
from train import CNN
import pygame

# --- Game constants ---
GAME_WIDTH = 400
GAME_HEIGHT = 600
BIRD_SIZE = 40
GRAVITY = 0.6
JUMP_VEL = -10
PIPE_WIDTH = 60
PIPE_GAP = 180
PIPE_SPEED = 3
PIPE_SPAWN_INTERVAL = 90
BIRD_X = 80
GROUND_HEIGHT = 50
SKY_COLOR = (113, 197, 207)
PIPE_COLOR = (34, 139, 34)
GROUND_COLOR = (222, 184, 135)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CAM_WIDTH, CAM_HEIGHT = 640, 480

# --- Globals ---
timer = 0
prev_timer = 0
initial_time = time.monotonic_ns()
latest_res = {"result": None}


def push_next_stamp(prev: int, initial: int) -> int:
    cur = (time.monotonic_ns() - initial) // 1_000_000
    if cur <= prev:
        cur = prev + 1
    return int(cur)


def face_callback(result, output_image, timestamp_ms):
    latest_res["result"] = result


def main():
    global timer, prev_timer, initial_time

    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cam.isOpened():
        print("Camera not opening")
        return

    model_path = "./blaze_face_short_range.tflite"
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=face_callback,
        min_detection_confidence=0.7,
    )

    pygame.init()
    screen = pygame.display.set_mode((GAME_WIDTH + CAM_WIDTH, max(GAME_HEIGHT, CAM_HEIGHT)))
    pygame.display.set_caption("Smile to Flap!")
    clock = pygame.time.Clock()

    # Load bird sprite (make black transparent)
    try:
        bird_surf = pygame.image.load("bird.png").convert_alpha()
        bird_surf = pygame.transform.scale(bird_surf, (BIRD_SIZE, BIRD_SIZE))
        # Colorkey black so we don't get a black rect
        bird_surf.set_colorkey(BLACK)
    except pygame.error:
        bird_surf = pygame.Surface((BIRD_SIZE, BIRD_SIZE))
        bird_surf.fill((255, 200, 0))
        bird_surf.set_colorkey(BLACK)

    font = pygame.font.Font(None, 48)
    small_font = pygame.font.Font(None, 28)

    # Emotion model
    device = torch.device("cpu")
    cnn = CNN()
    cnn.load_state_dict(torch.load("emotion_set2.pth", map_location=device))
    cnn.eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])

    # Smile smoothing: require majority smile over last few frames to trigger jump (reduces false jumps)
    emotion_history = []
    HISTORY_LEN = 5
    SMILE_CLASS = 4
    ANGER_CLASS = 0
    jump_cooldown_frames = 0
    COOLDOWN = 12  # frames between allowed jumps

    # Game state
    bird_y = GAME_HEIGHT // 2 - BIRD_SIZE // 2
    bird_vel = 0
    pipes = []  # list of {"x": int, "gap_y": int, "scored": bool}
    score = 0
    game_over = False
    frame_count = 0

    def reset_game():
        nonlocal bird_y, bird_vel, pipes, score, game_over
        bird_y = GAME_HEIGHT // 2 - BIRD_SIZE // 2
        bird_vel = 0
        pipes.clear()
        score = 0
        game_over = False

    def add_pipe():
        gap_y = np.random.randint(GROUND_HEIGHT + 80, GAME_HEIGHT - GROUND_HEIGHT - 80 - PIPE_GAP)
        pipes.append({"x": GAME_WIDTH, "gap_y": gap_y, "scored": False})

    def bird_rect():
        return pygame.Rect(BIRD_X, bird_y, BIRD_SIZE, BIRD_SIZE)

    def pipe_rects(p):
        top = pygame.Rect(p["x"], 0, PIPE_WIDTH, p["gap_y"])
        bottom = pygame.Rect(p["x"], p["gap_y"] + PIPE_GAP, PIPE_WIDTH, GAME_HEIGHT - (p["gap_y"] + PIPE_GAP))
        return top, bottom

    add_pipe()
    pipe_spawn_counter = 0

    with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
        running = True
        cropped_face = None

        while running:
            frame_count += 1
            initial_time = time.monotonic_ns() if frame_count == 1 else initial_time
            timer = push_next_stamp(prev_timer, initial_time)

            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    reset_game()

            # Camera + face detection
            ret, frame = cam.read()
            if not ret:
                continue
            frame = cv.flip(frame, 1)
            h, w, _ = frame.shape
            face_bboxes = []

            if timer != prev_timer:
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                detector.detect_async(mp_image, timer)

            if latest_res["result"]:
                for det in latest_res["result"].detections:
                    bbox = det.bounding_box
                    x, y, bw, bh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                    face_bboxes.append((x, y, bw, bh))
                    # Crop with padding for emotion model
                    x1 = max(0, x - bw // 2)
                    y1 = max(0, y - bh // 2)
                    x2 = min(w, x + bw + bw // 2)
                    y2 = min(h, y + bh + bh // 2)
                    cropped_face = frame[y1:y2, x1:x2]
                    if cropped_face.size > 0:
                        cropped_face = cv.resize(cropped_face, (256, 256))
            else:
                cropped_face = None

            # Draw face bounding boxes on camera frame
            for (x, y, bw, bh) in face_bboxes:
                cv.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

            # Emotion -> jump when smile (class 4)
            if cropped_face is not None and cropped_face.size > 0 and not game_over and jump_cooldown_frames <= 0:
                try:
                    pil_img = PIL.Image.fromarray(cv.cvtColor(cropped_face, cv.COLOR_BGR2RGB))
                    tensor_in = transform(pil_img).unsqueeze(0)
                    with torch.no_grad():
                        out = cnn(tensor_in)
                    pred = torch.argmax(out, dim=1).item()
                    emotion_history.append(pred)
                    if len(emotion_history) > HISTORY_LEN:
                        emotion_history.pop(0)
                    if len(emotion_history) == HISTORY_LEN and sum(1 for x in emotion_history if x == ANGER_CLASS) >= 3:
                        bird_vel = JUMP_VEL
                        jump_cooldown_frames = COOLDOWN
                except Exception:
                    pass

            if jump_cooldown_frames > 0:
                jump_cooldown_frames -= 1

            # Game update
            if not game_over:
                bird_vel += GRAVITY
                bird_y += int(bird_vel)
                bird_y = max(0, min(GAME_HEIGHT - GROUND_HEIGHT - BIRD_SIZE, bird_y))

                pipe_spawn_counter += 1
                if pipe_spawn_counter >= PIPE_SPAWN_INTERVAL:
                    add_pipe()
                    pipe_spawn_counter = 0

                for p in pipes:
                    p["x"] -= PIPE_SPEED

                pipes = [p for p in pipes if p["x"] + PIPE_WIDTH > 0]

                if len(pipes) == 0:
                    add_pipe()

                # Score when bird passes pipe
                for p in pipes:
                    if not p["scored"] and p["x"] + PIPE_WIDTH < BIRD_X:
                        p["scored"] = True
                        score += 1

                # Collision
                br = bird_rect()
                for p in pipes:
                    top_r, bot_r = pipe_rects(p)
                    if br.colliderect(top_r) or br.colliderect(bot_r):
                        game_over = True
                        break
                if bird_y >= GAME_HEIGHT - GROUND_HEIGHT - BIRD_SIZE or bird_y <= 0:
                    game_over = True

            # Draw game (left side)
            screen.fill(SKY_COLOR, (0, 0, GAME_WIDTH, GAME_HEIGHT))
            for p in pipes:
                top_r, bot_r = pipe_rects(p)
                pygame.draw.rect(screen, PIPE_COLOR, top_r)
                pygame.draw.rect(screen, (20, 100, 20), top_r, 2)
                pygame.draw.rect(screen, PIPE_COLOR, bot_r)
                pygame.draw.rect(screen, (20, 100, 20), bot_r, 2)
            pygame.draw.rect(screen, GROUND_COLOR, (0, GAME_HEIGHT - GROUND_HEIGHT, GAME_WIDTH, GROUND_HEIGHT))
            screen.blit(bird_surf, (BIRD_X, bird_y))

            score_text = font.render(str(score), True, BLACK)
            screen.blit(score_text, (GAME_WIDTH // 2 - score_text.get_width() // 2, 30))

            if game_over:
                over_text = font.render("Game Over", True, BLACK)
                screen.blit(over_text, (GAME_WIDTH // 2 - over_text.get_width() // 2, GAME_HEIGHT // 2 - 30))
                hint = small_font.render("Press R to reset", True, BLACK)
                screen.blit(hint, (GAME_WIDTH // 2 - hint.get_width() // 2, GAME_HEIGHT // 2 + 20))

            # Draw camera feed (right side) with face boxes already drawn on frame
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            try:
                cam_surf = pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
            except AttributeError:
                cam_surf = pygame.image.fromstring(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
            cam_surf = pygame.transform.scale(cam_surf, (CAM_WIDTH, CAM_HEIGHT))
            screen.blit(cam_surf, (GAME_WIDTH, 0))
            label = small_font.render("Smile to jump!", True, WHITE)
            screen.blit(label, (GAME_WIDTH + 10, 10))

            pygame.display.flip()
            prev_timer = timer
            clock.tick(60)

    cam.release()
    cv.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()
