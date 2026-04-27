"""
Flappy Bird controlled by eye state (blink vs. open) via MediaPipe Face Mesh.
Eyes open: lift. Blinking/closed: extra downward acceleration.
Press R to reset. Camera on the right with face + eye status.
"""

import cv2 as cv
import numpy as np
import mediapipe as mp
import pygame
import time
from pathlib import Path
from urllib.request import urlretrieve
from collections import Counter, deque
import torch
from torchvision import transforms
import PIL.Image
from torchvision.transforms import InterpolationMode
from model import create_binary_emotion_model

# --- Game constants ---
GAME_WIDTH = 1040
GAME_HEIGHT = 1520
BIRD_SIZE = 40
GRAVITY = 0.6
PIPE_WIDTH = 60
PIPE_GAP = 300
PIPE_SPEED = 4
PIPE_SPAWN_INTERVAL = 125
BIRD_X = 80
GROUND_HEIGHT = 50
SKY_COLOR = (113, 197, 207)
PIPE_COLOR = (34, 139, 34)
GROUND_COLOR = (222, 184, 135)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CAM_WIDTH, CAM_HEIGHT = 1600, 1200

# Binary CNN classes:
# 0=anger, 1=blink
BLINK_CLASS = 1
CLASS_NAMES = ["anger", "blink"]
VOTE_WINDOW = 3
# Inference bias tuning (logit offsets) for binary model [anger, blink].
CLASS_LOGIT_BIAS = [0.0, 0.0]

# Physics: instant direction switch (no inertia buildup)
RISE_SPEED = -9.0     # px/frame while eyes open
FALL_SPEED = 9.0      # px/frame while blinking/closed

# Audio height cue
SAMPLE_RATE = 22050
TONE_MS = 45
TONE_VOL = 0.16
PITCH_MIN_HZ = 220
PITCH_MAX_HZ = 1100
PITCH_UPDATE_MS = 55

FACE_LANDMARKER_MODEL = "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
EMOTION_MODEL_PATH = "emotion_set_binary.pth"


def _ensure_face_landmarker_model(model_path: Path) -> None:
    if model_path.exists():
        return
    print(f"Downloading {model_path.name} for MediaPipe FaceLandmarker...")
    urlretrieve(FACE_LANDMARKER_URL, model_path)
    print(f"Saved model to {model_path}")


def _make_tone(freq_hz: float, duration_ms: int = TONE_MS) -> pygame.mixer.Sound:
    n_samples = max(8, int(SAMPLE_RATE * duration_ms / 1000))
    t = np.arange(n_samples, dtype=np.float32) / SAMPLE_RATE
    wave = np.sin(2.0 * np.pi * freq_hz * t) * TONE_VOL
    pcm = np.int16(np.clip(wave * 32767.0, -32768, 32767))
    stereo = np.column_stack((pcm, pcm))
    return pygame.sndarray.make_sound(stereo)


def main():
    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cam.isOpened():
        print("Camera not opening")
        return

    model_path = Path(FACE_LANDMARKER_MODEL)
    _ensure_face_landmarker_model(model_path)

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    face_landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
    )
    device = torch.device("cpu")
    emotion_model = create_binary_emotion_model().to(device)
    emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=device))
    emotion_model.eval()
    emotion_transform = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    class_bias = torch.tensor(CLASS_LOGIT_BIAS, dtype=torch.float32, device=device).unsqueeze(0)

    pygame.mixer.pre_init(SAMPLE_RATE, size=-16, channels=2, buffer=256)
    pygame.init()
    screen = pygame.display.set_mode((GAME_WIDTH + CAM_WIDTH, max(GAME_HEIGHT, CAM_HEIGHT)))
    pygame.display.set_caption("Eyes open = rise · Blink = fall")
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
    audio_ok = True
    try:
        cue_channel = pygame.mixer.Channel(0)
    except pygame.error:
        audio_ok = False
        cue_channel = None
    next_pitch_time_ms = 0

    eyes_open = True
    pred_label = "none"
    pred_conf = 0.0
    pred_history = deque(maxlen=VOTE_WINDOW)

    # Game state
    bird_y = GAME_HEIGHT // 2 - BIRD_SIZE // 2
    bird_vel = 0
    pipes = []  # list of {"x": int, "gap_y": int, "scored": bool}
    score = 0
    game_over = False
    frame_count = 0

    def reset_game():
        nonlocal bird_y, bird_vel, pipes, score, game_over, eyes_open, pred_label, pred_conf, pred_history
        bird_y = GAME_HEIGHT // 2 - BIRD_SIZE // 2
        bird_vel = 0
        pipes.clear()
        score = 0
        game_over = False
        eyes_open = True
        pred_label = "none"
        pred_conf = 0.0
        pred_history.clear()

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
    running = True

    while running:
        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                reset_game()

        # Camera + face mesh (sync every frame = low-latency eye state)
        ret, frame = cam.read()
        if not ret:
            continue
        frame = cv.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.monotonic_ns() // 1_000_000)
        res = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        if res.face_landmarks and len(res.face_landmarks) > 0:
            lm = res.face_landmarks[0]
            xs = [p.x for p in lm]
            ys = [p.y for p in lm]
            x0, y0 = int(min(xs) * w), int(min(ys) * h)
            x1, y1 = int(max(xs) * w), int(max(ys) * h)
            cv.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 1)
            pad_x = int(0.25 * max(1, x1 - x0))
            pad_y = int(0.25 * max(1, y1 - y0))
            cx0 = max(0, x0 - pad_x)
            cy0 = max(0, y0 - pad_y)
            cx1 = min(w, x1 + pad_x)
            cy1 = min(h, y1 + pad_y)
            crop = frame[cy0:cy1, cx0:cx1]
            if crop.size > 0:
                pil = PIL.Image.fromarray(cv.cvtColor(crop, cv.COLOR_BGR2RGB))
                model_in = emotion_transform(pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = emotion_model(model_in) + class_bias
                    probs = torch.softmax(logits, dim=1)
                    pred_idx = int(torch.argmax(probs, dim=1).item())
                    pred_conf = float(probs[0, pred_idx].item())
                pred_history.append(pred_idx)
                voted_idx = Counter(pred_history).most_common(1)[0][0]
                pred_label = CLASS_NAMES[voted_idx]
                eyes_open = voted_idx != BLINK_CLASS
        else:
            pred_history.append(BLINK_CLASS)
            voted_idx = Counter(pred_history).most_common(1)[0][0]
            pred_label = CLASS_NAMES[voted_idx]
            eyes_open = voted_idx != BLINK_CLASS
            pred_conf = 0.0

        status = "OPEN" if eyes_open else "BLINK"
        cv.putText(
            frame,
            f"model={pred_label} ({pred_conf:.2f})  {status}",
            (8, 28),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if eyes_open else (0, 120, 255),
            2,
        )

        if not game_over:
            bird_vel = RISE_SPEED if eyes_open else FALL_SPEED
            bird_y += int(bird_vel)
            bird_y = max(0, min(GAME_HEIGHT - GROUND_HEIGHT - BIRD_SIZE, bird_y))

            if audio_ok:
                now_ms = pygame.time.get_ticks()
                if now_ms >= next_pitch_time_ms:
                    # Higher bird -> higher pitch
                    play_area = GAME_HEIGHT - GROUND_HEIGHT - BIRD_SIZE
                    height_ratio = 1.0 - (bird_y / max(1, play_area))
                    freq = PITCH_MIN_HZ + (PITCH_MAX_HZ - PITCH_MIN_HZ) * height_ratio
                    tone = _make_tone(freq)
                    cue_channel.play(tone)
                    next_pitch_time_ms = now_ms + PITCH_UPDATE_MS

            pipe_spawn_counter += 1
            if pipe_spawn_counter >= PIPE_SPAWN_INTERVAL:
                add_pipe()
                pipe_spawn_counter = 0

            for p in pipes:
                p["x"] -= PIPE_SPEED

            pipes = [p for p in pipes if p["x"] + PIPE_WIDTH > 0]

            if len(pipes) == 0:
                add_pipe()

            for p in pipes:
                if not p["scored"] and p["x"] + PIPE_WIDTH < BIRD_X:
                    p["scored"] = True
                    score += 1

            br = bird_rect()
            for p in pipes:
                top_r, bot_r = pipe_rects(p)
                if br.colliderect(top_r) or br.colliderect(bot_r):
                    game_over = True
                    break
            if bird_y >= GAME_HEIGHT - GROUND_HEIGHT - BIRD_SIZE or bird_y <= 0:
                game_over = True

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

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        try:
            cam_surf = pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
        except AttributeError:
            cam_surf = pygame.image.fromstring(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
        cam_surf = pygame.transform.scale(cam_surf, (CAM_WIDTH, CAM_HEIGHT))
        screen.blit(cam_surf, (GAME_WIDTH, 0))
        label = small_font.render("Eyes open = rise   Blink = fall", True, WHITE)
        screen.blit(label, (GAME_WIDTH + 10, 10))

        pygame.display.flip()
        clock.tick(60)

    face_landmarker.close()
    if audio_ok:
        pygame.mixer.stop()
    cam.release()
    cv.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()
