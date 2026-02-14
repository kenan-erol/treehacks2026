import cv2
import torch
import time
import tempfile
import os
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Config ---
CAMERA_INDEX = 0          # 0 for default webcam, or /dev/video0
CLIP_DURATION = 3         # seconds per analysis chunk
FPS = 4                   # match Cosmos training fps
MODEL_NAME = "nvidia/Cosmos-Reason2-8B"  # or 8B if you have the memory

SECURITY_PROMPT = """Describe every person visible in this camera feed in detail. For each person include:

1. Estimated age range and gender
2. Clothing (colors, type, logos, accessories)
3. Physical features (hair color/style, height estimate, build)
4. What they are doing (walking, standing, carrying something, etc.)
5. Their location in the frame and direction of movement

If no person is visible, say "NO PERSON DETECTED."

Answer the question using the following format:
<think>
your reasoning
</think>

<answer>
your answer
</answer>."""

# --- Load model once ---
print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("Model loaded!")

def capture_clip(cap, duration=3, fps=4):
    """Capture a short video clip from webcam and save as temp mp4."""
    frames = []
    num_frames = duration * fps
    frame_interval = 1.0 / fps

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        time.sleep(frame_interval)

    if not frames:
        return None

    # Save as temp video file
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return tmp.name

def analyze_clip(video_path):
    """Send a video clip to Cosmos Reason 2 for analysis."""
    messages = [
        {"role": "system", "content": "You are a security monitoring AI agent."},
        {"role": "user", "content": [
            {"type": "video", "video": f"file://{video_path}", "fps": FPS},
            {"type": "text", "text": SECURITY_PROMPT}
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], videos=[video_path], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, temperature=0.6)

    response = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

# --- Main loop ---
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    print(f"Security agent running - analyzing {CLIP_DURATION}s clips at {FPS}fps")
    print("Press Ctrl+C to stop\n")

    clip_count = 0
    try:
        while True:
            clip_count += 1
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # Capture clip
            print(f"[{timestamp}] Capturing clip #{clip_count}...")
            video_path = capture_clip(cap, CLIP_DURATION, FPS)
            if video_path is None:
                print("Failed to capture clip, retrying...")
                continue

            # Analyze
            print(f"[{timestamp}] Analyzing...")
            try:
                result = analyze_clip(video_path)

                # # Check for alerts
                # if "ALERT" in result.upper():
                #     print(f"\n🚨 [{timestamp}] {result}\n")
                #     # TODO: trigger notification (email, webhook, etc.)
                # else:
                #     print(f"✅ [{timestamp}] {result.strip()[:100]}")
                # Check for person detection
                if "NO PERSON" in result.upper():
                    print(f"✅ [{timestamp}] No person detected")
                else:
                    print(f"\n👤 [{timestamp}] Person detected:\n{result}\n")

            except Exception as e:
                print(f"Analysis error: {e}")
            finally:
                os.unlink(video_path)  # clean up temp file

    except KeyboardInterrupt:
        print("\nStopping security agent...")
    finally:
        cap.release()

if __name__ == "__main__":
    main()
