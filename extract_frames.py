import argparse
import os

import cv2


FRAME_SKIP = 15
FRAMES_DIR = "frames"


def extract_frames(video_path, frame_skip=FRAME_SKIP):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(FRAMES_DIR, video_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open video: {video_path}")
        return

    frame_index = 0
    saved_count = 0

    print(f"Extracting frames from: {video_path}")
    print(f"Saving every {frame_skip} frame(s) to: {output_dir}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_skip == 0:
            output_path = os.path.join(output_dir, f"frame_{frame_index:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1

            if saved_count % 10 == 0:
                print(f"Saved {saved_count} frames...")

        frame_index += 1

    cap.release()

    print("Done.")
    print(f"Processed {frame_index} frames.")
    print(f"Saved {saved_count} frames to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract skipped frames from a video.")
    parser.add_argument("video_path", help="Path to an .mp4 or .mov video file.")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=FRAME_SKIP,
        help=f"Save one frame every N frames. Default: {FRAME_SKIP}",
    )
    args = parser.parse_args()

    if args.frame_skip <= 0:
        print("Error: --frame-skip must be greater than 0.")
        return

    extract_frames(args.video_path, args.frame_skip)


if __name__ == "__main__":
    main()
