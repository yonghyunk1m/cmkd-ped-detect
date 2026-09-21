"""
Teacher-input preprocessing: 3D-projection crop of the ASPED v.a video.

For every recorder we take the ground region within a fixed radius (the same
radius that defines the audio label) and turn it into a rectangular video crop
that the Mask2Former teacher then reads. The crop is built as follows:

  1. Per session and station, a hand-fit ground ellipse marks the radius buffer
     in the image plane (see ``ellipses/ellipses_va.py``). The buffer name maps
     to a real-world diameter (``BUFFER_DIAMETER_MAP``), e.g. the 6 m radius is a
     12 m diameter.
  2. The pixel-to-meter scale is the ellipse major axis over that diameter.
  3. A person height of ``PERSON_REAL_HEIGHT_METERS`` (2 m) is converted to
     pixels with that scale.
  4. The tightest rotated box around the ground ellipse gives four ground
     points; these are extruded upward by the person height to four "sky"
     points, so the eight points bound the volume a standing pedestrian can
     occupy over the buffer.
  5. The axis-aligned bounding rectangle of the eight points is the crop. The
     saved input is therefore a rectangle, not an elliptical mask.

The script writes one cropped ``.mp4`` per buffer and recorder
(``{buffer}_ellipse_{k}.mp4``). The KD teacher in this repo consumes the 6 m
crop; the other radii are produced for completeness.

Downstream: the cropped videos feed ``scripts/extract_embeddings.py`` /
``scripts/extract_full_queries_va.py``, which run Mask2Former and store the
258-d teacher embeddings.

Dependencies: opencv-python, numpy, tqdm, matplotlib.

Example:
    python preprocessing/crop_video_frames.py \\
        --base-video-dir /path/to/ASPED_v1 \\
        --output-dir /path/to/cropped_videos \\
        --temp-dir /tmp/kd_video_crop_temp
"""

import os
import sys
import argparse
import shutil

import cv2
import numpy as np
from tqdm import tqdm

# Make the sibling ``ellipses`` package importable regardless of the working
# directory the script is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ellipses.ellipses_va import (
    ellipse_va_5242023,
    ellipse_va_6012023,
    ellipse_va_6072023,
    ellipse_va_6212023,
    ellipse_va_6282023,
)

# Assumed standing-pedestrian height, used to extrude the ground buffer upward.
PERSON_REAL_HEIGHT_METERS = 2.0

# Buffer name -> real-world diameter in meters (radius x 2).
BUFFER_DIAMETER_MAP = {"1m": 2.0, "3m": 6.0, "6m": 12.0, "9m": 18.0}

# Session date -> the function that returns its per-station ground ellipses.
ELLIPSE_FUNCTION_MAP = {
    "5242023": ellipse_va_5242023,
    "6012023": ellipse_va_6012023,
    "6072023": ellipse_va_6072023,
    "6212023": ellipse_va_6212023,
    "6282023": ellipse_va_6282023,
}


def find_video_files(base_dir):
    """Scan ``base_dir`` for ASPED v.a ``.mp4`` files with a known session date.

    Expects the ASPED layout ``.../{Session_<date>}/{station}/.../<name>.mp4``.
    """
    video_files_info = []
    print(f"Starting video file scan in: {base_dir}")
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                try:
                    parts = video_path.split(os.sep)
                    station, session_name = parts[-4], parts[-5]
                    date_str = session_name.split("_")[-1]
                    video_name = os.path.splitext(file)[0]
                    if date_str in ELLIPSE_FUNCTION_MAP:
                        video_files_info.append(
                            (video_path, session_name, date_str, station, video_name)
                        )
                except IndexError:
                    pass
    return video_files_info


def process_single_video(video_path, session, date_str, station, video_name,
                         temp_base_dir, final_base_dir):
    """Read one source video and write one cropped video per buffer/recorder."""

    get_ellipse_func = ELLIPSE_FUNCTION_MAP.get(date_str)
    if not get_ellipse_func:
        return

    ellipse_data = get_ellipse_func(station)
    if not ellipse_data:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\n> Error opening video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writers = {}
    file_paths = {}  # roi_key -> (temp_path, final_path)
    rois = {}        # roi_key -> (x, y, w, h)

    for buffer_name, ellipses in ellipse_data.items():
        for i, ellipse in enumerate(ellipses):
            roi_key = f"{buffer_name}_ellipse_{i + 1}"

            real_world_diameter = BUFFER_DIAMETER_MAP.get(buffer_name)
            if not real_world_diameter:
                continue

            # Pixel-to-meter scale from the ground ellipse, then the person
            # height in pixels.
            ellipse_major_axis = max(ellipse.width, ellipse.height)
            pixel_to_meter_scale = ellipse_major_axis / real_world_diameter
            person_height_in_pixels = PERSON_REAL_HEIGHT_METERS * pixel_to_meter_scale

            # Tightest rotated box around the ground ellipse.
            center_cv = (int(ellipse.center[0]), int(ellipse.center[1]))
            axes_cv = (int(ellipse.width / 2), int(ellipse.height / 2))
            pts = cv2.ellipse2Poly(center_cv, axes_cv, int(ellipse.angle), 0, 360, 1)
            rotated_rect = cv2.minAreaRect(pts)
            ground_points = cv2.boxPoints(rotated_rect)

            # Extrude the ground box upward by the person height.
            sky_points = np.copy(ground_points)
            sky_points[:, 1] -= person_height_in_pixels

            # Axis-aligned bounding rectangle of the eight points is the crop.
            all_roi_points = np.concatenate((ground_points, sky_points), axis=0)
            (x, y, w, h) = cv2.boundingRect(all_roi_points)

            if w <= 0 or h <= 0:
                print(f"  > Warning: Invalid ROI size (w={w}, h={h}) for {roi_key}. Skipping.")
                continue

            rois[roi_key] = (x, y, w, h)

            temp_save_dir = os.path.join(temp_base_dir, session, station, video_name)
            final_save_dir = os.path.join(final_base_dir, session, station, video_name)
            os.makedirs(temp_save_dir, exist_ok=True)
            os.makedirs(final_save_dir, exist_ok=True)

            temp_video_path = os.path.join(temp_save_dir, f"{roi_key}.mp4")
            final_video_path = os.path.join(final_save_dir, f"{roi_key}.mp4")
            file_paths[roi_key] = (temp_video_path, final_video_path)

            # Write to the (fast) temp location first, move to final on close.
            video_writers[roi_key] = cv2.VideoWriter(temp_video_path, fourcc, fps, (w, h))

    # Crop every frame into each ROI writer.
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h_img, w_img = frame.shape[:2]

        for roi_key, (x, y, w, h) in rois.items():
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_img, x + w), min(h_img, y + h)
            cropped_frame = frame[y1:y2, x1:x2]

            if cropped_frame.size > 0:
                # Clamping at the image border can shrink the crop; resize back
                # to (w, h) so the writer sees a constant frame size.
                resized_crop = cv2.resize(cropped_frame, (w, h))
                if roi_key in video_writers:
                    video_writers[roi_key].write(resized_crop)

    cap.release()
    for writer in video_writers.values():
        writer.release()

    # Move the finished crops from temp to their final destination.
    for roi_key, (temp_path, final_path) in file_paths.items():
        if os.path.exists(temp_path):
            try:
                shutil.move(temp_path, final_path)
            except Exception as e:
                print(f"\n> Error moving file {temp_path} to {final_path}: {e}")


def parse_args():
    p = argparse.ArgumentParser(description="3D-projection crop of ASPED v.a video for the KD teacher.")
    p.add_argument("--base-video-dir", required=True,
                   help="Root of the source ASPED v.a videos (.../{Session_<date>}/{station}/.../*.mp4).")
    p.add_argument("--output-dir", required=True,
                   help="Destination root for the cropped videos.")
    p.add_argument("--temp-dir", default="/tmp/kd_video_crop_temp",
                   help="Fast scratch directory written first, then moved to --output-dir (default: %(default)s).")
    p.add_argument("--person-height-m", type=float, default=PERSON_REAL_HEIGHT_METERS,
                   help="Assumed standing-pedestrian height in meters (default: %(default)s).")
    return p.parse_args()


def main():
    global PERSON_REAL_HEIGHT_METERS
    args = parse_args()
    PERSON_REAL_HEIGHT_METERS = args.person_height_m

    os.makedirs(args.temp_dir, exist_ok=True)

    video_files = find_video_files(args.base_video_dir)
    print(f"\nFound a total of {len(video_files)} video files.\n")

    with tqdm(total=len(video_files), desc="Overall Video Cropping Progress", unit="video") as pbar:
        for video_path, session, date_str, station, video_name in video_files:
            pbar.set_description(f"Processing: {session}/{station}/{video_name}")
            try:
                process_single_video(video_path, session, date_str, station, video_name,
                                     args.temp_dir, args.output_dir)
            except Exception as e:
                print(f"\nError processing {video_path}: {e}")
            pbar.update(1)

    print(f"\n--- All video cropping tasks are complete. ---")
    print(f"Cropped videos can be found in: {args.output_dir}")

    try:
        if os.path.exists(args.temp_dir):
            shutil.rmtree(args.temp_dir)
            print(f"Cleaned up temporary directory: {args.temp_dir}")
    except Exception as e:
        print(f"\n> Warning: Could not clean up root temp dir: {e}")


if __name__ == "__main__":
    main()
