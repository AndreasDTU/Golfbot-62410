"""Temporal tracking and smoothing for ball field coordinates."""

from __future__ import annotations

from collections import deque
import math

import numpy as np

from perception.vision.geometry import CoordinateMapper
from perception.vision.models import BallDetection, SmoothedBallCoordinate, SmoothedCoordinateTrack


class BallCoordinateSmoother:
    """Smooth per-ball field coordinates with median filtering and stationary hold."""

    def __init__(
        self,
        mapper: CoordinateMapper | None = None,
        alpha_stationary: float = 0.12,
        alpha_moving: float = 0.35,
        max_match_distance_cm: float = 10.0,
        max_missed_frames: int = 5,
        median_window_size: int = 5,
        stationary_deadband_cm: float = 0.75,
        moving_threshold_cm: float = 2.0,
        stationary_confirm_frames: int = 3,
    ) -> None:
        self.mapper = mapper or CoordinateMapper()
        self.alpha_stationary = alpha_stationary
        self.alpha_moving = alpha_moving
        self.max_match_distance_cm = max_match_distance_cm
        self.max_missed_frames = max_missed_frames
        self.median_window_size = median_window_size
        self.stationary_deadband_cm = stationary_deadband_cm
        self.moving_threshold_cm = moving_threshold_cm
        self.stationary_confirm_frames = stationary_confirm_frames
        self.next_track_id = 0
        self.tracks: dict[int, SmoothedCoordinateTrack] = {}

    def reset(self) -> None:
        """Clear all smoothing state."""
        self.next_track_id = 0
        self.tracks.clear()

    def live_coordinates(self) -> list[SmoothedBallCoordinate]:
        """Return every currently-live track, not just the latest frame's hits.

        ``update`` returns only the balls matched in the frame it was called
        with.  After a multi-frame detection burst the live tracks hold the
        *union* of balls seen across the burst (each survives up to
        ``max_missed_frames`` non-detections), so a caller that runs several
        detection frames in a row can recover every ball that appeared in any
        of them.  Pixel fields are not retained per track and are reported as
        zero; callers needing them should use ``update``'s return value.
        """
        return [
            SmoothedBallCoordinate(
                track_id=track_id,
                label=track.label,
                center_px=(0, 0),
                corrected_center_px=(0, 0),
                radius_px=0,
                cm_x=track.x_cm,
                cm_y=track.y_cm,
            )
            for track_id, track in self.tracks.items()
        ]

    def update(
        self,
        detections: list[BallDetection],
        frame_shape: tuple[int, int, int],
    ) -> list[SmoothedBallCoordinate]:
        """Update smoothing tracks from the current frame's detections."""
        source_height, source_width = frame_shape[:2]
        observations = [
            (
                index,
                detection,
                self.mapper.pixel_to_field_cm(
                    detection.corrected_center,
                    (source_width, source_height),
                ),
            )
            for index, detection in enumerate(detections)
        ]

        matched_tracks: set[int] = set()
        matched_observations: set[int] = set()
        smoothed_results: dict[int, SmoothedBallCoordinate] = {}

        labels = sorted({detection.label for detection in detections})
        for label in labels:
            label_track_ids = [track_id for track_id, track in self.tracks.items() if track.label == label]
            label_observations = [
                (index, detection, cm_point)
                for index, detection, cm_point in observations
                if detection.label == label
            ]
            candidate_pairs: list[tuple[float, int, int]] = []

            for track_id in label_track_ids:
                track = self.tracks[track_id]
                for observation_index, _detection, (raw_x_cm, raw_y_cm) in label_observations:
                    distance_cm = math.hypot(raw_x_cm - track.x_cm, raw_y_cm - track.y_cm)
                    candidate_pairs.append((distance_cm, track_id, observation_index))

            candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
            for distance_cm, track_id, observation_index in candidate_pairs:
                if distance_cm > self.max_match_distance_cm:
                    continue
                if track_id in matched_tracks or observation_index in matched_observations:
                    continue

                track = self.tracks[track_id]
                detection = detections[observation_index]
                raw_x_cm, raw_y_cm = self.mapper.pixel_to_field_cm(
                    detection.corrected_center,
                    (source_width, source_height),
                )
                if track.history_x_cm.maxlen != self.median_window_size:
                    track.history_x_cm = deque(track.history_x_cm, maxlen=self.median_window_size)
                if track.history_y_cm.maxlen != self.median_window_size:
                    track.history_y_cm = deque(track.history_y_cm, maxlen=self.median_window_size)

                track.history_x_cm.append(raw_x_cm)
                track.history_y_cm.append(raw_y_cm)
                median_x_cm = float(np.median(np.array(track.history_x_cm, dtype=np.float32)))
                median_y_cm = float(np.median(np.array(track.history_y_cm, dtype=np.float32)))

                delta_cm = math.hypot(median_x_cm - track.x_cm, median_y_cm - track.y_cm)
                if delta_cm < self.stationary_deadband_cm:
                    track.stationary_frames += 1
                else:
                    track.stationary_frames = 0

                if track.stationary_frames >= self.stationary_confirm_frames:
                    smoothed_x_cm = track.x_cm
                    smoothed_y_cm = track.y_cm
                else:
                    alpha = self.alpha_moving if delta_cm >= self.moving_threshold_cm else self.alpha_stationary
                    smoothed_x_cm = alpha * median_x_cm + (1.0 - alpha) * track.x_cm
                    smoothed_y_cm = alpha * median_y_cm + (1.0 - alpha) * track.y_cm
                    track.x_cm = smoothed_x_cm
                    track.y_cm = smoothed_y_cm

                track.missed_frames = 0

                matched_tracks.add(track_id)
                matched_observations.add(observation_index)
                smoothed_results[observation_index] = SmoothedBallCoordinate(
                    track_id=track_id,
                    label=detection.label,
                    center_px=detection.center,
                    corrected_center_px=detection.corrected_center,
                    radius_px=detection.radius_px,
                    cm_x=smoothed_x_cm,
                    cm_y=smoothed_y_cm,
                )

        for observation_index, detection, (raw_x_cm, raw_y_cm) in observations:
            if observation_index in matched_observations:
                continue

            track_id = self.next_track_id
            self.next_track_id += 1
            new_track = SmoothedCoordinateTrack(
                label=detection.label,
                x_cm=raw_x_cm,
                y_cm=raw_y_cm,
            )
            new_track.history_x_cm.append(raw_x_cm)
            new_track.history_y_cm.append(raw_y_cm)
            self.tracks[track_id] = new_track
            matched_tracks.add(track_id)
            smoothed_results[observation_index] = SmoothedBallCoordinate(
                track_id=track_id,
                label=detection.label,
                center_px=detection.center,
                corrected_center_px=detection.corrected_center,
                radius_px=detection.radius_px,
                cm_x=raw_x_cm,
                cm_y=raw_y_cm,
            )

        stale_track_ids = []
        for track_id, track in self.tracks.items():
            if track_id in matched_tracks:
                continue
            track.missed_frames += 1
            if track.missed_frames > self.max_missed_frames:
                stale_track_ids.append(track_id)

        for track_id in stale_track_ids:
            del self.tracks[track_id]

        return [smoothed_results[index] for index in range(len(detections))]
