# Local Video Recordings

Put recorded camera videos for `tools/topdown_object_detector.py` in this folder.

Example:

```sh
python3 tools/topdown_object_detector.py --video videos/run-001.mp4
```

If OpenCV reads the recording at a different resolution than the saved camera
calibration, replay it with an explicit resize override:

```sh
python3 tools/topdown_object_detector.py --video videos/run-001.mov --resize-video-to-calibration
```

Video replay loops automatically when the file reaches the end. Press `q` or
`Esc` to close the detector.

Recordings are ignored by git because they are usually large. Keep videos at the
same resolution as `calibration_data.npz`, otherwise undistortion will reject
the frames unless the resize override is used. The override is useful for replay
debugging, but recording at the calibration resolution is still preferred for
geometry-sensitive testing.
