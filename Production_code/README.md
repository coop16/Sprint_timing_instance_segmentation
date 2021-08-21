# Timing Sprints with Mask R-CNN Instance Segmentation (Production Code)


__Steps to run code:__

1) Load .mp4 video into "input_video" directory.

2) Navigate to scripts folder.

3) Activate conda environment with correct package versions:

`activate tensorflow_1_14`

4) Run `time_sprint.py <video filename> <optional manual fps>`

- Requires argument for filename (e.g. "sprint_video.mp4")
- Optional argument for manual fps (e.g. if video is editted and fps in metadata shows frames per video second, but not actual time second).

5) Output will be .png file in the output folder with the original video name (without .mp4) and the sprint time in the filename (e.g. "sprint_video_time_1_113s.png")

