"""Test audio/video artifact logging and rendering

Run with: pytest tests/domains/test_audio_video.py -v
"""

import time

import pytest

import artifacta as ds
from tests.helpers.audio import create_test_audio_tone
from tests.helpers.video import create_test_video_animation


def run_parameter_sweep(project_name, base_config, param_variations, run_fn):
    """Helper to run multiple experiments with parameter variations"""
    for idx, variation in enumerate(param_variations):
        config = {**base_config, **variation}
        run_name = f"{project_name}-run-{idx + 1}"
        ds.init(project=project_name, name=run_name, config=config)
        run_fn(config, seed=42 + idx)
        time.sleep(0.3)


@pytest.mark.e2e
def test_audio_playback():
    """Test single audio artifact logging and playback"""

    def run_test(config, seed=42):
        run = ds.get_run()

        # Create test audio file (440Hz tone, 1 second)
        audio_path = create_test_audio_tone(frequency=440, duration=1.0, filename="test_tone.wav")

        # Log as output artifact
        run.log_output(audio_path)

    run_parameter_sweep("audio-test-single", {}, [{}], run_test)


@pytest.mark.e2e
def test_audio_collection():
    """Test logging multiple audio files (C major chord)"""

    def run_test(config, seed=42):
        import os

        run = ds.get_run()

        # Create directory for audio files
        audio_dir = os.path.join("uploads", "audio_collection")
        os.makedirs(audio_dir, exist_ok=True)

        frequencies = [261, 329, 392, 523]  # C, E, G, C (C major chord)
        notes = ["C4", "E4", "G4", "C5"]

        for freq, note in zip(frequencies, notes):
            filename = f"note_{note}.wav"
            create_test_audio_tone(
                frequency=freq, duration=2.0, filename=os.path.join("audio_collection", filename)
            )

        # Log entire directory as single artifact
        run.log_output(audio_dir, name="audio_collection")

    run_parameter_sweep("audio-test", {}, [{}], run_test)


@pytest.mark.e2e
def test_video_collection():
    """Test logging multiple video files (different animations)"""

    def run_test(config, seed=42):
        import os

        run = ds.get_run()

        # Create directory for video files
        video_dir = os.path.join("uploads", "video_collection")
        os.makedirs(video_dir, exist_ok=True)

        # Create different animated videos
        videos = [
            ("fast_animation.mp4", 1, 30),  # name, duration, fps
            ("slow_animation.mp4", 3, 10),
            ("medium_animation.mp4", 2, 20),
        ]

        for name, duration, fps in videos:
            create_test_video_animation(
                width=320,
                height=240,
                duration_seconds=duration,
                fps=fps,
                filename=os.path.join("video_collection", name),
            )

        # Log entire directory as single artifact
        run.log_output(video_dir, name="video_collection")

    run_parameter_sweep("video-test", {}, [{}], run_test)
