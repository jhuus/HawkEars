#!/usr/bin/env python3

from abc import ABC, abstractmethod


class HeuristicsManager(ABC):
    """Base class for species handlers."""

    @abstractmethod
    def process_recording(self, recording_path, frame_map):
        """Called once per recording. May mutate frame_map in place."""
        ...
