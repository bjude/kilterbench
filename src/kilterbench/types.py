from typing import TypedDict


class GradeHistogramEntry(TypedDict):
    difficulty: int
    count: int


class QualityHistogramEntry(TypedDict):
    quality: int
    count: int


class AscentEntry(TypedDict):
    attempt_id: int
    angle: int
    quality: int
    difficulty: int
    is_benchmark: bool
    is_mirror: bool
    comment: str
    climbed_at: str
    user_id: int
    user_username: int
    user_avatar_image: None | str


class ClimbStats(TypedDict):
    difficulty: list[GradeHistogramEntry]
    quality: list[QualityHistogramEntry]
    ascents: list[AscentEntry]
