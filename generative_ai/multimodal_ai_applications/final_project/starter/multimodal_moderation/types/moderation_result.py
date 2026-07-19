from pydantic import BaseModel, Field


class ModerationResult(BaseModel):

    rationale: str = Field(description="Explanation of what was harmful and why")


class TextModerationResult(ModerationResult):

    contains_pii: bool = Field(description="Whether the message contains any personally-identifiable information (PII)")
    is_unfriendly: bool = Field(description="Whether unfriendly tone or content was detected")
    is_unprofessional: bool = Field(description="Whether unprofessional tone or content was detected")

    @property
    def is_flagged(self) -> bool:
        return any([self.contains_pii, self.is_unfriendly, self.is_unprofessional])


class ImageModerationResult(ModerationResult):

    contains_pii: bool = Field(
        description="Whether the image contains any person, part of a person, or personally-identifiable information (PII)"
    )
    is_disturbing: bool = Field(description="Whether the image is disturbing")
    is_low_quality: bool = Field(description="Whether the image is low quality")

    @property
    def is_flagged(self) -> bool:
        return any([self.contains_pii, self.is_disturbing, self.is_low_quality])


class VideoModerationResult(ModerationResult):

    contains_pii: bool = Field(
        description="Whether the video contains any person or personally-identifiable information (PII)"
    )
    is_disturbing: bool = Field(description="Whether the video is disturbing")
    is_low_quality: bool = Field(description="Whether the video is low quality")

    @property
    def is_flagged(self) -> bool:
        return any([self.contains_pii, self.is_disturbing, self.is_low_quality])


class AudioModerationResult(ModerationResult):

    transcription: str = Field(description="Transcription of the audio content")
    contains_pii: bool = Field(description="Whether the audio contains personally-identifiable information (PII)")
    is_unfriendly: bool = Field(description="Whether unfriendly tone or content was detected")
    is_unprofessional: bool = Field(description="Whether unprofessional tone or content was detected")

    @property
    def is_flagged(self) -> bool:
        return any([self.contains_pii, self.is_unfriendly, self.is_unprofessional])
