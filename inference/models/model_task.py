from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, constr


class ModelTask(str, Enum):
    """ModelTask represents the task associated with a machine learning model"""

    TEXTTOTEXT = "TextToText"
    TEXTTOIMAGE = "TextToImage"
    IMAGETOTEXT = "ImageToText"
    IMAGETOIMAGE = "ImageToImage"
    TEXTTOAUDIO = "TextToAudio"
    AUDIOTOTEXT = "AudioToText"
    TEXTTOVIDEO = "TextToVideo"
    VIDEOTOTEXT = "VideoToText"
    TEXTTOSPEECH = "TextToSpeech"
    SPEECHTOTEXT = "SpeechToText"
    TEXTTOEMBEDDINGS = "TextToEmbeddings"
    VISIONTEXTTOTEXT = "VisionTextToText"
    IMAGETEXTTOIMAGE = "ImageTextToImage"
    TEXTTORANKING = "TextToRanking"
    IMAGETO3D = "ImageTo3D"
