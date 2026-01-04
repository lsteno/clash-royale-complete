from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FrameFormat(_message.Message):
    __slots__ = ("width", "height", "channels", "color_model")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    COLOR_MODEL_FIELD_NUMBER: _ClassVar[int]
    width: int
    height: int
    channels: int
    color_model: str
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ..., channels: _Optional[int] = ..., color_model: _Optional[str] = ...) -> None: ...

class RegionOfInterest(_message.Message):
    __slots__ = ("x", "y", "width", "height")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    x: int
    y: int
    width: int
    height: int
    def __init__(self, x: _Optional[int] = ..., y: _Optional[int] = ..., width: _Optional[int] = ..., height: _Optional[int] = ...) -> None: ...

class ProcessFrameRequest(_message.Message):
    __slots__ = ("frame_id", "timestamp", "format", "frame_bgr", "card_hints", "want_action", "schema_version", "roi")
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    FRAME_BGR_FIELD_NUMBER: _ClassVar[int]
    CARD_HINTS_FIELD_NUMBER: _ClassVar[int]
    WANT_ACTION_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    ROI_FIELD_NUMBER: _ClassVar[int]
    frame_id: int
    timestamp: float
    format: FrameFormat
    frame_bgr: bytes
    card_hints: _containers.RepeatedScalarFieldContainer[str]
    want_action: bool
    schema_version: str
    roi: RegionOfInterest
    def __init__(self, frame_id: _Optional[int] = ..., timestamp: _Optional[float] = ..., format: _Optional[_Union[FrameFormat, _Mapping]] = ..., frame_bgr: _Optional[bytes] = ..., card_hints: _Optional[_Iterable[str]] = ..., want_action: bool = ..., schema_version: _Optional[str] = ..., roi: _Optional[_Union[RegionOfInterest, _Mapping]] = ...) -> None: ...

class StateGrid(_message.Message):
    __slots__ = ("channels", "height", "width", "values", "dtype")
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    channels: int
    height: int
    width: int
    values: _containers.RepeatedScalarFieldContainer[float]
    dtype: str
    def __init__(self, channels: _Optional[int] = ..., height: _Optional[int] = ..., width: _Optional[int] = ..., values: _Optional[_Iterable[float]] = ..., dtype: _Optional[str] = ...) -> None: ...

class Action(_message.Message):
    __slots__ = ("card_idx", "grid_x", "grid_y")
    CARD_IDX_FIELD_NUMBER: _ClassVar[int]
    GRID_X_FIELD_NUMBER: _ClassVar[int]
    GRID_Y_FIELD_NUMBER: _ClassVar[int]
    card_idx: int
    grid_x: int
    grid_y: int
    def __init__(self, card_idx: _Optional[int] = ..., grid_x: _Optional[int] = ..., grid_y: _Optional[int] = ...) -> None: ...

class ProcessFrameResponse(_message.Message):
    __slots__ = ("frame_id", "state_grid", "reward", "done", "action", "latency_ms", "ocr_failed", "info_str", "info_num", "model_version", "schema_version")
    class InfoStrEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class InfoNumEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_GRID_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    OCR_FAILED_FIELD_NUMBER: _ClassVar[int]
    INFO_STR_FIELD_NUMBER: _ClassVar[int]
    INFO_NUM_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    frame_id: int
    state_grid: StateGrid
    reward: float
    done: bool
    action: Action
    latency_ms: float
    ocr_failed: bool
    info_str: _containers.ScalarMap[str, str]
    info_num: _containers.ScalarMap[str, float]
    model_version: str
    schema_version: str
    def __init__(self, frame_id: _Optional[int] = ..., state_grid: _Optional[_Union[StateGrid, _Mapping]] = ..., reward: _Optional[float] = ..., done: bool = ..., action: _Optional[_Union[Action, _Mapping]] = ..., latency_ms: _Optional[float] = ..., ocr_failed: bool = ..., info_str: _Optional[_Mapping[str, str]] = ..., info_num: _Optional[_Mapping[str, float]] = ..., model_version: _Optional[str] = ..., schema_version: _Optional[str] = ...) -> None: ...

class HeartbeatRequest(_message.Message):
    __slots__ = ("schema_version", "model_version")
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    schema_version: str
    model_version: str
    def __init__(self, schema_version: _Optional[str] = ..., model_version: _Optional[str] = ...) -> None: ...

class HeartbeatResponse(_message.Message):
    __slots__ = ("schema_version", "model_version")
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    schema_version: str
    model_version: str
    def __init__(self, schema_version: _Optional[str] = ..., model_version: _Optional[str] = ...) -> None: ...
