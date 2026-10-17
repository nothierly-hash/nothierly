from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class TradeRequest(_message.Message):
    __slots__ = ("uuid", "asset", "fiat", "volume", "max_slippage")
    UUID_FIELD_NUMBER: _ClassVar[int]
    ASSET_FIELD_NUMBER: _ClassVar[int]
    FIAT_FIELD_NUMBER: _ClassVar[int]
    VOLUME_FIELD_NUMBER: _ClassVar[int]
    MAX_SLIPPAGE_FIELD_NUMBER: _ClassVar[int]
    uuid: str
    asset: str
    fiat: str
    volume: float
    max_slippage: float
    def __init__(self, uuid: _Optional[str] = ..., asset: _Optional[str] = ..., fiat: _Optional[str] = ..., volume: _Optional[float] = ..., max_slippage: _Optional[float] = ...) -> None: ...

class TradeResponse(_message.Message):
    __slots__ = ("status", "executed_price", "timestamp")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    EXECUTED_PRICE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    status: str
    executed_price: float
    timestamp: float
    def __init__(self, status: _Optional[str] = ..., executed_price: _Optional[float] = ..., timestamp: _Optional[float] = ...) -> None: ...
