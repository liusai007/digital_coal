from pydantic import BaseModel
from typing import List
from io import BytesIO


class CoalRadar(BaseModel):
    id: int
    name: str
    ip: str
    port: int
    axisX: float
    axisY: float
    shiftX: float
    shiftY: float
    shiftZ: float
    rotateX: float
    rotateY: float
    rotateZ: float
    bytes_buffer: bytes = bytes()


class HeapPoint(BaseModel):
    x: float
    y: float


class CoalHeap(BaseModel):
    coalHeapId: int
    coalHeapName: str
    coalHeapArea: List[HeapPoint]
    density: float = None
    mesId: int = None


class CoalYard(BaseModel):
    coalYardId: int
    coalYardName: str
    coalRadarList: List[CoalRadar] = None
    coalHeapList: List[CoalHeap]
    conn_radarsBucket: list = []


class InventoryCoalResult:
    coalHeapId: int
    coalHeapName: str
    volume: float
    maxHeight: float
    cloudInfo: str
    density: float = None
    mesId: int = None
    bytes_buffer: bytes
