import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Config
from .engine import VIEWS, Engine

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("EVAP_CONFIG", ROOT / "config.json"))

engine = Engine(CONFIG_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI):
    engine.start()
    yield
    engine.stop()


app = FastAPI(title="evaporwave", lifespan=lifespan)


def check_view(view: str):
    if view not in VIEWS:
        raise HTTPException(400, f"view must be one of {VIEWS}")


@app.get("/api/config")
def get_config() -> Config:
    return engine.get_config()


@app.put("/api/config")
def put_config(cfg: Config) -> Config:
    engine.set_config(cfg)
    return cfg


@app.post("/api/panic")
def panic():
    engine.panic()
    return {"ok": True}


@app.get("/api/status")
def status():
    return engine.status()


class PickRequest(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


@app.post("/api/pick")
def pick(req: PickRequest):
    color = engine.pick_color(req.x, req.y)
    if color is None:
        raise HTTPException(503, "no frame yet")
    return {"color": color}


@app.get("/snapshot.jpg")
async def snapshot(view: str = "raw"):
    check_view(view)
    for _ in range(50):
        latest = engine.jpeg(view)
        if latest:
            return Response(latest[1], media_type="image/jpeg")
        await asyncio.sleep(0.05)
    raise HTTPException(503, "no frame yet")


@app.get("/stream.mjpg")
async def stream(view: str = "raw"):
    check_view(view)

    async def frames():
        last = -1
        while True:
            latest = engine.jpeg(view)
            if latest and latest[0] != last:
                last, jpg = latest
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            await asyncio.sleep(0.02)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    try:
        while True:
            await socket.send_json(engine.status())
            await asyncio.sleep(1 / 20)
    except (WebSocketDisconnect, RuntimeError):
        pass


class UncachedStatic(StaticFiles):
    """Makes browsers revalidate the UI files so updates show without a hard reload."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/", UncachedStatic(directory=ROOT / "static", html=True), name="static")
