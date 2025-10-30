import asyncio
import time
import logging
import msgpack
from asyncio import Future
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
from pydantic import BaseModel
from uvicorn import run
from tile import Agent

# ------------------------------------------------------------------------------
# Some type definitions and constants
# ------------------------------------------------------------------------------
ObsType = list[float]
PriorType = dict[int, float]
ValueType = float
BATCH_TIMEOUT = 0.001
MAX_BATCH_SIZE = 1024

# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ------------------------------------------------------------------------------
# Data schemas
# ------------------------------------------------------------------------------
class InferenceRequest(BaseModel):
    # list of observations
    observation_batch: list[ObsType]


class InferenceResponse(BaseModel):
    # list of prior probability dicts and list of values
    prior_batch: list[PriorType]
    value_batch: list[ValueType]

# ------------------------------------------------------------------------------
# Inference model (dummy implementation)
# ------------------------------------------------------------------------------
class DummyModel:
    def __init__(self, num_actions: int) -> None:
        self.num_actions = num_actions
    
    def batch_infer(
        self,
        observations: list[ObsType],
    ) -> tuple[list[PriorType], list[ValueType]]:
        prior_batch = []
        value_batch = []
        for obs in observations:
            priors = {action: 1.0 / self.num_actions for action in range(self.num_actions)}
            value = -1.0
            prior_batch.append(priors)
            value_batch.append(value)
        return prior_batch, value_batch

def latest_checkpoint() -> str | None:
    ckpt_path = "/pscratch/sd/m/mtweiden/tile_mcts/checkpoints"
    files = sorted([_ for _ in Path(ckpt_path).glob("*")])
    if len(files) == 0:
        return None
    return files[-1]

MODEL = Agent()
ckpt = latest_checkpoint()
if ckpt is not None:
    MODEL.load_state(ckpt)

# ------------------------------------------------------------------------------
# Inference endpoint
# ------------------------------------------------------------------------------
class InferenceBatcher:
    """
    Lock-free batcher using an asyncio.Queue to aggregate concurrent requests.

    Each /infer call enqueues its observations and awaits a Future.
    A background worker periodically drains the queue and performs batched inference.
    """

    def __init__(self, model):
        self.queue: asyncio.Queue[tuple[list[ObsType], Future]] = asyncio.Queue()
        self.model = model

    async def enqueue(self, obs: list[ObsType]) -> tuple[list[PriorType], list[ValueType]]:
        loop = asyncio.get_running_loop()
        fut: Future = loop.create_future()
        await self.queue.put((obs, fut))
        return await fut

    async def _batch_worker(self):
        while True:
            obs_all, futs = [], []
            try:
                first_item = await self.queue.get()
                obs_all.append(first_item[0])
                futs.append(first_item[1])
            except asyncio.CancelledError:
                break

            start_time = time.perf_counter()
            start_loop_time = asyncio.get_running_loop().time()

            # Gather more until batch full or timeout
            while (
                len(obs_all) < MAX_BATCH_SIZE
                and (asyncio.get_running_loop().time() - start_loop_time) < BATCH_TIMEOUT
            ):
                try:
                    item = self.queue.get_nowait()
                    obs_all.append(item[0])
                    futs.append(item[1])
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0)
                    continue

            obs_flat = [obs for batch in obs_all for obs in batch]
            counts = [len(batch) for batch in obs_all]

            # Run inference
            priors_all, values_all = self.model(obs_flat)

            # Finish futures
            start_idx = 0
            for count, fut in zip(counts, futs):
                end_idx = start_idx + count
                priors_slice = priors_all[start_idx:end_idx]
                values_slice = values_all[start_idx:end_idx]
                start_idx = end_idx
                if not fut.done():
                    fut.set_result((priors_slice, values_slice))

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000.0
            queue_depth = self.queue.qsize()
            batch_size = len(obs_flat)

            m = f"[Batcher] size={batch_size:3d} latency={latency_ms:6.2f}ms queue_depth={queue_depth}"
            logging.info(m)

# ------------------------------------------------------------------------------
# Start up and Inference Endpoint
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.batcher = InferenceBatcher(MODEL)
    asyncio.create_task(app.state.batcher._batch_worker())
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/infer")
async def infer(request: Request) -> Response:
    raw = await request.body()
    data = msgpack.unpackb(raw, raw=False)
    obs_batch = data["observation_batch"]

    batcher = request.app.state.batcher
    prior_batch, value_batch = await batcher.enqueue(obs_batch)

    response_payload = {
        "prior_batch": prior_batch,
        "value_batch": value_batch,
    }
    packed = msgpack.packb(response_payload, use_bin_type=True)

    return Response(content=packed, media_type="application/msgpack")

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000)
