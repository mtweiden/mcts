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

from torch import bool
from torch import int32
from torch import stack
from torch import tensor
from torch import zeros
import torch.nn.functional as F

# ------------------------------------------------------------------------------
# Some type definitions and constants
# ------------------------------------------------------------------------------
ObsType = list[float]
PriorType = dict[int, float]
ValueType = float
BATCH_TIMEOUT = 0.005
MAX_BATCH_SIZE = 1024
MAX_INFERENCE_BATCH_SIZE = 100

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
    files = sorted([str(x) for x in Path(ckpt_path).glob("*")])
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

    def __init__(self, model: Agent) -> None:
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

            # Unpack Observation packets
            # Do collation and convert to tensors
            # 0: placement (list[int])
            # 1: objectives_0 (list[int])
            # 2: objectives_1 (list[int])
            # 3: height (int)
            # 4: width (int)
            # 5: action_masks (list[int])
            placements = []
            objectives_0 = []
            objectives_1 = []
            heights = []
            widths = []
            action_masks = []
            for p, o0, o1, h, w, va in [obs for batch in obs_all for obs in batch]:
                p = tensor(p, dtype=int32)
                o0 = tensor(o0, dtype=int32)
                o1 = tensor(o1, dtype=int32)
                if va:
                    action_mask = zeros((max(va) + 1,), dtype=bool)
                    action_mask[va] = True
                else:
                    action_mask = zeros((1,), dtype=bool)

                placements.append(p)
                objectives_0.append(o0)
                objectives_1.append(o1)
                heights.append(h)
                widths.append(w)
                action_masks.append(action_mask)
            
            # Run inference in chunks of MAX_INFERENCE_BATCH_SIZE (timed)
            infer_start = time.perf_counter()
            priors_all = []
            values_all = []
            total_obs = len(placements)
            if total_obs > 0:
                for chunk_start in range(0, total_obs, MAX_INFERENCE_BATCH_SIZE):
                    chunk_end = min(chunk_start + MAX_INFERENCE_BATCH_SIZE, total_obs)

                    # Pad tensors to max length in batch
                    max_p_len = max(len(p) for p in placements[chunk_start:chunk_end])
                    max_o0_len = max(len(o0) for o0 in objectives_0[chunk_start:chunk_end])
                    max_o1_len = max(len(o1) for o1 in objectives_1[chunk_start:chunk_end])
                    max_am_len = max(len(am) for am in action_masks[chunk_start:chunk_end])
                    for i in range(chunk_start, chunk_end):
                        p = placements[i]
                        o0 = objectives_0[i]
                        o1 = objectives_1[i]
                        am = action_masks[i]

                        if len(p) < max_p_len:
                            pad_size = max_p_len - len(p)
                            p = F.pad(p, (0, pad_size), "constant", 0)
                            placements[i] = p
                        if len(o0) < max_o0_len:
                            pad_size = max_o0_len - len(o0)
                            o0 = F.pad(o0, (0, pad_size), "constant", 0)
                            objectives_0[i] = o0
                        if len(o1) < max_o1_len:
                            pad_size = max_o1_len - len(o1)
                            o1 = F.pad(o1, (0, pad_size), "constant", 0)
                            objectives_1[i] = o1
                        if len(am) < max_am_len:
                            pad_size = max_am_len - len(am)
                            am = F.pad(am, (0, pad_size), "constant", False)
                            action_masks[i] = am

                    counts = [len(batch) for batch in obs_all]


                    # Chunk the inference requests
                    placements_chunk = stack(placements[chunk_start:chunk_end])
                    objectives_0_chunk = stack(objectives_0[chunk_start:chunk_end])
                    objectives_1_chunk = stack(objectives_1[chunk_start:chunk_end])
                    heights_chunk = tensor(heights[chunk_start:chunk_end])
                    widths_chunk = tensor(widths[chunk_start:chunk_end])
                    action_masks_chunk = stack(action_masks[chunk_start:chunk_end])

                    # Do inference
                    priors_chunk, values_chunk = self.model(
                        placements=placements_chunk,
                        objectives=objectives_0_chunk,
                        objectives_1=objectives_1_chunk,
                        heights=heights_chunk,
                        widths=widths_chunk,
                        action_masks=action_masks_chunk,
                    )

                    priors_all.extend(priors_chunk.detach().cpu().tolist())
                    values_all.extend(values_chunk.detach().cpu().tolist())

            infer_elapsed = time.perf_counter() - infer_start
            logging.info(f"[batcher] inference completed: {total_obs} observations in {infer_elapsed*1000:.2f}ms")

            # Send results back to request futures
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
            batch_size = len(placements)

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
