import asyncio
from asyncio import Future
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi import Request
from pydantic import BaseModel
from uvicorn import run
# ------------------------------------------------------------------------------
# Some type definitions and constants
# ------------------------------------------------------------------------------
ObsType = list[float]
PriorType = dict[int, float]
ValueType = float
BATCH_TIMEOUT = 0.01
MAX_BATCH_SIZE = 64

# ------------------------------------------------------------------------------
# Data schemas
# ------------------------------------------------------------------------------
class InferenceRequest(BaseModel):
    # List of observations
    observation_batch: list[ObsType]


class InferenceResponse(BaseModel):
    # List of prior probability dicts and list of values
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


# Default dummy model for 2 ancilla case
MODEL = DummyModel(num_actions=11)

# ------------------------------------------------------------------------------
# Inference endpoint
# ------------------------------------------------------------------------------
class InferenceBatcher:
    def __init__(self) -> None:
        self.queue: list[tuple[list[ObsType], Future]] = []
        self.lock = asyncio.Lock()
    
    async def enqueue(self, obs: list[ObsType]) -> tuple[list[PriorType], list[ValueType]]:
        # Get currently running loop
        loop = asyncio.get_running_loop()
        fut: Future = loop.create_future()  # type: ignore
        async with self.lock:
            self.queue.append((obs, fut))
        return await fut
    
    async def _batch_worker(self) -> None:
        while True:
            await asyncio.sleep(BATCH_TIMEOUT)
            async with self.lock:
                if not self.queue:
                    continue
                batch = self.queue[:MAX_BATCH_SIZE]
                self.queue = self.queue[MAX_BATCH_SIZE:]

            # Combine all observations into a single tensor
            obs_all, futs = zip(*batch)
            obs_flat = [obs for obs_batch in obs_all for obs in obs_batch]
            counts = [len(obs_batch) for obs_batch in obs_all]

            priors_all, values_all = self._run_batch(obs_flat)

            # Split results per original request
            start = 0
            for count, fut in zip(counts, futs):
                end = start + count
                priors_slice = priors_all[start:end]
                values_slice = values_all[start:end]
                start = end
                if not fut.done():
                    fut.set_result((priors_slice, values_slice))
    
    def _run_batch(
        self,
        observations: list[ObsType],
    ) -> tuple[list[PriorType], list[ValueType]]:
        return MODEL.batch_infer(observations)


# ------------------------------------------------------------------------------
# Start up and Inference Endpoint
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.batcher = InferenceBatcher()
    asyncio.create_task(app.state.batcher._batch_worker())
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/infer", response_model=InferenceResponse)
async def infer(req: InferenceRequest, request: Request) -> InferenceResponse:
    batcher = request.app.state.batcher
    prior_batch, value_batch = await batcher.enqueue(req.observation_batch)
    return InferenceResponse(prior_batch=prior_batch, value_batch=value_batch)

# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000)