use axum::body::Bytes;
use axum::{
    serve,
    extract::State,
    routing::post,
    response::IntoResponse,
    Router,
};
use mcts::Agent;
use mcts::agent::DummyAgent;
use rmp_serde::to_vec_named;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{sleep, Duration};
use tokio::net::TcpListener;

/// ---------------------------------------------------------------------------
/// Configuration
/// ---------------------------------------------------------------------------
const MAX_BATCH_SIZE: usize = 1024;
const BATCH_TIMEOUT: Duration = Duration::from_millis(10);

/// ---------------------------------------------------------------------------
/// Request / Response types
/// ---------------------------------------------------------------------------
#[derive(Deserialize, Debug)]
struct InferenceRequest {
    observation_batch: Vec<Vec<usize>>,
}

#[derive(Serialize, Debug)]
struct InferenceResponse {
    prior_batch: Vec<HashMap<usize, f32>>,
    value_batch: Vec<f32>,
}

/// ---------------------------------------------------------------------------
/// InferenceBatcher
/// ---------------------------------------------------------------------------
struct InferenceBatcher {
    tx: mpsc::UnboundedSender<(InferenceRequest, oneshot::Sender<InferenceResponse>)>,
}

impl InferenceBatcher {
    fn new(agent: Arc<DummyAgent>) -> Self {
        let (tx, mut rx) =
            mpsc::unbounded_channel::<(InferenceRequest, oneshot::Sender<InferenceResponse>)>();

        // background worker
        tokio::spawn(async move {
            let mut buffer: Vec<(InferenceRequest, oneshot::Sender<InferenceResponse>)> = Vec::new();

            loop {
                // wait for at least one request
                if let Some(item) = rx.recv().await {
                    buffer.push(item);
                }

                // short delay to accumulate more
                sleep(BATCH_TIMEOUT).await;

                // pull additional requests up to max batch
                while buffer.len() < MAX_BATCH_SIZE {
                    match rx.try_recv() {
                        Ok(item) => buffer.push(item),
                        Err(_) => break,
                    }
                }

                if buffer.is_empty() {
                    continue;
                }

                // Flatten all observations
                let obs_all: Vec<Vec<usize>> = buffer
                    .iter()
                    .flat_map(|(req, _)| req.observation_batch.clone())
                    .collect();
                let counts: Vec<usize> = buffer
                    .iter()
                    .map(|(req, _)| req.observation_batch.len())
                    .collect();

                // Run inference
                let (priors_all, values_all) = agent.batch_infer(&obs_all);

                // Send results back
                let mut start = 0;
                for ((_, tx), count) in buffer.drain(..).zip(counts) {
                    let end = start + count;
                    let resp = InferenceResponse {
                        prior_batch: priors_all[start..end].to_vec(),
                        value_batch: values_all[start..end].to_vec(),
                    };
                    let _ = tx.send(resp);
                    start = end;
                }
            }
        });

        Self { tx }
    }

    async fn infer(&self, req: InferenceRequest) -> InferenceResponse {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx.send((req, reply_tx)).unwrap();
        reply_rx.await.unwrap()
    }
}

/// ---------------------------------------------------------------------------
/// HTTP handler
/// ---------------------------------------------------------------------------
async fn infer_handler(
    State(batcher): State<Arc<InferenceBatcher>>,
    body: Bytes,
) -> impl IntoResponse {
    let req: InferenceRequest = rmp_serde::from_slice(&body).unwrap();
    let resp = batcher.infer(req).await;
    let packed = to_vec_named(&resp).unwrap();

    (
        [(axum::http::header::CONTENT_TYPE, "application/msgpack")],
        packed,
    )
}

/// ---------------------------------------------------------------------------
/// Entry point
/// ---------------------------------------------------------------------------
#[tokio::main]
async fn main() {
    let agent = Arc::new(DummyAgent::new(11));
    let batcher = Arc::new(InferenceBatcher::new(agent));

    let app = Router::new()
        .route("/infer", post(infer_handler))
        .with_state(batcher);

    let addr = "0.0.0.0:8000".parse::<std::net::SocketAddr>().unwrap();
    let listener = TcpListener::bind(addr).await.unwrap();
    println!("🚀 Server running on http://{}", addr);
    serve(listener, app.into_make_service()).await.unwrap();
}
