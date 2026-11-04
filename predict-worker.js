/* ONNX inference runs in a worker so model computation does not block the UI. */
importScripts("vendor/ort.wasm.min.js");
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("vendor/", self.location.href).href;
ort.env.logLevel = "error";
const sessions = new Map();
let queue = Promise.resolve();

async function getSession(model) {
  if (!["cnn", "lstm", "cnn_lstm", "cnn_lstm_aug"].includes(model)) throw new Error("Unknown recognition model.");
  if (!sessions.has(model)) {
    const loading = ort.InferenceSession.create(new URL(`models/${model}.onnx`, self.location.href).href, { executionProviders: ["wasm"], graphOptimizationLevel: "all" })
      .then(async session => {
        const output = await session.run({ sensors: new ort.Tensor("float32", new Float32Array(128 * 6), [1, 128, 6]) });
        Object.values(output).forEach(tensor => tensor.dispose());
        return session;
      });
    sessions.set(model, loading);
    loading.catch(() => sessions.delete(model));
  }
  return sessions.get(model);
}

self.onmessage = ({ data }) => {
  queue = queue.then(async () => {
    try {
      const session = await getSession(data.model);
      if (data.type === "load") { self.postMessage({ id: data.id, ready: true }); return; }
      const input = new ort.Tensor("float32", data.values, [data.count, 128, 6]);
      const started = performance.now();
      const output = await session.run({ sensors: input });
      const milliseconds = performance.now() - started;
      const scores = Float32Array.from(output.probabilities.data);
      input.dispose();
      Object.values(output).forEach(tensor => tensor.dispose());
      self.postMessage({ id: data.id, scores, milliseconds }, [scores.buffer]);
    } catch (error) {
      self.postMessage({ id: data.id, error: error.message || String(error) });
    }
  });
};

