import {
  sequential,
  layers,
  train,
  type Sequential,
} from "@tensorflow/tfjs-node-gpu";

function createModel(): Sequential {
  const model = sequential();
  model.add(
    layers.dense({ units: 1, useBias: true, activation: "linear", inputDim: 1 })
  );
  const optimizer = train.adam();
  model.compile({ loss: "meanSquaredError", optimizer });
  return model;
}

export { createModel };
