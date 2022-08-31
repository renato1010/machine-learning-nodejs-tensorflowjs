import type { Rank, Sequential, Tensor } from "@tensorflow/tfjs-node-gpu";

async function trainModel(
  model: Sequential,
  trainingFeatureTensor: Tensor<Rank>,
  trainingLabelTensor: Tensor<Rank>
) {
  return model.fit(trainingFeatureTensor, trainingLabelTensor, {
    epochs: 20,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, log) =>
        console.log(`Epoch ${epoch}: loss = ${log?.["loss"]}`),
    },
  });
}

export { trainModel };
