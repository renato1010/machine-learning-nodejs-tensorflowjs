import {
  split,
  type Tensor,
  type Rank,
  type Scalar,
} from "@tensorflow/tfjs-node-gpu";
import { createModel, trainModel, createFeatures } from "./utils";

const getDataSet = async () => {
  try {
    let { normalizedFeatureTensor, normalizedLabelTensor } =
      await createFeatures();

    const [trainingFeature, testingFeature] = split<Tensor<Rank>>(
      normalizedFeatureTensor,
      2
    );
    const [trainingLabel, testingLabel] = split<Tensor<Rank>>(
      normalizedLabelTensor,
      2
    );
    const model = createModel();
    // const layer = model.getLayer(undefined, 0);
    if (!trainingFeature || !trainingLabel) {
      throw new Error("No training dataset available");
    }
    const result = await trainModel(model, trainingFeature, trainingLabel);
    const trainingLoss = (result.history?.["loss"] as number[])?.at(-1);
    console.log({ trainingLoss: trainingLoss ?? "training loss error" });
    const validationLoss = result.history["val_loss"]?.at(-1);
    console.log({ validationLoss });
    if (!testingFeature || !testingLabel) {
      throw new Error("No testing dataset available");
    }
    const lossTensor = model.evaluate(testingFeature, testingLabel);
    const loss = (lossTensor as Scalar).dataSync();
    console.log(`Testing set loss: ${loss}`);
    // save model
    const saveResults = await model.save(
      "file://models/house-price-regression-v1"
    );
    console.log({ saveResults });
  } catch (error) {
    console.error(error);
  }
};

(async () => {
  await getDataSet();
})();
