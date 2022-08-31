import {
  tensor1d,
  loadLayersModel,
  type Tensor,
  type Rank,
} from "@tensorflow/tfjs-node-gpu";
import { createFeatures } from "./create-features";
import { deNormalize, normalize } from "./normalise";

async function predictHousePrice(houseArea: number): Promise<number> {
  if (isNaN(houseArea)) {
    throw new Error("House are must be a number value");
  }
  const inputTensor = tensor1d([houseArea]);
  let { minFeature, maxFeature, minLabel, maxLabel } = await createFeatures();
  minFeature.print();
  maxFeature.print();
  const { tensor: inputTensorNormalized } = normalize(
    inputTensor,
    minFeature,
    maxFeature
  );
  const loadedModel = await loadLayersModel(
    "file://models/house-price-regression-v1/model.json"
  );
  const normalizedOutputTensor = loadedModel.predict(inputTensorNormalized);
  if (!minLabel || !maxLabel) {
    throw new Error("Missing parameters to run de-noramalization");
  }
  const outputTensor = deNormalize(
    normalizedOutputTensor as Tensor<Rank>,
    minLabel,
    maxLabel
  );
  const outputValue = outputTensor.dataSync()[0];
  console.log({ outputValue });
  if (!outputValue) {
    throw new Error("Error getting predicted price");
  }
  const roundedOutput = +(outputValue / 1000).toFixed(0) * 1000;
  return roundedOutput;
}

(async () => {
  const price1 = await predictHousePrice(2000);
  console.log({ price1 });
})();

export { predictHousePrice };
