import {
  data,
  tensor2d,
  util,
  split,
  type Tensor,
  type Rank,
  type Scalar,
} from "@tensorflow/tfjs-node-gpu";
import { normalize, createModel, trainModel } from "./utils";

const shuffle = util.shuffle;
let normalizedFeatureTensor: Tensor<Rank> | undefined;
let minFeature: Tensor<Rank> | undefined;
let maxFeature: Tensor<Rank> | undefined;
let normalizedLabelTensor: Tensor<Rank> | undefined;
let minLabel: Tensor<Rank> | undefined;
let maxLabel: Tensor<Rank> | undefined;
const getDataSet = async () => {
  try {
    // import dataset from CSV file
    const houseSalesDataset = data.csv("file://data/kc_house_data.csv");
    // get x,y values
    const pointsDataset = await houseSalesDataset.map((record) => ({
      //@ts-ignore
      x: record.sqft_living,
      //@ts-ignore
      y: record.price,
    }));
    const points = await pointsDataset.toArray();
    // if odd number will eliminate last one so can split in two exactly
    if (points.length % 2 !== 0) {
      points.pop();
    }
    // shuffle the points
    shuffle(points);
    // Features (input)
    const featureValues = points.map(({ x }) => x as number);
    const featureTensor = tensor2d(featureValues, [featureValues.length, 1]);
    // Labels (output)
    const labelValues = points.map(({ y }) => y as number);
    const labelTensor = tensor2d(labelValues, [labelValues.length, 1]);
    // normalize Tensors
    ({
      tensor: normalizedFeatureTensor,
      min: minFeature,
      max: maxFeature,
    } = normalize(featureTensor));
    ({
      tensor: normalizedLabelTensor,
      min: minLabel,
      max: maxLabel,
    } = normalize(labelTensor));
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
    const loss = await (lossTensor as Scalar).dataSync();
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

export {
  normalizedFeatureTensor,
  normalizedLabelTensor,
  minFeature,
  minLabel,
  maxFeature,
  maxLabel,
};
