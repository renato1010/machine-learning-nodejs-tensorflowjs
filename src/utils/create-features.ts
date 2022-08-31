import { data, tensor2d, util } from "@tensorflow/tfjs-node-gpu";
import { normalize } from "./normalise";

const shuffle = util.shuffle;

async function createFeatures() {
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
  let {
    tensor: normalizedFeatureTensor,
    min: minFeature,
    max: maxFeature,
  } = normalize(featureTensor);
  let {
    tensor: normalizedLabelTensor,
    min: minLabel,
    max: maxLabel,
  } = normalize(labelTensor);
  return {
    normalizedFeatureTensor,
    normalizedLabelTensor,
    minFeature,
    maxFeature,
    minLabel,
    maxLabel,
  };
}

export { createFeatures };
