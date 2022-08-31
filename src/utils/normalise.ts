import type {  Tensor, Rank } from "@tensorflow/tfjs-node-gpu";

type Normalized2DResult = {
  tensor: Tensor<Rank>;
  min: Tensor<Rank>;
  max: Tensor<Rank>;
};
function normalize(
  tensor: Tensor<Rank>,
  passedMin: null | Tensor<Rank> = null,
  passedMax: null | Tensor<Rank> = null
): Normalized2DResult {
  const min = passedMin ?? tensor.min();
  const max = passedMax ?? tensor.max();
  const normalizedTensor = tensor.sub(min).div(max.sub(min));
  return { tensor: normalizedTensor, min, max };
}

function deNormalize(
  tensor: Tensor<Rank>,
  min: Tensor<Rank>,
  max: Tensor<Rank>
): Tensor<Rank> {
  return tensor.mul(max.sub(min)).add(min);
}

export { normalize, deNormalize };
