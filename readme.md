## Tensorflow in NodeJS

The project is based on the course I took at **Udemy** [Machine Learning in Javascript with Tensorflow.js](https://www.udemy.com/course/machine-learning-in-javascript-with-tensorflow-js/learn/lecture/15381100?start=0#overview)  
<br />
I gave it a twist, the author of the course used the Tensorflowjs library that can be used in the [browser](https://js.tensorflow.org/api/3.19.0/),  
I think that library is pretty cool because it uses WebGL(gpu) to run the Tensor processes, which is much faster than simple JavaScript.

I thought it could be cool if I tried to use the [Nodejs](https://js.tensorflow.org/api_node/3.19.0/) library which has [CUDA support](https://www.npmjs.com/package/@tensorflow/tfjs-node-gpu) and **serve the prediction as endpoint**

My laptop has a CUDA GPU(not a good one), but I notice that runs a bit faster than the Nodejs lib that has only C++ bindings,  
So I need to set the [CUDA-enabled](https://www.tensorflow.org/install/pip#linux) features first

Once you have the settings, the rest is easy😉

### Install dependencies

Nodejs lib with Cuda support:

```bash
npm i @tensorflow/tfjs-node-gpu
```

We need a **Dataset** for that go to [Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) and download `kc_house_data.csv` (**798 kB**)

It is interesting to note that we have options, we can host the dataset(.csv) prepare the features/labels,  
train the model then save that model to a local file and finally load that same model to make predictions.  
The other option may be that the model has been worked on, trained, tested elsewhere(python)  
and with our Nodejs server we can simply load it from the url. For me the idea I wanted to test was to  
do the round trip here with Nodejs and using Typescript (the Tensorflowjs lib is written in Typescript).

So doing everything in node will look like:

### Data preparation

- Save dataset(.csv) file locally and from there prepare data see: [create-features](src/utils/create-features.ts)
  interesting to note the format for saving csv file.

```typescript
// import dataset from CSV file
const houseSalesDataset = data.csv("file://data/kc_house_data.csv");
```

- Our problem will be: given a house living area square feet liven we will predict the price
  so we will get only two columns of data set:

```typescript
const pointsDataset = await houseSalesDataset.map((record) => ({
  //@ts-ignore
  x: record.sqft_living,
  //@ts-ignore
  y: record.price,
}));
```

- Will split the data in 2 parts 1 for training and the other for testing/validating  
  To do that we need to have an even number of records, so if out number of records is odd
  we need to remove one:

```typescript
const points = await pointsDataset.toArray();
// if odd number will eliminate last one so can split in two exactly
if (points.length % 2 !== 0) {
  points.pop();
}
```

- We'll need to shuffle data, to avoid bias on records order

```typescript
// shuffle the points
shuffle(points);
```

- This simple **linear regression** problem will have a 2D Tensor for both Features(input) and Labels(output)

```typescript
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
```

### Defining the model

For our simple model will have only one **layer**(Layers API) and that layer will have only one **node** as **Sequential model**
our activition function will be a **linear** input = output, our loss function is \*\*Mean Squared Error"  
`src/utils/create-model.ts`:

```typescript
function createModel(): Sequential {
  const model = sequential();
  model.add(
    layers.dense({ units: 1, useBias: true, activation: "linear", inputDim: 1 })
  );
  const optimizer = train.adam();
  model.compile({ loss: "meanSquaredError", optimizer });
  return model;
}
```

### Training the model

Having our **features**/**labels**
and running only 20 epochs  
`src/utils/train-model.ts`:

```typescript
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
```

_Those 20 epochs took no more than a minute to execute._

### Testing model

`src/housesalesdataset.ts (line#35)`

```typescript
const lossTensor = model.evaluate(testingFeature, testingLabel);
const loss = (lossTensor as Scalar).dataSync();
console.log(`Testing set loss: ${loss}`);
```

### Predict

Having the model **trained** and **tested**, we only have to load it from the saved location and run a **prediction**.  
`src/utils/predict-house-price.ts`:

```typescript
async function predictHousePrice(houseArea: number): Promise<number> {
  if (isNaN(houseArea)) {
    throw new Error("House are must be a number value");
  }
  const inputTensor = tensor1d([houseArea]);
  let { minFeature, maxFeature, minLabel, maxLabel } = await createFeatures();
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
  if (!outputValue) {
    throw new Error("Error getting predicted price");
  }
  const roundedOutput = +(outputValue / 1000).toFixed(0) * 1000;
  return roundedOutput;
}
```

### Expose prediction functionality as endpoint

_That's easy! just expose the prediction as an **ExpressJS** endpoint_
`src/server.ts`:

```typescript
app.use("/prediction/house-price-by-squareft", housePriceRouter);
```

`src/controllers/house-price.ts`:

```typescript
const housePriceRouter = Router();
interface PredictionBody {
  area: number;
}
interface PredictionResponse {
  data: { price: number };
}
housePriceRouter.post(
  "/",
  async (
    req: Request<core.ParamsDictionary, PredictionResponse, PredictionBody>,
    res: Response
  ) => {
    const { area } = req.body;
    const price = await predictHousePrice(area);
    res.status(200).json({ data: { price } });
  }
);
export { housePriceRouter };
```

### Hiting the endpoint

![prediction endpoint](https://icons-images.s3.us-east-2.amazonaws.com/prediction_endpoint_ok.png)
_A property of 2,000 square foot has a price of US$ 511,000_
