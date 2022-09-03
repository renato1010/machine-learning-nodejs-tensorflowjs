import { Router } from "express";
import type * as core from "express-serve-static-core";
import type { Request, Response } from "express";
import { predictHousePrice } from "../utils";

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
