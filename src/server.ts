import express, { urlencoded, json, type Application } from "express";
import { housePriceRouter } from "./controllers";
import { errorHandler } from "./middlewares";

const createServer = (): Application => {
  const app = express();

  app.use(urlencoded({ extended: true }));
  app.use(json());

  app.disable("x-powered-by");
  app.get("/health", (_req, res) => res.send("UP"));
  app.use("/prediction/house-price-by-squareft", housePriceRouter);
  app.use(errorHandler);
  return app;
};

export { createServer };
