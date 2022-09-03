import http from "http";
import type { AddressInfo } from "net";
import { createServer } from "./server";

const host = process.env?.["HOST"] ?? "0.0.0.0";
const port = process.env?.["PORT"] ?? "4000";

const startSever = async () => {
  const app = await createServer();
  const server = http.createServer(app).listen({ host, port }, () => {
    const addressInfo = server.address() as AddressInfo;
    console.log(
      `Server ready at http://${addressInfo.address}:${addressInfo.port}`
    );
  });
};

startSever().catch((error) => console.error(error));
