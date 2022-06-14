import mongoose from "mongoose";
import * as dotenv from "dotenv";
dotenv.config();

import { HandModel } from "./models/Hand";
import { UserModel } from "./models/User";

const MONGO_URL: any = process.env.MONGO_URL;

mongoose
  .connect(MONGO_URL)
  .then(() => console.log(`${MONGO_URL} 연결 성공`))
  .catch(() => console.log("몽고 디비 연결 실패 ㅠㅠ"));

export { HandModel, UserModel };
