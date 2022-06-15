// ./src/customType/express.d.ts
// eslint-disable-next-line no-unused-vars
import User from "../models/schemas/user";

declare global {
  // eslint-disable-next-line no-unused-vars
  namespace Express {
    // eslint-disable-next-line no-unused-vars
    interface Request {
      user?: string;
    }
  }
}
