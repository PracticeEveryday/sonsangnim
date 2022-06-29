import User from "../schemas/user";
import { IUserModel } from "../../models";
export class MongoUserModel implements IUserModel {
  async studyList(userId: string) {
    const studyList = await User.findById(userId, { _id: 0, study: 1 });
    return studyList;
  }

  async study(userId: string, word: string) {
    const user = await User.findById(userId);
    user.study.push(word);
    user.save();
    return user;
  }

  async createUser(userData) {
    const user = await User.create(userData);
    return user;
  }

  async updateUser(userId, updateUserData) {
    const user = await User.findByIdAndUpdate(userId, { $set: updateUserData }, { new: true });
    return user;
  }

  async deleteUser(userId: string) {
    const user = await User.findByIdAndDelete(userId);
    return user;
  }

  async findByEmail(email: string) {
    const user = await User.findOne({ email }).lean();
    return user;
  }

  async findById(userId: string) {
    const user = await User.findById(userId).lean();
    return user;
  }

  async pushScore(userId: string, newScore: number) {
    const user = await User.findById(userId).lean();
    const updatedUser = await User.findByIdAndUpdate(
      userId,
      {
        $push: {
          scores: {
            username: user.username,
            score: newScore,
            createdAt: new Date(),
          },
        },
      },
      { new: true },
    );
    return updatedUser;
  }
}
