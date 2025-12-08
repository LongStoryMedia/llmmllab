import { LllabUser, NewUserReq, UserInfo } from "./types";
export declare const getUserInfo: () => Promise<UserInfo[]>;
export declare const getAllUserInfo: () => Promise<UserInfo[]>;
export declare const getLllabUsers: () => Promise<LllabUser[]>;
export declare const updatePassword: (oldPassword: string, newPassword: string) => Promise<{
    message: string;
    success: boolean;
}>;
export declare const addUser: (newUser: NewUserReq) => Promise<{
    message: string;
    success: boolean;
}>;
export declare const deleteUser: (username: string) => Promise<{
    message: string;
    success: boolean;
}>;
