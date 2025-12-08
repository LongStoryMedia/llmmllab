import { ReactNode } from 'react';
import { UserManager, User } from 'oidc-client-ts';
import { UserInfo } from '../api';
export interface AuthContextType {
    isAuthenticated: boolean;
    evaluating: boolean;
    userManager: UserManager;
    user?: User;
    isAdmin: boolean;
    userInfo?: UserInfo;
    logout: () => Promise<void>;
}
export declare const AuthProvider: ({ children }: {
    children: ReactNode;
}) => import("react/jsx-runtime").JSX.Element;
