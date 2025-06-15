import {
  // useEffect,
  useState,
  ReactNode,
  useEffect
} from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { AuthContext } from './useAuth';
import { UserManager, Log, User } from 'oidc-client-ts';
import { userManager, logoutSession } from './userManager';
import config from '../config';
import { getUserInfo, UserInfo } from '../api';
Log.setLogger(console);
export interface AuthContextType {
    isAuthenticated: boolean;
    evaluating: boolean;
    userManager: UserManager;
    user?: User;
    isAdmin: boolean;
    userInfo?: UserInfo;
    logout: () => Promise<void>;
}

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [isAuthenticated] = useState(true);
  const [evaluating, setEvaluating] = useState(true);
  const [user, setUser] = useState<User>();
  const [userInfo, setUserInfo] = useState<UserInfo | undefined>(undefined);
  const [isAdmin, setIsAdmin] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  const logout = async () => {
    await logoutSession();
    userManager.stopSilentRenew();
    setUser(undefined);
    setEvaluating(true);
  };

  useEffect(() => {
    setEvaluating(true);

    const redirect = () => {
      const redirectPath = sessionStorage.getItem('redirectPath');
      if (redirectPath) {
        sessionStorage.removeItem('redirectPath');
        navigate(redirectPath);
      } else {
        // If no redirect path, navigate to home or default page
        navigate('/');
      }
    };

    const checkAuthState = async (usr: User | null) => {
      if (usr) {
        setUser(usr);
        userManager.startSilentRenew();
        const groups = usr.profile.groups as string[] || [];
        setIsAdmin(groups.includes('admins'));
        const userInfo = await getUserInfo();
        setUserInfo(userInfo[0]);
        setEvaluating(false);
        return true;
      } 
      console.warn('User not authenticated');
      return false;
    };

    (async () => {
      if (!(await checkAuthState(await userManager.getUser()))) {
        if (location.pathname === '/callback') {
          if (!(await checkAuthState((await userManager.signinCallback()) ?? null))) {
            console.error('User not found after signin callback');
          }
          redirect();
        } else {
          try {
            if (!(await checkAuthState(await userManager.signinSilent({silentRequestTimeoutInSeconds: 5})))) {
              sessionStorage.setItem('redirectPath', location.pathname);
              console.warn('Silent signin failed, redirecting to login');
              await userManager.signinRedirect(config.auth.oidc);
            }
          } catch {
            sessionStorage.setItem('redirectPath', location.pathname);
            console.warn('Silent signin failed, redirecting to login');
            await userManager.signinRedirect(config.auth.oidc);
          }
        }
      }
    })();
  }, [location.pathname, navigate]);

  return (
    <AuthContext.Provider value={{ user, isAuthenticated, evaluating, userManager, logout, isAdmin, userInfo }}>
      {children}
    </AuthContext.Provider>
  );
};
