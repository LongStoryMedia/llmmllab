import { jsx as _jsx } from "react/jsx-runtime";
import { 
// useEffect,
useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { AuthContext } from './useAuth';
import { Log } from 'oidc-client-ts';
import { userManager, logoutSession } from './userManager';
import config from '../config';
import { getUserInfo } from '../api';
Log.setLogger(console);
export const AuthProvider = ({ children }) => {
    const [isAuthenticated] = useState(true);
    const [evaluating, setEvaluating] = useState(true);
    const [user, setUser] = useState();
    const [userInfo, setUserInfo] = useState(undefined);
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
            }
            else {
                // If no redirect path, navigate to home or default page
                navigate('/');
            }
        };
        const checkAuthState = async (usr) => {
            if (usr) {
                setUser(usr);
                userManager.startSilentRenew();
                const groups = usr.profile.groups || [];
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
                }
                else {
                    try {
                        if (!(await checkAuthState(await userManager.signinSilent({ silentRequestTimeoutInSeconds: 5 })))) {
                            sessionStorage.setItem('redirectPath', location.pathname);
                            console.warn('Silent signin failed, redirecting to login');
                            await userManager.signinRedirect(config.auth.oidc);
                        }
                    }
                    catch {
                        sessionStorage.setItem('redirectPath', location.pathname);
                        console.warn('Silent signin failed, redirecting to login');
                        await userManager.signinRedirect(config.auth.oidc);
                    }
                }
            }
        })();
    }, [location.pathname, navigate]);
    return (_jsx(AuthContext.Provider, { value: { user, isAuthenticated, evaluating, userManager, logout, isAdmin, userInfo }, children: children }));
};
