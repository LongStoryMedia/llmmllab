import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import { Button, TextField, Typography, Paper, Box, Alert, useTheme } from '@mui/material';
import { useAuth } from '../auth';
import LoadingAnimation from '../components/Shared/LoadingAnimation';
const LoginPage = () => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const auth = useAuth();
    const theme = useTheme();
    const handleLogin = async (e) => {
        e.preventDefault();
        setError(null);
        setIsLoading(true);
        try {
            await auth.userManager.getUser();
            // No need to navigate here as the AuthProvider will handle this
        }
        catch (error) {
            console.error('Login failed:', error);
            setError('Login failed. Please check your credentials and try again.');
        }
        finally {
            setIsLoading(false);
        }
    };
    return (_jsx(Box, { sx: {
            height: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: theme => theme.palette.background.default
        }, children: _jsxs(Paper, { elevation: 3, sx: {
                p: theme.spacing(4),
                maxWidth: 400,
                width: '100%',
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing(2)
            }, children: [_jsx(Typography, { variant: "h4", align: "center", gutterBottom: true, children: "Welcome" }), _jsx(Typography, { variant: "body1", align: "center", color: "textSecondary", children: "Sign in to access your chat assistant" }), error && (_jsx(Alert, { severity: "error", sx: { mt: theme.spacing(2) }, children: error })), _jsxs(Box, { component: "form", onSubmit: handleLogin, sx: { mt: theme.spacing(2) }, children: [_jsx(TextField, { label: "Username", variant: "outlined", fullWidth: true, margin: "normal", value: username, onChange: (e) => setUsername(e.target.value), disabled: isLoading, required: true }), _jsx(TextField, { label: "Password", type: "password", variant: "outlined", fullWidth: true, margin: "normal", value: password, onChange: (e) => setPassword(e.target.value), disabled: isLoading, required: true }), _jsx(Button, { variant: "contained", color: "primary", fullWidth: true, type: "submit", disabled: isLoading || !username || !password, sx: { mt: theme.spacing(3), mb: theme.spacing(2) }, children: isLoading ? _jsx(LoadingAnimation, {}) : 'Sign In' })] })] }) }));
};
export default LoginPage;
