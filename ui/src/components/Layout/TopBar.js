import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { AppBar, Toolbar, Typography, Button, useTheme, IconButton, Box } from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import { useAuth } from '../../auth';
import ThemeToggle from '../Shared/ThemeToggle';
import useColorMode from '../../hooks/useColorMode';
// import Icon from '../Shared/Icon';
// import Title from '../Shared/Title';
const TopBar = ({ onMenuClick }) => {
    const { user, logout } = useAuth();
    const theme = useTheme();
    const [mode, setMode] = useColorMode();
    return (_jsx(AppBar, { children: _jsxs(Toolbar, { children: [onMenuClick && (_jsx(IconButton, { color: "inherit", edge: "start", onClick: onMenuClick, sx: { mr: 2 }, children: _jsx(MenuIcon, {}) })), _jsxs(Box, { sx: { flexGrow: 1, display: 'flex', justifyContent: 'left' }, children: [_jsx("img", { src: "/nurturebot2.png", alt: "Logo", style: { height: 75 } }), _jsx(Typography, { variant: "h1", component: "div", sx: {
                                ml: 2,
                                color: theme.palette.primary.light,
                                alignSelf: 'center',
                                fontWeight: 300
                            }, children: "llmmllab" })] }), user?.profile.name && (_jsxs(Typography, { variant: "body1", sx: { mr: theme.spacing(2.5) }, children: ["Welcome, ", user.profile.name] })), _jsx(ThemeToggle, { mode: mode, setMode: setMode }), _jsx(Button, { color: "inherit", onClick: logout, children: "Logout" })] }) }));
};
export default TopBar;
