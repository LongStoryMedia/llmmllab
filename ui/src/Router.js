import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import './index.css';
import ChatPage from './pages/ChatPage';
import SettingsPage from './pages/SettingsPage';
import { Route, Routes } from 'react-router-dom';
import ModelProfilesPage from './pages/ModelProfilesPage';
import ImagePage from './pages/ImagePage';
function Router() {
    return (_jsxs(Routes, { children: [_jsx(Route, { path: "/", element: _jsx(ChatPage, {}) }), _jsx(Route, { path: "/chat/:conversationId", element: _jsx(ChatPage, {}) }), _jsx(Route, { path: "/images", element: _jsx(ImagePage, {}) }), _jsx(Route, { path: "/settings/:tab?", element: _jsx(SettingsPage, {}) }), _jsx(Route, { path: "/model-profiles", element: _jsx(ModelProfilesPage, {}) })] }));
}
export default Router;
