import { jsx as _jsx } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
import { Box, Alert, Fade, IconButton } from '@mui/material';
import { Close as CloseIcon } from '@mui/icons-material';
const FloatingNotification = ({ message, duration = 4000, onClose }) => {
    const [visible, setVisible] = useState(true);
    useEffect(() => {
        if (duration > 0) {
            const timer = setTimeout(() => {
                setVisible(false);
                setTimeout(() => onClose?.(), 300); // Wait for fade out animation
            }, duration);
            return () => clearTimeout(timer);
        }
    }, [duration, onClose]);
    const handleClose = () => {
        setVisible(false);
        setTimeout(() => onClose?.(), 300);
    };
    return (_jsx(Fade, { in: visible, timeout: 300, children: _jsx(Alert, { severity: "info", variant: "filled", action: _jsx(IconButton, { size: "small", "aria-label": "close", color: "inherit", onClick: handleClose, children: _jsx(CloseIcon, { fontSize: "small" }) }), sx: {
                mb: 1,
                fontSize: '0.875rem',
                '& .MuiAlert-message': {
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    maxWidth: '300px'
                }
            }, children: message }) }));
};
const FloatingNotifications = ({ messages, className }) => {
    const [notifications, setNotifications] = useState([]);
    useEffect(() => {
        if (messages && messages.length > 0) {
            const newNotifications = messages.map((message, index) => ({
                id: Date.now() + index,
                message
            }));
            setNotifications(prev => [...prev, ...newNotifications]);
        }
    }, [messages]);
    const removeNotification = (id) => {
        setNotifications(prev => prev.filter(n => n.id !== id));
    };
    if (notifications.length === 0) {
        return null;
    }
    return (_jsx(Box, { className: className, sx: {
            position: 'fixed',
            top: 20,
            right: 20,
            zIndex: 1300,
            maxWidth: 400,
            pointerEvents: 'auto'
        }, children: notifications.map(notification => (_jsx(FloatingNotification, { message: notification.message, onClose: () => removeNotification(notification.id) }, notification.id))) }));
};
export default FloatingNotifications;
