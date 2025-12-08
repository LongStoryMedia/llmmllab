import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useRef } from 'react';
import { IconButton, Menu, MenuItem, ListItemIcon, ListItemText, Fade, Box } from '@mui/material';
import { MoreVert as MoreVertIcon, Delete as DeleteIcon, Replay as ReplayIcon, Edit as EditIcon } from '@mui/icons-material';
import { useChat } from '../../chat';
const MessageActions = ({ message, isUser }) => {
    const [anchorEl, setAnchorEl] = useState(null);
    const { deleteMessage, replayMessage, startEditMessage } = useChat();
    const open = Boolean(anchorEl);
    const buttonRef = useRef(null);
    const handleClick = (event) => {
        setAnchorEl(event.currentTarget);
    };
    const handleClose = () => {
        setAnchorEl(null);
    };
    const handleDelete = async () => {
        if (message.id) {
            try {
                await deleteMessage(message.id);
                handleClose();
            }
            catch (error) {
                console.error('Failed to delete message:', error);
                // Error handling is managed by the chat context
            }
        }
    };
    const handleReplay = async () => {
        try {
            await replayMessage(message);
            handleClose();
        }
        catch (error) {
            console.error('Failed to replay message:', error);
            // Error handling is managed by the chat context
        }
    };
    const handleEdit = () => {
        startEditMessage(message);
        handleClose();
    };
    return (_jsxs(Box, { sx: { position: 'relative' }, children: [_jsx(Fade, { in: true, timeout: 300, children: _jsx(IconButton, { ref: buttonRef, size: "small", onClick: handleClick, sx: {
                        opacity: 0.6,
                        transition: 'opacity 0.2s ease-in-out',
                        '&:hover': {
                            opacity: 1,
                            backgroundColor: isUser ? 'primary.dark' : 'action.hover'
                        },
                        color: isUser ? 'primary.contrastText' : 'text.secondary',
                        padding: '4px'
                    }, "aria-label": "Message actions", children: _jsx(MoreVertIcon, { fontSize: "small" }) }) }), _jsxs(Menu, { anchorEl: anchorEl, open: open, onClose: handleClose, transformOrigin: { horizontal: 'right', vertical: 'top' }, anchorOrigin: { horizontal: 'right', vertical: 'bottom' }, slotProps: {
                    paper: {
                        sx: {
                            borderRadius: '12px',
                            minWidth: '120px',
                            boxShadow: '0 4px 20px rgba(0,0,0,0.15)',
                            border: '1px solid',
                            borderColor: 'divider'
                        }
                    }
                }, children: [isUser && (_jsxs(MenuItem, { onClick: handleEdit, sx: {
                            color: 'warning.main',
                            '&:hover': {
                                backgroundColor: 'warning.light',
                                '& .MuiListItemIcon-root': {
                                    color: 'warning.dark'
                                }
                            },
                            borderRadius: '8px',
                            margin: '4px'
                        }, children: [_jsx(ListItemIcon, { children: _jsx(EditIcon, { fontSize: "small", color: "warning" }) }), _jsx(ListItemText, { primary: "Edit & Replay" })] })), isUser && (_jsxs(MenuItem, { onClick: handleReplay, sx: {
                            color: 'primary.main',
                            '&:hover': {
                                backgroundColor: 'primary.light',
                                '& .MuiListItemIcon-root': {
                                    color: 'primary.dark'
                                }
                            },
                            borderRadius: '8px',
                            margin: '4px'
                        }, children: [_jsx(ListItemIcon, { children: _jsx(ReplayIcon, { fontSize: "small", color: "primary" }) }), _jsx(ListItemText, { primary: "Replay" })] })), _jsxs(MenuItem, { onClick: handleDelete, sx: {
                            color: 'error.main',
                            '&:hover': {
                                backgroundColor: 'error.light',
                                '& .MuiListItemIcon-root': {
                                    color: 'error.dark'
                                }
                            },
                            borderRadius: '8px',
                            margin: '4px'
                        }, children: [_jsx(ListItemIcon, { children: _jsx(DeleteIcon, { fontSize: "small", color: "error" }) }), _jsx(ListItemText, { primary: "Delete" })] })] })] }));
};
export default MessageActions;
