import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect, useCallback } from 'react';
import { Box, TextField, IconButton, Paper, Fade, Typography } from '@mui/material';
import { Check as CheckIcon, Close as CloseIcon } from '@mui/icons-material';
import { useChat } from '../../chat';
const MessageEditor = ({ messageId, initialContent }) => {
    const { saveEditAndReplay, cancelEditMessage, editingMessageContent } = useChat();
    const [localContent, setLocalContent] = useState(initialContent);
    const [isSaving, setIsSaving] = useState(false);
    // Update local content when the global editing content changes (for controlled input)
    useEffect(() => {
        setLocalContent(editingMessageContent);
    }, [editingMessageContent]);
    const handleSave = useCallback(async () => {
        if (!localContent.trim()) {
            return;
        }
        setIsSaving(true);
        try {
            await saveEditAndReplay(messageId, localContent);
        }
        catch (error) {
            console.error('Failed to save and replay message:', error);
        }
        finally {
            setIsSaving(false);
        }
    }, [messageId, localContent, saveEditAndReplay]);
    const handleCancel = useCallback(() => {
        setLocalContent(initialContent);
        cancelEditMessage();
    }, [initialContent, cancelEditMessage]);
    const handleKeyDown = useCallback((event) => {
        if (event.key === 'Enter' && event.ctrlKey) {
            event.preventDefault();
            handleSave();
        }
        else if (event.key === 'Escape') {
            event.preventDefault();
            handleCancel();
        }
    }, [handleSave, handleCancel]);
    return (_jsx(Fade, { in: true, timeout: 300, children: _jsxs(Paper, { sx: {
                p: 2,
                backgroundColor: 'background.default',
                border: '2px solid',
                borderColor: 'primary.main',
                borderRadius: 2,
                position: 'relative'
            }, children: [_jsx(Box, { sx: { mb: 1 }, children: _jsx(Typography, { variant: "caption", color: "primary", sx: { fontWeight: 'medium' }, children: "Editing message (Ctrl+Enter to save, Esc to cancel)" }) }), _jsx(TextField, { multiline: true, fullWidth: true, value: localContent, onChange: (e) => setLocalContent(e.target.value), onKeyDown: handleKeyDown, placeholder: "Edit your message...", variant: "outlined", minRows: 3, maxRows: 10, autoFocus: true, disabled: isSaving, sx: {
                        mb: 2,
                        '& .MuiOutlinedInput-root': {
                            backgroundColor: 'background.paper'
                        }
                    } }), _jsxs(Box, { sx: { display: 'flex', justifyContent: 'flex-end', gap: 1 }, children: [_jsx(IconButton, { size: "small", onClick: handleCancel, disabled: isSaving, sx: {
                                color: 'error.main',
                                '&:hover': {
                                    backgroundColor: 'error.light'
                                }
                            }, title: "Cancel editing (Esc)", children: _jsx(CloseIcon, { fontSize: "small" }) }), _jsx(IconButton, { size: "small", onClick: handleSave, disabled: isSaving || !localContent.trim(), sx: {
                                color: 'primary.main',
                                '&:hover': {
                                    backgroundColor: 'primary.light'
                                }
                            }, title: "Save and replay (Ctrl+Enter)", children: _jsx(CheckIcon, { fontSize: "small" }) })] })] }) }));
};
export default MessageEditor;
