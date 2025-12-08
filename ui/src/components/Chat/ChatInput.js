import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState, useRef } from 'react';
import { styled, FormControlLabel, Switch, useTheme, useMediaQuery } from '@mui/material';
import { useChat } from '../../chat';
import { PromptInput, PromptInputProvider, PromptInputBody, PromptInputTextarea, PromptInputHeader, PromptInputFooter, PromptInputSubmit, PromptInputTools, PromptInputButton, PromptInputActionMenu, PromptInputActionMenuTrigger, PromptInputActionMenuContent, PromptInputActionAddAttachments, PromptInputAttachments, PromptInputAttachment, PromptInputSpeechButton } from '../ai-elements/prompt-input';
import { MessageContentTypeValues } from '../../types/MessageContentType';
import { MessageRoleValues } from '../../types/MessageRole';
import { useConfigContext } from '../../context/ConfigContext';
import { getToken } from '../../api';
import { useAuth } from '../../auth';
import { listModelProfiles, updateModelProfile } from '../../api/model';
import { Image as ImageIcon, Stop as StopIcon } from '@mui/icons-material';
import { uuidv4 as uuid } from '../../lib/utils';
const InputContainer = styled('div')(({ theme }) => ({
    padding: theme.spacing(1.5),
    backgroundColor: theme.palette.background.default,
    borderTop: `1px solid ${theme.palette.divider}`,
    boxShadow: theme.shadows[2],
    [theme.breakpoints.down('sm')]: {
        padding: theme.spacing(1)
    }
}));
const StyledPromptInput = styled(PromptInput)(({ theme }) => ({
    maxWidth: theme.breakpoints.values.md,
    margin: '0 auto',
    [theme.breakpoints.down('sm')]: {
        maxWidth: '100%'
    }
}));
const ChatInput = () => {
    const { currentConversation, isTyping, isLoading, sendMessage, cancelRequest } = useChat();
    const theme = useTheme();
    const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
    const { config } = useConfigContext();
    const auth = useAuth();
    const [primaryProfileThink, setPrimaryProfileThink] = useState(false);
    const [thinkLoading, setThinkLoading] = useState(false);
    const textareaRef = useRef(null);
    const inProgress = isTyping || isLoading;
    // Load primary profile and its 'think' setting
    React.useEffect(() => {
        const loadPrimary = async () => {
            try {
                const primaryId = config?.model_profiles?.primary_profile_id || '';
                if (!primaryId) {
                    return;
                }
                const profiles = await listModelProfiles(getToken(auth.user));
                const primary = profiles.find(p => p.id === primaryId);
                if (primary && primary.parameters) {
                    setPrimaryProfileThink(!!primary.parameters.think);
                }
            }
            catch {
                // ignore errors silently for now
            }
        };
        loadPrimary();
    }, [config, auth.user]);
    const toggleThink = async () => {
        const primaryId = config?.model_profiles?.primary_profile_id || '';
        if (!primaryId) {
            return;
        }
        setThinkLoading(true);
        try {
            const profiles = await listModelProfiles(getToken(auth.user));
            const primary = profiles.find(p => p.id === primaryId);
            if (!primary) {
                return;
            }
            const updated = await updateModelProfile(getToken(auth.user), primaryId, {
                ...primary,
                parameters: {
                    ...primary.parameters,
                    think: !primary.parameters?.think
                }
            });
            setPrimaryProfileThink(!!updated.parameters?.think);
        }
        catch (error) {
            console.error('Failed toggling think on primary profile', error);
        }
        finally {
            setThinkLoading(false);
        }
    };
    const handleSubmit = async (message) => {
        if (!currentConversation?.id || !message.text.trim()) {
            return;
        }
        // {
        //   "text": "test",
        //     "files": [
        //       {
        //         "type": "file",
        //         "url": "data:text/x-python-script;base64,aW1wb3J0IHBhbmRhcyBhcyBwZApmcm9tIC5tbCBpbXBvcnQgTUxTdHJhdGVneQoKZGVmIG1sX3N0cmF0ZWd5KGRmLCB3aW5kb3c9MjAsIHRocmVzaG9sZD0wLjAwNSk6CiAgICBtbCA9IE1MU3RyYXRlZ3kod2luZG93PXdpbmRvdykKICAgIG1sLnRyYWluKGRmKQogICAgcmV0dXJuIG1sLmdlbmVyYXRlX3NpZ25hbChkZikK",
        //         "mediaType": "text/x-python-script",
        //         "filename": "file.py"
        //       }
        //     ]
        // }
        // Start with text content
        const content = [
            {
                type: MessageContentTypeValues.TEXT,
                text: message.text.trim()
            }
        ];
        // Process file attachments
        if (message.files && message.files.length > 0) {
            for (const file of message.files) {
                try {
                    // Convert file to base64 if it has a URL (blob URL)
                    let fileData;
                    let fileFormat;
                    if (file.url) {
                        if (file.url.startsWith('blob:')) {
                            // Fetch the blob and convert to base64
                            const response = await fetch(file.url);
                            const blob = await response.blob();
                            const arrayBuffer = await blob.arrayBuffer();
                            const uint8Array = new Uint8Array(arrayBuffer);
                            const binaryString = Array.from(uint8Array, byte => String.fromCharCode(byte)).join('');
                            fileData = btoa(binaryString);
                            fileFormat = file.mediaType || blob.type;
                        }
                        else if (file.url.startsWith('data:')) {
                            // Data URL - extract base64 data
                            const match = file.url.match(/^data:(.*?);base64,(.*)$/);
                            if (match) {
                                fileFormat = match[1];
                                fileData = match[2];
                            }
                        }
                    }
                    // Determine content type based on media type
                    let contentType = MessageContentTypeValues.FILE;
                    if (file.mediaType?.startsWith('image/')) {
                        contentType = MessageContentTypeValues.IMAGE;
                    }
                    else if (file.mediaType?.startsWith('audio/')) {
                        contentType = MessageContentTypeValues.AUDIO;
                    }
                    else if (file.mediaType?.startsWith('video/')) {
                        contentType = MessageContentTypeValues.VIDEO;
                    }
                    const contentItem = {
                        type: contentType,
                        name: file.filename || uuid(),
                        data: fileData,
                        format: fileFormat,
                        url: file.url && !file.url.startsWith('blob:') ? file.url : undefined
                    };
                    // Add file content to message
                    content.push(contentItem);
                }
                catch (error) {
                    console.error('Failed to process file attachment:', error);
                    // Continue with other files
                }
            }
        }
        console.log('Sending message content:', content);
        await sendMessage({
            role: MessageRoleValues.USER,
            content,
            conversation_id: currentConversation.id
        });
    };
    const handleCancel = async () => {
        try {
            await cancelRequest();
        }
        catch (error) {
            console.error('Failed to cancel request:', error);
        }
    };
    return (_jsx(PromptInputProvider, { children: _jsx(InputContainer, { children: _jsxs(StyledPromptInput, { onSubmit: handleSubmit, multiple: true, globalDrop: true, children: [_jsx(PromptInputHeader, { children: _jsx(PromptInputAttachments, { children: (attachment) => (_jsx(PromptInputAttachment, { data: attachment }, attachment.id)) }) }), _jsx(PromptInputBody, { children: _jsx(PromptInputTextarea, { ref: textareaRef, placeholder: !currentConversation
                                ? "No active conversation..."
                                : "Type your message...", disabled: !currentConversation?.id, className: "min-h-12 max-h-32" }) }), _jsxs(PromptInputFooter, { children: [_jsxs(PromptInputTools, { children: [_jsxs(PromptInputActionMenu, { children: [_jsx(PromptInputActionMenuTrigger, {}), _jsx(PromptInputActionMenuContent, { children: _jsx(PromptInputActionAddAttachments, { label: "Add files or images" }) })] }), _jsx(PromptInputSpeechButton, { textareaRef: textareaRef, disabled: !currentConversation?.id }), _jsx(PromptInputButton, { disabled: !currentConversation?.id, title: "Generate Image (Coming Soon)", children: _jsx(ImageIcon, { className: "text-muted-foreground", fontSize: "small" }) }), _jsx(FormControlLabel, { control: _jsx(Switch, { checked: primaryProfileThink, onChange: toggleThink, disabled: !config?.model_profiles?.primary_profile_id || thinkLoading, size: "small" }), label: "Think", sx: {
                                            mr: 1,
                                            '& .MuiFormControlLabel-label': {
                                                fontSize: isMobile ? '0.75rem' : '0.875rem'
                                            }
                                        } })] }), _jsx("div", { className: "flex items-center gap-2", children: inProgress ? (_jsx(PromptInputButton, { onClick: handleCancel, variant: "outline", size: "sm", title: "Stop Generation", className: "text-warning", children: _jsx(StopIcon, { fontSize: "small" }) })) : (_jsx(PromptInputSubmit, { status: inProgress ? 'streaming' : undefined, disabled: !currentConversation?.id })) })] })] }) }) }));
};
export default ChatInput;
