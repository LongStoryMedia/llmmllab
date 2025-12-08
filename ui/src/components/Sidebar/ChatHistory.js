import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { List, Typography, Box, useTheme, Accordion, AccordionSummary, AccordionDetails, styled } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import PersonIcon from '@mui/icons-material/Person';
import ChatItem from './ChatItem';
import { useChat } from '../../chat';
import { useAuth } from '../../auth';
import { useMemo, useState } from 'react';
const StyledAccordion = styled(Accordion)({
    backgroundColor: 'transparent',
    boxShadow: 'none',
    '&:before': {
        display: 'none'
    },
    '&.Mui-expanded': {
        margin: 0
    }
});
const StyledAccordionSummary = styled(AccordionSummary)(({ theme }) => ({
    padding: theme.spacing(0.5, 1),
    minHeight: '36px',
    '&.Mui-expanded': {
        minHeight: '36px'
    },
    '& .MuiAccordionSummary-content': {
        margin: theme.spacing(0.5, 0),
        alignItems: 'center',
        '&.Mui-expanded': {
            margin: theme.spacing(0.5, 0)
        }
    }
}));
const StyledAccordionDetails = styled(AccordionDetails)(({ theme }) => ({
    padding: theme.spacing(0, 1, 1, 2)
}));
const UserLabel = styled(Box)(({ theme }) => ({
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(1)
}));
const ChatHistory = () => {
    const { conversations } = useChat();
    const { user } = useAuth();
    const theme = useTheme();
    const [expandedUsers, setExpandedUsers] = useState([]);
    // Get current user's identifier (using preferred_username which matches the username in conversations)
    const currentUserId = user?.profile?.preferred_username;
    // Sort conversations to put current user first
    const sortedConversationEntries = useMemo(() => {
        const entries = Object.entries(conversations || {});
        return entries.sort(([usernameA], [usernameB]) => {
            // Current user always comes first
            if (usernameA === currentUserId) {
                return -1;
            }
            if (usernameB === currentUserId) {
                return 1;
            }
            // Sort others alphabetically
            return usernameA.localeCompare(usernameB);
        });
    }, [conversations, currentUserId]);
    // Auto-expand current user's accordion
    const handleAccordionChange = (username) => (_event, isExpanded) => {
        setExpandedUsers(prev => isExpanded
            ? [...prev, username]
            : prev.filter(id => id !== username));
    };
    // Initialize current user as expanded
    useMemo(() => {
        if (currentUserId && !expandedUsers.includes(currentUserId)) {
            setExpandedUsers(prev => [...prev, currentUserId]);
        }
    }, [currentUserId, expandedUsers]);
    const isUserExpanded = (username) => expandedUsers.includes(username);
    const isCurrentUser = (username) => username === currentUserId;
    return (_jsxs(Box, { children: [_jsx(Typography, { variant: "subtitle1", sx: { mb: theme.spacing(1) }, children: "Conversations" }), sortedConversationEntries.length ? (_jsx(Box, { sx: { overflow: 'auto' }, children: sortedConversationEntries.map(([username, chats]) => (_jsxs(StyledAccordion, { expanded: isUserExpanded(username), onChange: handleAccordionChange(username), disableGutters: true, children: [_jsx(StyledAccordionSummary, { expandIcon: _jsx(ExpandMoreIcon, {}), sx: {
                                backgroundColor: isCurrentUser(username)
                                    ? theme.palette.primary.main + '20'
                                    : 'transparent',
                                borderRadius: theme.spacing(0.5)
                            }, children: _jsxs(UserLabel, { children: [_jsx(PersonIcon, { fontSize: "small", color: isCurrentUser(username) ? 'primary' : 'action' }), _jsx(Typography, { variant: "subtitle2", sx: {
                                            fontWeight: isCurrentUser(username) ? 'bold' : 'medium',
                                            color: isCurrentUser(username)
                                                ? theme.palette.primary.main
                                                : theme.palette.text.primary
                                        }, children: isCurrentUser(username) ? `${username} (You)` : username }), _jsx(Typography, { variant: "caption", sx: {
                                            ml: 'auto',
                                            color: theme.palette.text.secondary
                                        }, children: chats?.length || 0 })] }) }), _jsx(StyledAccordionDetails, { children: _jsx(List, { dense: true, sx: { padding: 0 }, children: chats?.map(chat => (_jsx(ChatItem, { chatId: chat.id, chatTitle: chat.title || `Chat ${chat.id}` }, chat.id))) || [] }) })] }, username))) })) : (_jsx(Typography, { variant: "body2", color: "text.secondary", sx: { textAlign: 'center', mt: theme.spacing(2) }, children: "No conversation history" }))] }));
};
export default ChatHistory;
