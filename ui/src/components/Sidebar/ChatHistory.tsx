import { 
  List, 
  Typography, 
  Box, 
  useTheme, 
  Accordion, 
  AccordionSummary, 
  AccordionDetails,
  styled 
} from '@mui/material';
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
  const [expandedUsers, setExpandedUsers] = useState<string[]>([]);

  // Get current user's identifier (using profile.name as the key)
  const currentUserId = user?.profile?.name;

  // Sort conversations to put current user first
  const sortedConversationEntries = useMemo(() => {
    const entries = Object.entries(conversations || {});
    
    return entries.sort(([uidA], [uidB]) => {
      // Current user always comes first
      if (uidA === currentUserId) {
        return -1;
      }
      if (uidB === currentUserId) {
        return 1;
      }
      // Sort others alphabetically
      return uidA.localeCompare(uidB);
    });
  }, [conversations, currentUserId]);

  // Auto-expand current user's accordion
  const handleAccordionChange = (userId: string) => (
    _event: React.SyntheticEvent,
    isExpanded: boolean
  ) => {
    setExpandedUsers(prev => 
      isExpanded 
        ? [...prev, userId]
        : prev.filter(id => id !== userId)
    );
  };

  // Initialize current user as expanded
  useMemo(() => {
    if (currentUserId && !expandedUsers.includes(currentUserId)) {
      setExpandedUsers(prev => [...prev, currentUserId]);
    }
  }, [currentUserId, expandedUsers]);

  const isUserExpanded = (userId: string) => expandedUsers.includes(userId);
  const isCurrentUser = (userId: string) => userId === currentUserId;

  return (
    <Box>
      <Typography variant="subtitle1" sx={{ mb: theme.spacing(1) }}>
        Conversations
      </Typography>
      
      {sortedConversationEntries.length ? (
        <Box sx={{ overflow: 'auto' }}>
          {sortedConversationEntries.map(([uid, chats]) => (
            <StyledAccordion
              key={uid}
              expanded={isUserExpanded(uid)}
              onChange={handleAccordionChange(uid)}
              disableGutters
            >
              <StyledAccordionSummary
                expandIcon={<ExpandMoreIcon />}
                sx={{
                  backgroundColor: isCurrentUser(uid) 
                    ? theme.palette.primary.main + '20' 
                    : 'transparent',
                  borderRadius: theme.spacing(0.5)
                }}
              >
                <UserLabel>
                  <PersonIcon 
                    fontSize="small" 
                    color={isCurrentUser(uid) ? 'primary' : 'action'}
                  />
                  <Typography
                    variant="subtitle2"
                    sx={{
                      fontWeight: isCurrentUser(uid) ? 'bold' : 'medium',
                      color: isCurrentUser(uid) 
                        ? theme.palette.primary.main 
                        : theme.palette.text.primary
                    }}
                  >
                    {isCurrentUser(uid) ? `${uid} (You)` : uid}
                  </Typography>
                  <Typography
                    variant="caption"
                    sx={{
                      ml: 'auto',
                      color: theme.palette.text.secondary
                    }}
                  >
                    {chats?.length || 0}
                  </Typography>
                </UserLabel>
              </StyledAccordionSummary>
              <StyledAccordionDetails>
                <List dense sx={{ padding: 0 }}>
                  {chats?.map(chat => (
                    <ChatItem
                      key={chat.id}
                      chatId={chat.id!}
                      chatTitle={chat.title || `Chat ${chat.id}`}
                    />
                  )) || []}
                </List>
              </StyledAccordionDetails>
            </StyledAccordion>
          ))}
        </Box>
      ) : (
        <Typography 
          variant="body2" 
          color="text.secondary"
          sx={{ textAlign: 'center', mt: theme.spacing(2) }}
        >
          No conversation history
        </Typography>
      )}
    </Box>
  );
};

export default ChatHistory;