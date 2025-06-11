import { List, Typography, Box, useTheme } from '@mui/material';
import ChatItem from './ChatItem';
import { useChat } from '../../chat';
import { Fragment, useMemo } from 'react';

const ChatHistory = () => {
  const { conversations } = useChat();
  const theme = useTheme();
  const conversationEntries = useMemo(() => Object.entries(conversations || {}), [conversations]);

  return (
    <Box>
      <Typography variant="subtitle1" sx={{ mb: theme.spacing(1) }}>
        Recent Conversations
      </Typography>
      
      {conversationEntries.length ? (
        <List sx={{ overflow: 'auto' }}>
          {conversationEntries.map(([uid, chats]) => (
            <Fragment key={uid}>
              <Typography
                variant="subtitle2"
                sx={{ mt: theme.spacing(2), mb: theme.spacing(1), fontWeight: 'bold' }}
              >
                {uid}
              </Typography>
              {chats?.map(chat => (
                <ChatItem
                  key={chat.id}
                  chatId={chat.id!}
                  chatTitle={chat.title || `Chat ${chat.id}`}
                />
              ))}
            </Fragment>
          ))}
        </List>
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