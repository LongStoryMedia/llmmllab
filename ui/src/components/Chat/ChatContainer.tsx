import { Box } from '@mui/material';
import { memo } from 'react';
import useScrollContainerRef from '../../hooks/useScrollContainerRef';

interface ChatContainerProps {
  children: React.ReactNode;
}

const ChatContainer: React.FC<ChatContainerProps> = memo(({ children }) => {
  const scrollContainerRef = useScrollContainerRef();
  return (
    <Box 
      sx={{ 
        display: 'flex', 
        flexDirection: 'column', 
        height: '100%',
        overflow: 'hidden'
      }} 
    >
      <Box 
        sx={{ 
          flex: 1, 
          p: 2, 
          overflowY: 'auto',
          pb: 8 // Account for input area
        }}
        ref={scrollContainerRef}
      >
        {children}
      </Box>
    </Box>
  );
});

export default ChatContainer;