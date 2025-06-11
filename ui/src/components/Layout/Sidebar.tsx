import { Box, Divider, Typography, useTheme, IconButton } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useAuth } from '../../auth';
import ChatHistory from '../Sidebar/ChatHistory';
import NewChatButton from '../Sidebar/NewChatButton';
import Navigation from './Navigation';

const Sidebar = ({ onClose }: { onClose?: () => void }) => {
  const { user } = useAuth();
  const theme = useTheme();

  return (
    <Box
      sx={{
        width: 250,
        bgcolor: 'background.paper',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        padding: theme.spacing(2),
        position: 'sticky',
        top: 0,
        alignSelf: 'flex-start'
      }}
    >
      {onClose && (
        <IconButton
          onClick={onClose}
          sx={{ position: 'absolute', top: 8, right: 8 }}
          size="small"
        >
          <CloseIcon />
        </IconButton>
      )}
      <Typography variant="h6" sx={{ mb: theme.spacing(2) }}>
        Welcome, {user?.profile.name || 'User'}
      </Typography>
      <Navigation />
      <Box sx={{ mt: theme.spacing(2), mb: theme.spacing(2) }}>
        <NewChatButton />
      </Box>
      <Divider sx={{ my: theme.spacing(1) }} />
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        <ChatHistory />
      </Box>
    </Box>
  );
};

export default Sidebar;