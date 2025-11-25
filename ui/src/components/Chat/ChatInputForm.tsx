import React, { useState, useEffect } from 'react';
import { Box, IconButton, Tooltip, Input, useTheme, useMediaQuery, Dialog, DialogTitle, DialogContent, DialogContentText, DialogActions, Button, FormControlLabel, Switch } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import SendIcon from '@mui/icons-material/Send';
import ImageIcon from '@mui/icons-material/Image';
import { useChat } from '../../chat';
import { CancelOutlined } from '@mui/icons-material';
import { useConfigContext } from '../../context/ConfigContext';
import { getToken } from '../../api';
import { useAuth } from '../../auth';
import { listModelProfiles, updateModelProfile } from '../../api/model';

interface ChatInputFormProps {
  input: string;
  setInput: (input: string) => void;
  selectedOptions: string[];
  handleSend: () => void;
}

const ChatInputForm: React.FC<ChatInputFormProps> = ({ input, setInput, selectedOptions, handleSend }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const { isPaused, cancelRequest } = useChat();
  const { currentConversation, isTyping } = useChat();
  const [confirmOpen, setConfirmOpen] = useState(false);
  const { config } = useConfigContext();
  const auth = useAuth();
  const [primaryProfileThink, setPrimaryProfileThink] = useState<boolean>(false);
  const [thinkLoading, setThinkLoading] = useState(false);

  const handleKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  // Load primary profile and its 'think' setting
  useEffect(() => {
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
      } catch {
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
      const updated = await updateModelProfile(getToken(auth.user), primaryId!, {
        ...primary,
        parameters: {
          ...primary.parameters,
          think: !primary.parameters?.think
        }
      });
      setPrimaryProfileThink(!!updated.parameters?.think);
    } catch (error) {
      console.error('Failed toggling think on primary profile', error);
    } finally {
      setThinkLoading(false);
    }
  };

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
      <IconButton sx={{ color: theme.palette.text.secondary, padding: isMobile ? '4px' : '8px' }}>
        <AddIcon fontSize={isMobile ? 'small' : 'medium'} />
      </IconButton>
      <Input
        fullWidth
        placeholder={selectedOptions.includes('generateImage')
          ? "Enter a prompt to generate an image..."
          : currentConversation
            ? "Type your message..."
            : "No active conversation..."
        }
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyPress}
        multiline
        maxRows={4}
        disabled={(isTyping || !currentConversation) && !isPaused}
        sx={{
          flexGrow: 1,
          fontSize: isMobile ? '0.875rem' : '1rem'
        }}
      />
      <Tooltip title={selectedOptions.includes('generateImage') ? "Generate Image" : "Send Message"} arrow>
        <IconButton
          sx={{
            color: theme.palette.text.secondary,
            alignContent: 'end',
            padding: isMobile ? '4px' : '8px'
          }}
          onClick={handleSend}
          type='submit'
          color={selectedOptions.includes('generateImage') ? "secondary" : "primary"}
        >
          {selectedOptions.includes('generateImage') ? <ImageIcon fontSize={isMobile ? 'small' : 'medium'} /> : <SendIcon fontSize={isMobile ? 'small' : 'medium'} />}
        </IconButton>
      </Tooltip>

      <Tooltip title="Cancel request" arrow>
        <IconButton
          color="inherit"
          onClick={() => setConfirmOpen(true)}
          size="small"
          sx={{ mr: 1 }}
        // disabled={!currentConversation || (!isTyping && !isPaused)}
        >
          <CancelOutlined fontSize={isMobile ? 'small' : 'medium'} />
        </IconButton>
      </Tooltip>

      <FormControlLabel
        control={<Switch checked={primaryProfileThink} onChange={toggleThink} disabled={!config?.model_profiles?.primary_profile_id || thinkLoading} />}
        label="Think"
      />

      <Dialog
        open={confirmOpen}
        onClose={() => setConfirmOpen(false)}
        aria-labelledby="cancel-confirm-title"
      >
        <DialogTitle id="cancel-confirm-title">Cancel Generation?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Cancelling will stop the current LLM generation and may clear in-memory pipelines. This can make the next response slower when models need to be recreated. Are you sure you want to cancel?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmOpen(false)} color="primary">No, keep generating</Button>
          <Button
            onClick={() => {
              setConfirmOpen(false);
              cancelRequest();
            }}
            color="secondary"
            variant="contained"
            autoFocus
          // disabled={!currentConversation || (!isTyping && !isPaused)}
          >
            Yes, cancel
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ChatInputForm;