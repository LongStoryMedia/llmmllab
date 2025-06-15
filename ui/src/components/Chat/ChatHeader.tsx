import React from 'react';
import { AppBar, Toolbar, Typography, IconButton, useTheme, Button, TextField, Dialog, DialogTitle, DialogContent, DialogActions } from '@mui/material';
import { Menu as MenuIcon, Pause as PauseIcon, PlayArrow as PlayArrowIcon } from '@mui/icons-material';

interface ChatHeaderProps {
  title: string;
  onMenuClick: React.MouseEventHandler<HTMLButtonElement>;
  isPaused?: boolean;
  isTyping?: boolean;
  onPauseClick?: () => void;
  onResumeClick?: (corrections: string) => void;
}

const ChatHeader = ({ 
  title, 
  onMenuClick, 
  isPaused = false, 
  isTyping = false,
  onPauseClick,
  onResumeClick
}: ChatHeaderProps) => {
  const theme = useTheme();
  const [correctionDialogOpen, setCorrectionDialogOpen] = React.useState(false);
  const [corrections, setCorrections] = React.useState('');
  
  const handleResumeClick = () => {
    setCorrectionDialogOpen(true);
  };
  
  const handleDialogClose = () => {
    setCorrectionDialogOpen(false);
  };
  
  const handleSubmitCorrections = () => {
    if (onResumeClick) {
      onResumeClick(corrections);
    }
    setCorrections('');
    setCorrectionDialogOpen(false);
  };
  
  return (
    <>
      <AppBar position="static">
        <Toolbar>
          <IconButton edge="start" color="inherit" onClick={onMenuClick}>
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" sx={{ flexGrow: 1, color: theme.palette.common.white }}>
            {title}
          </Typography>
          
          {/* Pause/Resume buttons */}
          {isTyping && !isPaused && onPauseClick && (
            <Button 
              startIcon={<PauseIcon />}
              color="inherit"
              onClick={onPauseClick}
              size="small"
              variant="outlined"
              sx={{ mr: 1 }}
            >
              Pause
            </Button>
          )}
          
          {isPaused && onResumeClick && (
            <Button
              startIcon={<PlayArrowIcon />}
              color="inherit"
              onClick={handleResumeClick}
              size="small"
              variant="outlined"
              sx={{ mr: 1 }}
            >
              Resume with Context
            </Button>
          )}
        </Toolbar>
      </AppBar>
      
      {/* Correction Dialog */}
      <Dialog open={correctionDialogOpen} onClose={handleDialogClose} maxWidth="md" fullWidth>
        <DialogTitle>Add Corrections or Additional Context</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
            Add clarifications, corrections, or additional context to help the AI continue with more accurate information.
          </Typography>
          <TextField
            autoFocus
            multiline
            rows={6}
            fullWidth
            variant="outlined"
            value={corrections}
            onChange={(e) => setCorrections(e.target.value)}
            placeholder="What would you like to correct or clarify?"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDialogClose}>Cancel</Button>
          <Button 
            onClick={handleSubmitCorrections} 
            variant="contained" 
            color="primary"
            disabled={!corrections.trim()}
          >
            Continue
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default ChatHeader;