import { useState } from 'react';
import { Dialog, DialogTitle, DialogContent, DialogActions, Button, IconButton, Tabs, Tab, Box } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useAuth } from '../../auth';
import ProfileSettings from './ProfileSettings';
import MemorySettings from './MemorySettings';
import SummarizationSettings from './SummarizationSettings';
import WebSearchSettings from './WebSearchSettings';

interface SettingsDialogProps {
  open: boolean;
  onClose: () => void;
}

const SettingsDialog = ({ open, onClose }: SettingsDialogProps) => {
  const { logout } = useAuth();
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    event.preventDefault();
    setActiveTab(newValue);
  };
  const handleLogout = () => {
    logout();
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        Settings
        <IconButton edge="end" color="inherit" onClick={onClose} aria-label="close">
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={handleTabChange} aria-label="settings tabs">
            <Tab label="Profile" id="settings-tab-0" />
            <Tab label="Memory Retrieval" id="settings-tab-1" />
            <Tab label="Summarization" id="settings-tab-2" />
            <Tab label="Web Search" id="settings-tab-3" />
          </Tabs>
        </Box>
        <Box sx={{ pt: 2 }}>
          {activeTab === 0 && <ProfileSettings />}
          {activeTab === 1 && <MemorySettings />}
          {activeTab === 2 && <SummarizationSettings />}
          {activeTab === 3 && <WebSearchSettings />}
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleLogout} color="secondary">
          Logout
        </Button>
        <Button onClick={onClose}>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default SettingsDialog;