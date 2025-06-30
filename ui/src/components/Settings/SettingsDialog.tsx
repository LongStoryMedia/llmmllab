import { useState } from 'react';
import { Typography } from '@mui/material';
import SettingsTabs from './SettingsTabs';


const SettingsDialog = () => {
  const [activeTab, setActiveTab] = useState('profile');

  return (
    <>
      <Typography>{`Settings for ${activeTab}`}</Typography>
      <SettingsTabs onTabChange={tab => setActiveTab(tab)} />
    </>
  );
};

export default SettingsDialog;