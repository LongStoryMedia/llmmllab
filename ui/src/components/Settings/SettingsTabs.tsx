import React, { useState } from 'react';
import { Grid, Typography, useTheme, Tabs, Tab, Box, Paper, Alert, CircularProgress, Button } from '@mui/material';
import { useConfig } from '../../hooks/useConfig';
import ProfileSettings from './ProfileSettings';
import ModelSettings from './ModelSettings';
import SummarizationSettings from './SummarizationSettings';
import MemorySettings from './MemorySettings';
import WebSearchSettings from './WebSearchSettings';
import SecuritySettings from './SecuritySettings';
import RefinementSettings from './RefinementSettings';
import ImageGenerationSettings from './ImageGenerationSettings';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 0 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `settings-tab-${index}`,
    'aria-controls': `settings-tabpanel-${index}`
  };
}

const tabRoutes = ["profile", "models", "summarization", "retrieval", "websearch", "security", "refinement", "image-generation"];

const SettingsTabs = ({onTabChange}: {onTabChange: (tab: string) => void}) => {
  const theme = useTheme();
  const [tabValue, setTabValue] = useState(tabRoutes.indexOf("profile"));
  const { isLoading, error, fetchConfig } = useConfig();

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
    onTabChange(tabRoutes[newValue]);
  };
  
  return (
    <Grid container spacing={3} sx={{ padding: theme.spacing(2.5) }}>
      <Grid size={12}>
        <Typography variant="h4" gutterBottom>
          Settings
        </Typography>
      </Grid>
      
      {error && (
        <Grid size={12}>
          <Alert 
            severity="error" 
            action={
              <Button color="inherit" size="small" onClick={fetchConfig}>
                Retry
              </Button>
            }
          >
            Error loading configuration: {error.message}
          </Alert>
        </Grid>
      )}
      
      <Grid size={12}>
        {isLoading ? (
          <Paper sx={{ padding: 3, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <CircularProgress />
            <Typography sx={{ ml: 2 }}>Loading settings...</Typography>
          </Paper>
        ) : (
          <Paper sx={{ width: '100%' }}>
            <Tabs
              value={tabValue}
              onChange={handleTabChange}
              indicatorColor="primary"
              textColor="primary"
              variant="scrollable"
              scrollButtons="auto"
              aria-label="settings tabs"
            >
              <Tab label="User Profile" {...a11yProps(0)} />
              <Tab label="Models" {...a11yProps(1)} />
              <Tab label="Summarization" {...a11yProps(2)} />
              <Tab label="Memory Retrieval" {...a11yProps(3)} />
              <Tab label="Web Search Settings" {...a11yProps(4)} />
              <Tab label="Security" {...a11yProps(5)} />
              <Tab label="Refinement" {...a11yProps(6)} />
              <Tab label="Image Generation" {...a11yProps(7)} />
            </Tabs>
            
            <Box sx={{ p: 2 }}>
              <TabPanel value={tabValue} index={0}>
                <ProfileSettings />
              </TabPanel>
              <TabPanel value={tabValue} index={1}>
                <ModelSettings />
              </TabPanel>
              <TabPanel value={tabValue} index={2}>
                <SummarizationSettings />
              </TabPanel>
              <TabPanel value={tabValue} index={3}>
                <MemorySettings />
              </TabPanel>
              <TabPanel value={tabValue} index={4}>
                <WebSearchSettings />
              </TabPanel>
              <TabPanel value={tabValue} index={5}>
                <SecuritySettings />
              </TabPanel>
              <TabPanel value={tabValue} index={6}>
                <RefinementSettings />
              </TabPanel>
              <TabPanel value={tabValue} index={7}>
                <ImageGenerationSettings />
              </TabPanel>
            </Box>
          </Paper>
        )}
      </Grid>
    </Grid>
  );
};

export default SettingsTabs;