import React from 'react';
import { Box, LinearProgress, Typography, Paper, Grid } from '@mui/material';
import { SocketStageType } from '../../types/SocketStageType';

// Interface for storing progress information
export interface StageProgress {
  id: string;
  stage: SocketStageType;
  progress: number;
  timestamp: Date;
}

interface StageProgressBarsProps {
  activeStages: StageProgress[];
}

// Helper function to get a friendly display name for each stage
export const getStageFriendlyName = (stage: SocketStageType): string => {
  // Replace underscores with spaces and capitalize each word
  return stage
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

const StageProgressBars: React.FC<StageProgressBarsProps> = ({ activeStages }) => {
  if (activeStages.length === 0) {
    return null;
  }

  return (
    <Box sx={{ 
      position: 'fixed', 
      bottom: 80, 
      left: 24, 
      width: 300, 
      zIndex: (theme) => theme.zIndex.drawer - 1 
    }}>
      <Paper elevation={3} sx={{ p: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Active Processes
        </Typography>
        <Box sx={{ mt: 1 }}>
          {activeStages.map((stageProgress) => (
            <Box key={stageProgress.id} sx={{ mb: 2 }}>
              <Grid container justifyContent="space-between" alignItems="center" sx={{ mb: 0.5 }}>
                <Grid>
                  <Typography variant="body2">{getStageFriendlyName(stageProgress.stage)}</Typography>
                </Grid>
                <Grid>
                  <Typography variant="caption">{Math.round(stageProgress.progress)}%</Typography>
                </Grid>
              </Grid>
              <LinearProgress 
                variant="determinate" 
                value={stageProgress.progress} 
                sx={{ height: 8, borderRadius: 1 }}
              />
            </Box>
          ))}
        </Box>
      </Paper>
    </Box>
  );
};

export default StageProgressBars;