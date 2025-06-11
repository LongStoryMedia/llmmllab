import React from 'react';
import { Card, CardContent, Typography, Button, useTheme } from '@mui/material';

interface ModelCardProps {
  modelName: string;
  modelDescription: string;
  onSelect: (model: string) => void;
}

const ModelCard: React.FC<ModelCardProps> = ({ modelName, modelDescription, onSelect }) => {
  const theme = useTheme();
  
  return (
    <Card 
      variant="outlined" 
      sx={{ 
        margin: theme.spacing(1.25), 
        cursor: 'pointer' 
      }}
    >
      <CardContent>
        <Typography variant="h5" component="div">
          {modelName}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {modelDescription}
        </Typography>
        <Button 
          variant="contained" 
          color="primary" 
          onClick={() => onSelect(modelName)}
          sx={{ mt: theme.spacing(1) }}
        >
          Select Model
        </Button>
      </CardContent>
    </Card>
  );
};

export default ModelCard;