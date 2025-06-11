import React from 'react';
import { Box, FormControl, InputLabel, Select, MenuItem, Typography, SelectChangeEvent } from '@mui/material';
import { useChat } from '../../chat';
import ControlLoader from '../Shared/ControlLoader';


interface ModelCardProps {
  onSelect: (event: SelectChangeEvent) => void;
  name: string;
}

const ModelSelector: React.FC<ModelCardProps> = ({ onSelect, name }) => {
  const { models, isLoading } = useChat();

  return (
    isLoading ? 
      <ControlLoader text='Loading models...' /> :
      <Box sx={{ mb: 2, p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Select a Model
        </Typography>
        <FormControl fullWidth>
          <InputLabel id="model-select-label">Model</InputLabel>
          <Select
            labelId="model-select-label"
            id="model-select"
            value={name}
            onChange={onSelect}
            label="Model"
          >
            {models && models?.map((model) => (
              <MenuItem key={model.name} value={model.name}>
                {model.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>
  );
};

export default ModelSelector;