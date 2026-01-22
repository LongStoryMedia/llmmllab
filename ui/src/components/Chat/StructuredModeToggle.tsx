import React from 'react';
import {
  Switch,
  FormControlLabel,
  Box,
  Tooltip,
  IconButton,
  Typography
} from '@mui/material';
import { Edit as EditIcon } from '@mui/icons-material';

interface StructuredModeToggleProps {
  readonly isStructuredMode: boolean;
  readonly onToggle: (enabled: boolean) => void;
  readonly onEditSchema: () => void;
}

export const StructuredModeToggle: React.FC<StructuredModeToggleProps> = ({
  isStructuredMode,
  onToggle,
  onEditSchema
}) => (
  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
    <FormControlLabel
      control={
        <Switch
          checked={isStructuredMode}
          size='small'
          onChange={e => {
            onToggle(e.target.checked);
          }}
        />
      }
      label={
        <Typography variant='caption' sx={{ fontSize: '0.75rem' }}>
          Structured
        </Typography>
      }
      sx={{
        margin: 0,
        '& .MuiFormControlLabel-label': {
          fontSize: '0.75rem'
        }
      }}
    />
    {isStructuredMode && (
      <Tooltip title='Edit JSON Schema'>
        <IconButton
          size='small'
          sx={{
            p: 0.5,
            color: 'primary.main',
            '&:hover': {
              backgroundColor: 'primary.light',
              color: 'primary.contrastText'
            }
          }}
          onClick={onEditSchema}
        >
          <EditIcon fontSize='small' />
        </IconButton>
      </Tooltip>
    )}
  </Box>
);
