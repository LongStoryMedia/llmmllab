import React from 'react';
import { Box, Typography, useTheme } from '@mui/material';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

interface ErrorMessageProps {
  message: string;
}

const ErrorMessage: React.FC<ErrorMessageProps> = ({ message }) => {
  const theme = useTheme();
  return (
    <Box sx={{ 
      display: 'flex', 
      alignItems: 'center', 
      color: theme.palette.error.main, 
      mb: theme.spacing(1),
      p: theme.spacing(1),
      bgcolor: theme.palette.error.light,
      borderRadius: theme.shape.borderRadius
    }}>
      <ErrorOutlineIcon fontSize="small" sx={{ mr: theme.spacing(1) }} />
      <Typography variant="caption">{message}</Typography>
    </Box>
  );
};

export default ErrorMessage;
