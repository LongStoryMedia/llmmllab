import { Box, Typography, useTheme } from '@mui/material';
import LoadingAnimation from './LoadingAnimation';

interface ControlLoaderProps {
  text?: string;
}

const ControlLoader = ({ text = '' }: ControlLoaderProps) => {
  const theme = useTheme();
  
  return (
    <Box 
      sx={{ 
        display: 'flex', 
        flexDirection: 'row',
        justifyContent: 'center', 
        alignItems: 'center', 
        p: theme.spacing(2)
      }}
    >
      <LoadingAnimation size={24} />
      {text && (
        <Typography 
          variant="body2" 
          color="text.secondary" 
          sx={{ mt: theme.spacing(1) }}
        >
          {text}
        </Typography>
      )}
    </Box>
  );
};

export default ControlLoader;