import React from 'react';
import { Box, Typography, Fade, useTheme } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import { useChat } from '../../chat';
import LoadingAnimation from '../Shared/LoadingAnimation';

interface SearchIndicatorProps {
  showLabel?: boolean;
}

const SearchIndicator: React.FC<SearchIndicatorProps> = ({ showLabel = true }) => {
  const { isSearching } = useChat();
  const theme = useTheme();

  if (!isSearching) {
    return null;
  }

  return (
    <Fade in={isSearching}>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          padding: showLabel ? theme.spacing(1, 2) : theme.spacing(1),
          borderRadius: theme.shape.borderRadius,
          backgroundColor: theme.palette.info.light,
          color: theme.palette.info.contrastText,
          boxShadow: 1,
          position: 'relative',
          my: 1
        }}
      >
        <LoadingAnimation size={20} />
        <SearchIcon fontSize="small" />
        {showLabel && (
          <Typography variant="body2" fontWeight="medium">
            Searching the web for information...
          </Typography>
        )}
      </Box>
    </Fade>
  );
};

export default SearchIndicator;