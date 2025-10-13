import React, { useState } from 'react';
import { useInView } from 'react-intersection-observer';
import { styled, Box, CircularProgress } from '@mui/material';

interface LazyImageProps {
  src: string;
  alt: string;
  maxWidth?: string | number;
  maxHeight?: string | number;
}

const ImageContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  minHeight: '100px',
  backgroundColor: theme.palette.background.paper,
  borderRadius: theme.shape.borderRadius,
  overflow: 'hidden'
}));

const LazyImage: React.FC<LazyImageProps> = ({
  src,
  alt,
  maxWidth = '100%',
  maxHeight = '400px'
}) => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [hasError, setHasError] = useState(false);
  const { ref, inView } = useInView({
    threshold: 0,
    triggerOnce: true, // Only load once when it comes into view
    rootMargin: '100px' // Start loading 100px before it's visible
  });

  const handleLoad = () => {
    setIsLoaded(true);
  };

  const handleError = () => {
    setHasError(true);
    setIsLoaded(true);
  };

  return (
    <ImageContainer ref={ref} sx={{ maxWidth, maxHeight }}>
      {!inView ? (
        <CircularProgress size={24} />
      ) : hasError ? (
        <Box color="text.secondary">Failed to load image</Box>
      ) : (
        <>
          {!isLoaded && <CircularProgress size={24} />}
          <img
            src={src}
            alt={alt}
            onLoad={handleLoad}
            onError={handleError}
            style={{
              maxWidth: '100%',
              maxHeight: maxHeight,
              objectFit: 'contain',
              display: isLoaded ? 'block' : 'none'
            }}
          />
        </>
      )}
    </ImageContainer>
  );
};

export default LazyImage;