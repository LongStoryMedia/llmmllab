import React from 'react';
import { ToggleButton, Tooltip, useTheme, useMediaQuery } from '@mui/material';

interface TooltipToggleButtonProps {
  value: string;
  tooltip: string;
  disabled?: boolean;
  children: React.ReactNode;
  'aria-label': string;
  color?: 'primary' | 'secondary' | 'success' | 'error' | 'info' | 'warning' | 'standard';
}

// Component for wrapping each toggle button with a tooltip
export const TooltipToggleButton: React.FC<TooltipToggleButtonProps> = ({
  value,
  tooltip,
  disabled = false,
  color = "standard",
  children,
  'aria-label': ariaLabel
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  
  return (
    <Tooltip 
      title={tooltip} 
      arrow 
      placement="bottom" 
      enterDelay={500}
      leaveDelay={200}
    >
      <ToggleButton
        value={value}
        aria-label={ariaLabel}
        disabled={disabled}
        color={color}
        sx={{
          padding: isMobile ? '4px 6px' : '6px 10px',
          minWidth: isMobile ? 'auto' : '100px'
        }}
      >
        {children}
      </ToggleButton>
    </Tooltip>
  );
};