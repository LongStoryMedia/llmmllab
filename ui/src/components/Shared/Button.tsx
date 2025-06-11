import React from 'react';
import { Button as MuiButton, ButtonProps } from '@mui/material';

const Button: React.FC<ButtonProps> = ({ children, ...props }) => {
  return (
    <MuiButton {...props}>
      {children}
    </MuiButton>
  );
};

export default Button;