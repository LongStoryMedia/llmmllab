import React from 'react';
import { Button as MuiButton, ButtonProps as MuiButtonProps, Typography } from '@mui/material';
import LoadingAnimation from './LoadingAnimation';
import Stamp from './Stamp';

interface ButtonProps extends MuiButtonProps {
  label: string;
  withStamp?: boolean;
}

const Button: React.FC<ButtonProps> = props => {
  return (
    <MuiButton
      size="small"
      variant="outlined"
      loadingIndicator={<LoadingAnimation size={50} sx={{ mr: 1 }} />}
      {...props}
    >
      <Typography sx={{ mr: 1 }} >{props.label}</Typography>
      {props.withStamp && <Stamp />}
    </MuiButton>
  );
};

export default Button;