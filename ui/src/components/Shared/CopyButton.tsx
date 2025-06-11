import React, { useState } from 'react';
import { Button } from '@mui/material';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import CheckIcon from '@mui/icons-material/Check';

interface CopyButtonProps {
  text: string;
  size?: 'small' | 'medium' | 'large';
}

const CopyButton: React.FC<CopyButtonProps> = ({ text, size = 'small' }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <Button
      size={size}
      aria-label="Copy code"
      onClick={handleCopy}
      sx={{
        position: 'absolute',
        top: 8,
        right: 8,
        minWidth: 0,
        p: 0.5,
        zIndex: 1,
        background: 'rgba(255,255,255,0.7)',
        borderRadius: 1,
        '&:hover': { background: 'rgba(255,255,255,0.9)' }
      }}
    >
      {copied ? <CheckIcon fontSize="small" color="success" /> : <ContentCopyIcon fontSize="small" />}
    </Button>
  );
};

export default CopyButton;
