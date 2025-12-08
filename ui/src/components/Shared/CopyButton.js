import { jsx as _jsx } from "react/jsx-runtime";
import { useState } from 'react';
import { Button } from '@mui/material';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import CheckIcon from '@mui/icons-material/Check';
const CopyButton = ({ text, size = 'small' }) => {
    const [copied, setCopied] = useState(false);
    const handleCopy = async () => {
        await navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
    };
    return (_jsx(Button, { size: size, "aria-label": "Copy code", onClick: handleCopy, sx: {
            position: 'absolute',
            top: 8,
            right: 8,
            minWidth: 0,
            p: 0.5,
            zIndex: 1,
            background: 'rgba(255,255,255,0.7)',
            borderRadius: 1,
            '&:hover': { background: 'rgba(255,255,255,0.9)' }
        }, children: copied ? _jsx(CheckIcon, { fontSize: "small", color: "success" }) : _jsx(ContentCopyIcon, { fontSize: "small" }) }));
};
export default CopyButton;
