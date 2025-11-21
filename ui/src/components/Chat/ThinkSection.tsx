import React, { useState } from 'react';
import { Box, Button, useTheme, Paper, Tooltip } from '@mui/material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { sanitizeForLaTeX } from './utils';
import Stamp from '../Shared/Stamp';

interface ThinkSectionProps {
  think?: string;
  thinking?: boolean;
  searching?: boolean;
  text?: string;
  inProgress?: boolean;
}

const ThinkSection: React.FC<ThinkSectionProps> = ({ think }) => {
  const [showThink, setShowThink] = useState(false);
  const theme = useTheme();

  return (
    <Paper
      elevation={1}
      sx={{
        p: 2,
        display: 'flex',
        alignItems: 'center',
        backgroundColor: theme.palette.background.default,
        alignSelf: 'flex-start',
        borderRadius: 2,
        position: 'relative'
      }}
    >
      {think && (<>
        <Box sx={{ mb: 1, ml: 2 }}>
          <Tooltip title={showThink ? 'Hide Thoughts' : 'Show Thoughts'} arrow placement='right'>
            <Button
              size="small"
              variant="outlined"
              onClick={() => setShowThink((v) => !v)}
            >
              <Stamp />
            </Button>
          </Tooltip>
          {showThink && (
            <Box sx={{
              bgcolor: theme.palette.background.paper,
              border: `1px solid ${theme.palette.grey[300]}`,
              borderRadius: theme.shape.borderRadius,
              p: 1,
              mt: 0.5,
              fontSize: '0.9em',
              color: theme.palette.text.secondary,
              whiteSpace: 'pre-wrap'
            }}>
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeKatex]}
              >
                {sanitizeForLaTeX(think)}
              </ReactMarkdown>
            </Box>
          )}
        </Box>
      </>)}
    </Paper>
  );
};

export default ThinkSection;
