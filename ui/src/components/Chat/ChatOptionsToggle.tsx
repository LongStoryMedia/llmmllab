import React from 'react';
import { ToggleButtonGroup, Typography, useTheme, useMediaQuery } from '@mui/material';
import SummarizeIcon from '@mui/icons-material/Summarize';
import MemoryIcon from '@mui/icons-material/Memory';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import LanguageIcon from '@mui/icons-material/Language';
import ImageIcon from '@mui/icons-material/Image';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import { TooltipToggleButton } from './TooltipToggleButton';
import { useChat } from '../../chat';

interface ChatOptionsToggleProps {
  selectedOptions: string[];
  handleToggleChange: (event: React.MouseEvent<HTMLElement>, newOptions: string[]) => void;
}

const ChatOptionsToggle: React.FC<ChatOptionsToggleProps> = ({ selectedOptions, handleToggleChange }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const { isLoading, isTyping } = useChat(); // Assuming useChat provides these

  return (
    <ToggleButtonGroup
      value={selectedOptions}
      onChange={handleToggleChange}
      aria-label="chat options"
      sx={{ 
        display: 'flex',
        flexWrap: 'wrap',
        alignSelf: 'start', 
        mt: theme.spacing(1),
        gap: '4px',
        justifyContent: isMobile ? 'center' : 'flex-start',
        width: isMobile ? '100%' : 'auto'
      }}
    >
      <TooltipToggleButton 
        value="summarization" 
        aria-label="summarization" 
        disabled={isLoading || isTyping}
        color="secondary"
        tooltip="Automatically creates a summary of long conversations to help the model maintain context over time."
      >
        <SummarizeIcon sx={{ mr: isMobile ? 0.5 : 1, fontSize: 'small' }} />
        <Typography 
          variant="body2" 
          sx={{ 
            display: isMobile ? 'none' : 'inline',
            '@media (min-width:620px) and (max-width:800px)': {
              fontSize: '0.7rem'
            }
          }}
        >
          Summarization
        </Typography>
      </TooltipToggleButton>
        
      <TooltipToggleButton 
        value="retrieval" 
        aria-label="memory retrieval" 
        disabled={isLoading || isTyping}
        color="secondary"
        tooltip="Enables memory retrieval from previous conversations when relevant."
      >
        <MemoryIcon sx={{ mr: isMobile ? 0.5 : 1, fontSize: 'small' }} />
        <Typography 
          variant="body2" 
          sx={{ 
            display: isMobile ? 'none' : 'inline',
            '@media (min-width:620px) and (max-width:800px)': {
              fontSize: '0.7rem'
            }
          }}
        >
          Memory Retrieval
        </Typography>
      </TooltipToggleButton>
        
      <TooltipToggleButton 
        value="alwaysRetrieve" 
        aria-label="always enable memory retrieval" 
        disabled={isLoading || isTyping}
        color="secondary"
        tooltip="Forces retrieval of relevant past conversations for every message."
      >
        <AutoAwesomeIcon sx={{ mr: isMobile ? 0.5 : 1, fontSize: 'small' }} />
        <Typography 
          variant="body2" 
          sx={{ 
            display: isMobile ? 'none' : 'inline',
            '@media (min-width:620px) and (max-width:800px)': {
              fontSize: '0.7rem'
            }
          }}
        >
          Always Retrieve
        </Typography>
      </TooltipToggleButton>
        
      <TooltipToggleButton 
        value="webSearch" 
        aria-label="web search" 
        disabled={isLoading || isTyping}
        color="secondary"
        tooltip="Performs web searches to provide up-to-date information relevant to your query."
      >
        <LanguageIcon sx={{ mr: isMobile ? 0.5 : 1, fontSize: 'small' }} />
        <Typography 
          variant="body2" 
          sx={{ 
            display: isMobile ? 'none' : 'inline',
            '@media (min-width:620px) and (max-width:800px)': {
              fontSize: '0.7rem'
            }
          }}
        >
          Web Search
        </Typography>
      </TooltipToggleButton>
        
      <TooltipToggleButton 
        value="generateImage"
        aria-label="generate image"
        disabled={isLoading || isTyping}
        color="secondary"
        tooltip="Generate an AI image based on your prompt while also getting a text response."
      >
        <ImageIcon sx={{ mr: isMobile ? 0.5 : 1, fontSize: 'small' }} />
        <Typography 
          variant="body2" 
          sx={{ 
            display: isMobile ? 'none' : 'inline',
            '@media (min-width:620px) and (max-width:800px)': {
              fontSize: '0.7rem'
            }
          }}
        >
          Generate Image
        </Typography>
      </TooltipToggleButton>
      
      <TooltipToggleButton 
        value="autoPromptRefinement"
        aria-label="auto prompt refinement"
        disabled={isLoading || isTyping || !selectedOptions.includes('generateImage')}
        color="secondary"
        tooltip="Automatically refine your image prompts to improve generation quality."
      >
        <AutoFixHighIcon sx={{ mr: isMobile ? 0.5 : 1, fontSize: 'small' }} />
        <Typography 
          variant="body2" 
          sx={{ 
            display: isMobile ? 'none' : 'inline',
            '@media (min-width:620px) and (max-width:800px)': {
              fontSize: '0.7rem'
            }
          }}
        >
          Auto Refine
        </Typography>
      </TooltipToggleButton>
    </ToggleButtonGroup>
  );
};

export default ChatOptionsToggle;