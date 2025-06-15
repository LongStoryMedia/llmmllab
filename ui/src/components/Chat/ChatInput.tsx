import React, { useState } from 'react';
import { Box, Typography, useTheme, IconButton, Tooltip, FormControl, Input, ToggleButton, ToggleButtonGroup, useMediaQuery } from '@mui/material';
import ImageIcon from '@mui/icons-material/Image';
import AddIcon from '@mui/icons-material/Add';
import SendIcon from '@mui/icons-material/Send';
import SummarizeIcon from '@mui/icons-material/Summarize';
import MemoryIcon from '@mui/icons-material/Memory';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import LanguageIcon from '@mui/icons-material/Language';
import { useChat } from '../../chat';
import { useConfigContext } from '../../context/ConfigContext';
import BackgroundActionIndicator from './BackgroundActionIndicator';

type TooltipToggleButtonProps = {
  value: string;
  tooltip: string;
  disabled?: boolean;
  children: React.ReactNode;
  'aria-label': string;
} & React.ComponentProps<typeof ToggleButton>;

// Component for wrapping each toggle button with a tooltip
const TooltipToggleButton: React.FC<TooltipToggleButtonProps> = ({
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

const ChatInput = () => {
  const [input, setInput] = useState('');
  const { sendMessage, isTyping, currentConversation, isWorkingInBackground, setBackgroundAction, backgroundAction } = useChat();
  const theme = useTheme();
  const { config, updatePartialConfig, isLoading } = useConfigContext();
  const alwaysRetrieve = config?.memory?.always_retrieve || false;
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  // Check if there's an active conversation
  const hasConversation = !!currentConversation?.id;

  // Get toggle states from config (default to false if not loaded)
  const summarizationEnabled = config?.summarization?.enabled !== false;
  const retrievalEnabled = config?.memory?.enabled !== false;
  const webSearchEnabled = config?.web_search?.enabled || false;
  
  // Create array of selected options for ToggleButtonGroup
  const [selectedOptions, setSelectedOptions] = useState<string[]>(() => {
    const initialOptions: string[] = [];
    if (summarizationEnabled) {
      initialOptions.push('summarization');
    }
    if (retrievalEnabled) {
      initialOptions.push('retrieval');
    }
    if (alwaysRetrieve) {
      initialOptions.push('alwaysRetrieve');
    }
    if (webSearchEnabled) {
      initialOptions.push('webSearch');
    }
    return initialOptions;
  });
  
  // Handler for all toggle buttons in the group
  const handleToggleChange = async (event: React.MouseEvent<HTMLElement>, newOptions: string[]) => {
    event.preventDefault();
    const wasSelected = (option: string) => selectedOptions.includes(option);
    const isSelected = (option: string) => newOptions.includes(option);
    
    // Update options array
    setSelectedOptions(newOptions);
    
    // Handle summarization change
    if (wasSelected('summarization') !== isSelected('summarization')) {
      await updatePartialConfig('summarization', {
        ...config!.summarization!,
        enabled: isSelected('summarization')
      });
    }
    
    // Handle retrieval change
    if (wasSelected('retrieval') !== isSelected('retrieval')) {
      await updatePartialConfig('memory', {
        ...config!.memory!,
        enabled: isSelected('retrieval')
      });
    }
    
    // Handle always retrieve change
    if (wasSelected('alwaysRetrieve') !== isSelected('alwaysRetrieve')) {
      await updatePartialConfig('memory', {
        ...config!.memory!,
        always_retrieve: isSelected('alwaysRetrieve'),
        enabled: config?.memory?.enabled ?? false
      });
    }
    
    // Handle web search change
    if (wasSelected('webSearch') !== isSelected('webSearch')) {
      await updatePartialConfig('web_search', {
        ...config!.web_search!,
        enabled: isSelected('webSearch')
      });
    }

    // Handle image generation toggle
    if (wasSelected('generateImage') !== isSelected('generateImage')) {
      if (isSelected('generateImage')) {
        setBackgroundAction('Generating Image...');
      } else {
        setBackgroundAction(null);
      }
    }
  };

  // Check if image generation is selected
  const generateImage = selectedOptions.includes('generateImage');

  const handleKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    const trimmedInput = input.trim();
    if (trimmedInput && !isTyping && hasConversation) {
      // Add a flag to the message metadata if image generation is requested
      sendMessage({ 
        content: trimmedInput, 
        conversation_id: currentConversation.id!,
        metadata: {
          generate_image: generateImage,
          is_continuation: false // should be false for new messages
        }
      });
      setInput('');
      
      // Reset the image toggle after sending if needed
      if (generateImage) {
        setSelectedOptions(prev => prev.filter(option => option !== 'generateImage'));
      }
    }
  };

  return (
    <Box 
      sx={{ 
        display: 'flex', 
        flexDirection: 'column',
        p: theme.spacing(isMobile ? 1 : 2),
        borderTop: `${theme.spacing(0.125)} solid`,
        borderColor: theme.palette.divider
      }}
    >
      {isWorkingInBackground && <BackgroundActionIndicator label={backgroundAction ?? ''} />}      
      {!hasConversation && (
        <Typography 
          variant="body2" 
          color="text.secondary" 
          sx={{ mb: theme.spacing(1), textAlign: 'center' }}
        >
          Start a new conversation to begin chatting
        </Typography>
      )}
      <FormControl sx={{ 
        display: 'flex', 
        alignItems: 'center',
        width: '100%',
        flexDirection: 'column', 
        backgroundColor: theme.palette.background.default,
        borderRadius: '28px',
        padding: isMobile ? '6px 12px' : '8px 16px',
        boxShadow: '0px 2px 4px rgba(0, 0, 0, 0.25)'
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
          <IconButton sx={{ color: theme.palette.text.secondary, padding: isMobile ? '4px' : '8px' }}>
            <AddIcon fontSize={isMobile ? 'small' : 'medium'} />
          </IconButton>
          <Input
            fullWidth
            placeholder={generateImage 
              ? "Enter a prompt to generate an image..." 
              : hasConversation 
                ? "Type your message..." 
                : "No active conversation..."
            }
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            multiline
            maxRows={4}
            disabled={isTyping || !hasConversation}
            sx={{
              flexGrow: 1,
              fontSize: isMobile ? '0.875rem' : '1rem'
            }}
          >
          </Input>
          <Tooltip title={generateImage ? "Generate Image" : "Send Message"} arrow>
            <IconButton 
              sx={{ 
                color: theme.palette.text.secondary, 
                alignContent:'end',
                padding: isMobile ? '4px' : '8px'
              }}
              onClick={handleSend} 
              type='submit'
              color={generateImage ? "secondary" : "primary"}
            >
              {generateImage ? <ImageIcon fontSize={isMobile ? 'small' : 'medium'} /> : <SendIcon fontSize={isMobile ? 'small' : 'medium'} />}
            </IconButton>
          </Tooltip>
        </Box>
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
        </ToggleButtonGroup>
      </FormControl>
    </Box>
  );
};

export default ChatInput;