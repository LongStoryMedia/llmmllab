import React from 'react';
import { Box, Typography, Paper, Accordion, AccordionSummary, AccordionDetails } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { styled } from '@mui/material/styles';
import ReactMarkdown from 'react-markdown';

const ToolCallContainer = styled(Paper)(({ theme }) => ({
  marginBottom: theme.spacing(1),
  backgroundColor: theme.palette.action.hover,
  border: `1px solid ${theme.palette.divider}`
}));

const ToolCallHeader = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
  padding: theme.spacing(1)
}));

const ToolNameChip = styled(Typography)(({ theme }) => ({
  backgroundColor: theme.palette.primary.main,
  color: theme.palette.primary.contrastText,
  padding: theme.spacing(0.5, 1),
  borderRadius: theme.shape.borderRadius,
  fontSize: '0.75rem',
  fontWeight: 600
}));

const StatusChip = styled(Typography, {
  shouldForwardProp: (prop) => prop !== 'success'
})<{ success: boolean }>(({ theme, success }) => ({
  backgroundColor: success ? theme.palette.success.main : theme.palette.error.main,
  color: success ? theme.palette.success.contrastText : theme.palette.error.contrastText,
  padding: theme.spacing(0.25, 0.75),
  borderRadius: theme.shape.borderRadius,
  fontSize: '0.6rem',
  fontWeight: 500
}));

interface ToolCall {
  tool_name?: string;
  name?: string;
  success?: boolean;
  execution_time_ms?: number;
  args?: Record<string, unknown>;
  result_data?: Record<string, unknown>;
  error_message?: string;
}

interface ToolCallsSectionProps {
  toolCalls: ToolCall[] | null;
  isTyping?: boolean;
}

// Helper function to extract and format content from tool call results
const formatToolCallResult = (result: Record<string, unknown>) => {
  // Handle web search results with nested content in output field
  if (result.output && typeof result.output === 'string') {
    // Parse the output string which may contain escaped content
    let content = result.output;
    
    // Check if it starts with content=" pattern (JSON-like string)
    const contentMatch = content.match(/^content="(.+?)"\s+name=/s);
    if (contentMatch) {
      // Extract the actual content and unescape it
      content = contentMatch[1]
        .replace(/\\n/g, '\n')  // Convert literal \n to actual newlines
        .replace(/\\"/g, '"')   // Convert literal \" to actual quotes
        .replace(/\\'/g, "'")   // Convert literal \' to actual apostrophes
        .replace(/\\\\/g, '\\') // Convert literal \\ to actual backslashes
        .replace(/\\xa0/g, ' '); // Convert non-breaking space
    } else {
      // Also handle cases where the entire string might be escaped
      content = content
        .replace(/\\n/g, '\n')
        .replace(/\\"/g, '"')
        .replace(/\\'/g, "'")
        .replace(/\\\\/g, '\\')
        .replace(/\\xa0/g, ' ');
    }
    
    return content;
  }
  
  // Handle direct content field
  if (result.content && typeof result.content === 'string') {
    return result.content;
  }
  
  // Handle other structured results
  return JSON.stringify(result, null, 2);
};

const ToolCallsSection: React.FC<ToolCallsSectionProps> = ({ toolCalls, isTyping = false }) => {
  if (!toolCalls || toolCalls.length === 0) {
    return null;
  }

  return (
    <Box sx={{ mt: 1, mb: 2 }}>
      <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
        Tool Calls {isTyping && '(executing...)'}
      </Typography>

      {toolCalls.map((toolCall: ToolCall, index: number) => (
        <ToolCallContainer key={index} elevation={0}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <ToolCallHeader>
                <ToolNameChip variant="body2">
                  {toolCall.tool_name || toolCall.name || 'Unknown Tool'}
                </ToolNameChip>

                {toolCall.success !== undefined && (
                  <StatusChip success={Boolean(toolCall.success)} variant="body2">
                    {toolCall.success ? 'Success' : 'Failed'}
                  </StatusChip>
                )}

                {toolCall.execution_time_ms && (
                  <Typography variant="caption" color="text.secondary">
                    {toolCall.execution_time_ms}ms
                  </Typography>
                )}
              </ToolCallHeader>
            </AccordionSummary>

            <AccordionDetails>
              {/* Arguments */}
              {toolCall.args && Object.keys(toolCall.args).length > 0 && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    Arguments:
                  </Typography>
                  <Box component="pre" sx={{
                    fontSize: '0.75rem',
                    backgroundColor: 'background.default',
                    padding: 1,
                    borderRadius: 1,
                    overflow: 'auto',
                    maxHeight: 200
                  }}>
                    {JSON.stringify(toolCall.args, null, 2)}
                  </Box>
                </Box>
              )}

              {/* Results */}
              {toolCall.result_data && Object.keys(toolCall.result_data).length > 0 && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    Results:
                  </Typography>
                  {(() => {
                    const formattedContent = formatToolCallResult(toolCall.result_data);
                    
                    // Check if content looks like markdown (contains ** or ## or emoji indicators)
                    const isMarkdown = typeof formattedContent === 'string' && 
                      (formattedContent.includes('**') || formattedContent.includes('##') || 
                       formattedContent.includes('📍') || formattedContent.includes('⭐') || 
                       formattedContent.includes('🔍') || formattedContent.includes('💡'));
                    
                    if (isMarkdown) {
                      return (
                        <Box sx={{
                          '& h1, & h2, & h3': { 
                            marginTop: 1, 
                            marginBottom: 0.5,
                            fontSize: '1rem',
                            fontWeight: 600
                          },
                          '& p': { 
                            marginBottom: 0.5,
                            fontSize: '0.875rem',
                            lineHeight: 1.4
                          },
                          '& strong': { 
                            fontWeight: 600,
                            color: 'primary.main'
                          },
                          '& hr': {
                            margin: '8px 0',
                            border: 'none',
                            borderTop: '1px solid',
                            borderColor: 'divider'
                          },
                          '& a': {
                            color: 'primary.light',
                            textDecoration: 'none',
                            '&:hover': {
                              textDecoration: 'underline'
                            }
                          },
                          '& code': {
                            backgroundColor: 'rgba(255, 255, 255, 0.1)',
                            padding: '2px 4px',
                            borderRadius: '3px',
                            fontSize: '0.8rem'
                          },
                          maxHeight: 400,
                          overflow: 'auto',
                          fontSize: '0.875rem'
                        }}>
                          <ReactMarkdown
                            components={{
                              a: ({ href, children, ...props }) => (
                                <a 
                                  href={href} 
                                  target="_blank" 
                                  rel="noopener noreferrer" 
                                  {...props}
                                >
                                  {children}
                                </a>
                              )
                            }}
                          >
                            {formattedContent}
                          </ReactMarkdown>
                        </Box>
                      );
                    } else {
                      return (
                        <Box component="pre" sx={{
                          fontSize: '0.75rem',
                          backgroundColor: 'background.default',
                          padding: 1,
                          borderRadius: 1,
                          overflow: 'auto',
                          maxHeight: 200
                        }}>
                          {formattedContent}
                        </Box>
                      );
                    }
                  })()}
                </Box>
              )}

              {/* Error message if failed */}
              {!toolCall.success && toolCall.error_message && (
                <Box>
                  <Typography variant="subtitle2" color="error" gutterBottom>
                    Error:
                  </Typography>
                  <Typography variant="body2" color="error">
                    {String(toolCall.error_message)}
                  </Typography>
                </Box>
              )}
            </AccordionDetails>
          </Accordion>
        </ToolCallContainer>
      ))}
    </Box>
  );
};

export default ToolCallsSection;