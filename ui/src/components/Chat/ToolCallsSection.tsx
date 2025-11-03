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

// Structured type for parsed web search result
interface ParsedWebSearchResult {
  title: string;
  url: string;
  content: string;
  relevance: string;
}

interface ParsedWebSearchOutput {
  queryHeader: string;
  resultsHeader: string;
  results: ParsedWebSearchResult[];
  note?: string;
}

// Parse raw web_search output into structured object
const parseWebSearchOutput = (raw: string): ParsedWebSearchOutput | null => {
  if (!raw) {
    return null;
  }
  // Extract quoted content portion if present
  const match = raw.match(/^content="(.+?)"\s+name=/s);
  let content = match ? match[1] : raw;

  // Unescape sequences
  content = content
    .replace(/\\n/g, '\n')
    .replace(/\\"/g, '"')
    .replace(/\\'/g, "'")
    .replace(/\\xa0/g, ' ')
    .replace(/\\\\/g, '\\');

  // Split into lines
  const lines = content.split(/\n+/).map(l => l.trim()).filter(Boolean);
  if (lines.length < 3) {
    return null;
  }

  const queryHeader = lines[0].replace(/^🔍\s*/, '');
  let idx = 1;
  const resultsHeader = lines[idx];
  idx++;

  const results: ParsedWebSearchResult[] = [];
  let buffer: string[] = [];

  const flushBuffer = () => {
    if (buffer.length === 0) {
      return;
    }
    // Buffer pattern: Title line (with ** ... **), URL line starting with 📍 URL:, Content line starting 📄 Content:, relevance line starting ⭐ Relevance:
    const titleLine = buffer.find(l => /\*\*.*\*\*/.test(l)) || buffer[0];
    const urlLine = buffer.find(l => /^📍\s*URL:/i.test(l)) || '';
    const contentLine = buffer.find(l => /^📄\s*Content:/i.test(l)) || '';
    const relevanceLine = buffer.find(l => /^⭐\s*Relevance:/i.test(l)) || '';

    const extractAfter = (line: string, prefix: RegExp) => line.replace(prefix, '').trim();

    const cleanedTitle = titleLine
      .replace(/^\*\*/,'')
      .replace(/\*\*$/,'')
      .replace(/\*\*/g,'')
      .replace(/<strong>|<\/strong>/g,'')
      .trim();
    const url = extractAfter(urlLine, /^📍\s*URL:\s*/i);
    const contentText = extractAfter(contentLine, /^📄\s*Content:\s*/i);
    const relevance = extractAfter(relevanceLine, /^⭐\s*Relevance:\s*/i);

    if (cleanedTitle || url) {
      results.push({
        title: cleanedTitle,
        url,
        content: contentText,
        relevance
      });
    }
    buffer = [];
  };

  for (; idx < lines.length; idx++) {
    const line = lines[idx];
    if (line === '---') {
      flushBuffer();
      continue;
    }
    // Skip note line later
    if (/^💡/.test(line)) {
      flushBuffer();
      break; // Note begins; rest not results.
    }
    buffer.push(line);
  }
  flushBuffer();

  const noteLine = lines.find(l => /^💡/.test(l));
  const note = noteLine ? noteLine.replace(/^💡\s*/, '') : undefined;

  if (!results.length) {
    return null;
  }
  return { queryHeader, resultsHeader, results, note };
};

// Helper function to extract and format generic content (fallback)
const formatToolCallResult = (result: Record<string, unknown>) => {
  if (result.output && typeof result.output === 'string') {
    return result.output;
  }
  if (result.content && typeof result.content === 'string') {
    return result.content;
  }
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
                    const rawOutput = typeof toolCall.result_data.output === 'string' ? toolCall.result_data.output : undefined;
                    const parsed = rawOutput ? parseWebSearchOutput(rawOutput) : null;
                    if (parsed) {
                      return (
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                          <Typography variant="body2" sx={{ fontWeight: 600, color: 'primary.dark' }}>{parsed.queryHeader}</Typography>
                          <Typography variant="caption" sx={{ mb: 1 }}>{parsed.resultsHeader}</Typography>
                          {parsed.results.map((r, i) => (
                            <Paper key={i} variant="outlined" sx={{ p: 1, backgroundColor: 'rgba(255,255,255,0.04)' }}>
                              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>
                                {r.title}
                              </Typography>
                              {r.url && (
                                <Typography variant="body2" sx={{ mb: 0.5 }}>
                                  URL: <a href={r.url} target="_blank" rel="noopener noreferrer" style={{ color: '#80cbc4' }}>{r.url}</a>
                                </Typography>
                              )}
                              {r.content && (
                                <Typography variant="body2" sx={{ mb: 0.5 }}>
                                  {r.content}
                                </Typography>
                              )}
                              {r.relevance && (
                                <Typography variant="caption" color="text.secondary">
                                  Relevance: {r.relevance}
                                </Typography>
                              )}
                            </Paper>
                          ))}
                          {parsed.note && (
                            <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                              {parsed.note}
                            </Typography>
                          )}
                        </Box>
                      );
                    }
                    // Fallback to previous markdown / raw renderer
                    const formattedContent = formatToolCallResult(toolCall.result_data);
                    const isMarkdown = typeof formattedContent === 'string' &&
                      (formattedContent.includes('**') || formattedContent.includes('##'));
                    if (isMarkdown) {
                      return (
                        <ReactMarkdown>{formattedContent}</ReactMarkdown>
                      );
                    }
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