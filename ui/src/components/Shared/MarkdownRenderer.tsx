import React from 'react';
import { Box, Typography, Link, Table, TableBody, TableCell, TableHead, TableRow, useTheme } from '@mui/material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Prism as SyntaxHighlighter, SyntaxHighlighterProps } from 'react-syntax-highlighter';
import { vscDarkPlus, vs } from 'react-syntax-highlighter/dist/esm/styles/prism';
import CopyButton from './CopyButton';
import 'katex/dist/katex.min.css';
import useColorMode from '../../hooks/useColorMode';

interface MarkdownRendererProps {
  children: string;
  sanitizeForLaTeX?: (text: string) => string;
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ children, sanitizeForLaTeX }) => {
  const theme = useTheme();
  const content = sanitizeForLaTeX ? sanitizeForLaTeX(children) : children;
  const [mode] = useColorMode();

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={{
        code({node, className, children, ...props}) {
          const match = /language-(\w+)/.exec(className || '');
          const codeString = String(children).replace(/\n$/, '');
          if (className) {
            return (
              <Box sx={{ position: 'relative', mb: 2 }}>
                <CopyButton text={codeString} />
                <SyntaxHighlighter
                  style={mode === 'dark' ? vscDarkPlus : vs}
                  language={match?.[1] || 'text'}
                  wrapLines={true}
                  showLineNumbers={!!className}
                  {...props as SyntaxHighlighterProps}
                >
                  {codeString}
                </SyntaxHighlighter>
              </Box>
            );
          }
          return (
            <Typography
              component="span"
              sx={{
                fontFamily:'monospace',
                backgroundColor: theme.palette.background.paper,
                px: theme.spacing(0.5),
                py: theme.spacing(0.25),
                fontSize: '0.9em',
                color: theme.palette.text.primary
              }}
              {...props}
            >{codeString}</Typography>
          );
        },
        a({node, children, href, ...props}) {
          return (
            <Link 
              href={href} 
              target="_blank" 
              rel="noopener noreferrer"
              color="primary"
              underline="hover"
              {...props}
            >
              {children}
            </Link>
          );
        },
        table({node, children, ...props}) {
          return (
            <Box sx={{ overflowX: 'auto', my: theme.spacing(2) }}>
              <Table size="small" {...props}>
                {children}
              </Table>
            </Box>
          );
        },
        thead({node, children, ...props}) {
          return <TableHead {...props}>{children}</TableHead>;
        },
        tbody({node, children, ...props}) {
          return <TableBody {...props}>{children}</TableBody>;
        },
        tr({node, children, ...props}) {
          return <TableRow {...props}>{children}</TableRow>;
        },
        th({node, children, ...props}) {
          // @ts-expect-error ts(2322)
          return <TableCell component="th" sx={{ fontWeight: 'bold' }} {...props}>{children}</TableCell>;
        },
        td({node, children, ...props}) {
          // @ts-expect-error ts(2322)
          return <TableCell {...props}>{children}</TableCell>;
        },
        blockquote({node, children, ...props}) {
          return (
            <Box 
              component="blockquote"
              sx={{ 
                borderLeft: `${theme.spacing(0.5)} solid`,
                borderColor: theme.palette.grey[400],
                pl: theme.spacing(2),
                py: theme.spacing(0.5),
                my: theme.spacing(1),
                bgcolor: theme.palette.grey[100],
                borderRadius: theme.shape.borderRadius / 4
              }}
              {...props}
            >
              {children}
            </Box>
          );
        },
        img({node, alt, src, ...props}) {
          return (
            <Box sx={{ textAlign: 'center', my: theme.spacing(2) }}>
              <img 
                src={src} 
                alt={alt} 
                style={{ 
                  maxWidth: '100%', 
                  borderRadius: theme.shape.borderRadius 
                }} 
                {...props} 
              />
            </Box>
          );
        }
      }}
    >
      {content}
    </ReactMarkdown>
  );
};

export default MarkdownRenderer;
