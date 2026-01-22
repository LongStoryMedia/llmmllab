import React, { useState, useRef } from 'react';
import { styled, FormControlLabel, Switch, useTheme, useMediaQuery, Box } from '@mui/material';
import { useChat } from '../../chat';
import {
  PromptInput,
  PromptInputProvider,
  PromptInputBody,
  PromptInputTextarea,
  PromptInputHeader,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTools,
  PromptInputButton,
  PromptInputActionMenu,
  PromptInputActionMenuTrigger,
  PromptInputActionMenuContent,
  PromptInputActionAddAttachments,
  PromptInputAttachments,
  PromptInputAttachment,
  PromptInputSpeechButton,
  type PromptInputMessage
} from '../ai-elements/prompt-input';
import { MessageContentType, MessageContentTypeValues } from '../../types/MessageContentType';
import { MessageRoleValues } from '../../types/MessageRole';
import { useConfigContext } from '../../context/ConfigContext';
import { getToken } from '../../api';
import { useAuth } from '../../auth';
import { listModelProfiles, updateModelProfile } from '../../api/model';
import { Image as ImageIcon, Stop as StopIcon } from '@mui/icons-material';
import { MessageContent } from '@/types';
import { type FileUIPart } from 'ai';
import { nanoid } from 'nanoid';
import { uint8ArrayToBase64 } from 'uint8array-extras';
import { StructuredModeToggle } from './StructuredModeToggle';
import { JsonSchemaEditor } from './JsonSchemaEditor';

const InputContainer = styled('div')(({ theme }) => ({
  padding: theme.spacing(1.5),
  backgroundColor: theme.palette.background.default,
  borderTop: `1px solid ${theme.palette.divider}`,
  boxShadow: theme.shadows[2],
  [theme.breakpoints.down('sm')]: {
    padding: theme.spacing(1)
  }
}));

const StyledPromptInput = styled(PromptInput)(({ theme }) => ({
  maxWidth: theme.breakpoints.values.md,
  margin: '0 auto',
  [theme.breakpoints.down('sm')]: {
    maxWidth: '100%'
  }
}));

type ProcessedFileAttachment = {
  type: MessageContentType;
  name: string;
  data?: string;
  format?: string;
  url?: string;
};

const processFileAttachment = async (file: FileUIPart): Promise<ProcessedFileAttachment | undefined> => {
  try {
    let fileData: string | undefined;
    let fileFormat: string | undefined;

    if (file.url.startsWith('blob:')) {
      const response = await fetch(file.url);
      const blob = await response.blob();
      const arrayBuffer = await blob.arrayBuffer();
      const uint8Array = new Uint8Array(arrayBuffer);
      fileData = uint8ArrayToBase64(uint8Array);
      fileFormat = file.mediaType || blob.type;
      fileFormat = file.mediaType || blob.type;
    } else if (file.url?.startsWith('data:')) {
      const match = /^data:(.*?);base64,(.*)$/.exec(file.url);
      if (match) {
        fileFormat = match[1];
        fileData = match[2];
      }
    }

    let contentType: MessageContentType = MessageContentTypeValues.FILE;
    if (file.mediaType?.startsWith('image/')) {
      contentType = MessageContentTypeValues.IMAGE;
    } else if (file.mediaType?.startsWith('audio/')) {
      contentType = MessageContentTypeValues.AUDIO;
    } else if (file.mediaType?.startsWith('video/')) {
      contentType = MessageContentTypeValues.VIDEO;
    }

    return {
      type: contentType,
      name: file.filename ?? nanoid(),
      data: fileData,
      format: fileFormat,
      url: file.url && !file.url.startsWith('blob:') ? file.url : undefined
    };
  } catch (error) {
    console.error('Failed to process file attachment:', error);
  }
};

const ChatInput = () => {
  const {
    currentConversation,
    isTyping,
    isLoading,
    error,
    sendMessage,
    cancelRequest,
    isStructuredMode,
    jsonSchema,
    setIsStructuredMode,
    setJsonSchema
  } = useChat();

  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const { config } = useConfigContext();
  const auth = useAuth();

  const [primaryProfileThink, setPrimaryProfileThink] = useState<boolean>(false);
  const [thinkLoading, setThinkLoading] = useState(false);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [showSchemaEditor, setShowSchemaEditor] = useState(false);
  const inProgress = isTyping || isLoading;

  // Load primary profile and its 'think' setting
  React.useEffect(() => {
    const loadPrimary = async () => {
      try {
        const primaryId = config?.model_profiles?.primary_profile_id || '';
        if (!primaryId) {
          return;
        }
        const profiles = await listModelProfiles(getToken(auth.user));
        const primary = profiles.find(p => p.id === primaryId);
        if (primary && primary.parameters) {
          setPrimaryProfileThink(!!primary.parameters.think);
        }
      } catch {
        // ignore errors silently for now
      }
    };
    loadPrimary();
  }, [config, auth.user]);

  const toggleThink = async () => {
    const primaryId = config?.model_profiles?.primary_profile_id || '';
    if (!primaryId) {
      return;
    }
    setThinkLoading(true);
    try {
      const profiles = await listModelProfiles(getToken(auth.user));
      const primary = profiles.find(p => p.id === primaryId);
      if (!primary) {
        return;
      }
      const updated = await updateModelProfile(getToken(auth.user), primaryId, {
        ...primary,
        parameters: {
          ...primary.parameters,
          think: !primary.parameters?.think
        }
      });
      setPrimaryProfileThink(!!updated.parameters?.think);
    } catch (error) {
      console.error('Failed toggling think on primary profile', error);
    } finally {
      setThinkLoading(false);
    }
  };

  const handleSubmit = async (message: PromptInputMessage) => {
    if (!currentConversation?.id || !message.text.trim()) {
      return;
    }

    // Start with text content
    const content: MessageContent[] = [
      {
        type: MessageContentTypeValues.TEXT,
        text: message.text.trim()
      }
    ];

    // Process file attachments
    if (message.files && message.files.length > 0) {
      const filePromises = message.files.map(processFileAttachment);
      const processedFiles = await Promise.all(filePromises);

      processedFiles.forEach(contentItem => {
        if (contentItem) {
          content.push(contentItem);
        }
      });
    }

    // Parse JSON schema if in structured mode
    let responseFormat: Record<string, unknown> | undefined;
    if (isStructuredMode) {
      try {
        responseFormat = JSON.parse(jsonSchema) as Record<string, unknown>;
      } catch (error) {
        console.error('Invalid JSON schema:', error);
        // You might want to show an error to the user here
        return;
      }
    }

    await sendMessage({
      role: MessageRoleValues.USER,
      content,
      conversation_id: currentConversation.id
    }, responseFormat);
  };

  const handleCancel = async () => {
    try {
      await cancelRequest();
    } catch (error) {
      console.error('Failed to cancel request:', error);
    }
  };

  return (
    <PromptInputProvider>
      <InputContainer>
        <StyledPromptInput
          onSubmit={handleSubmit}
          multiple
          globalDrop
        >
          <PromptInputHeader>
            <PromptInputAttachments>
              {(attachment) => (
                <PromptInputAttachment
                  key={attachment.id}
                  data={attachment}
                />
              )}
            </PromptInputAttachments>
          </PromptInputHeader>

          <PromptInputBody>
            <PromptInputTextarea
              ref={textareaRef}
              placeholder={
                !currentConversation
                  ? "No active conversation..."
                  : "Type your message..."
              }
              disabled={!currentConversation?.id}
              className="min-h-12 max-h-32"
            />
          </PromptInputBody>

          <PromptInputFooter>
            <PromptInputTools>
              {/* Attachments */}
              <PromptInputActionMenu>
                <PromptInputActionMenuTrigger />
                <PromptInputActionMenuContent>
                  <PromptInputActionAddAttachments label="Add files or images" />
                </PromptInputActionMenuContent>
              </PromptInputActionMenu>

              {/* Voice Input */}
              <PromptInputSpeechButton
                textareaRef={textareaRef}
                disabled={!currentConversation?.id}
              />

              {/* Image Generation (placeholder for future) */}
              <PromptInputButton
                disabled={!currentConversation?.id}
                title="Generate Image (Coming Soon)"
              >
                <ImageIcon className="text-muted-foreground" fontSize="small" />
              </PromptInputButton>

              {/* Think Toggle */}
              <FormControlLabel
                control={
                  <Switch
                    checked={primaryProfileThink}
                    onChange={toggleThink}
                    disabled={!config?.model_profiles?.primary_profile_id || thinkLoading}
                    size="small"
                  />
                }
                label="Think"
                sx={{
                  mr: 1,
                  '& .MuiFormControlLabel-label': {
                    fontSize: isMobile ? '0.75rem' : '0.875rem'
                  }
                }}
              />

              {/* Structured Mode Toggle */}
              <StructuredModeToggle
                isStructuredMode={isStructuredMode}
                onToggle={setIsStructuredMode}
                onEditSchema={() => {
                  setShowSchemaEditor(true);
                }}
              />
            </PromptInputTools>

            <Box className="flex items-center gap-2">
              {/* Cancel/Stop Button - only show when actively generating */}
              {inProgress ? (
                <PromptInputButton
                  onClick={handleCancel}
                  variant="outline"
                  size="sm"
                  title="Stop Generation"
                  className="text-warning"
                >
                  <StopIcon fontSize="small" />
                </PromptInputButton>
              ) : (
                <PromptInputSubmit
                  status={inProgress ? 'streaming' : error ? 'error' : 'ready'}
                  disabled={!currentConversation?.id}
                />
              )}
            </Box>
          </PromptInputFooter>
        </StyledPromptInput>

        {/* JSON Schema Editor Dialog */}
        <JsonSchemaEditor
          isOpen={showSchemaEditor}
          schema={jsonSchema}
          onOpenChange={setShowSchemaEditor}
          onSchemaChange={setJsonSchema}
          onSave={() => {
            // Schema is already saved via onSchemaChange
            console.log('JSON schema updated');
          }}
        />
      </InputContainer>
    </PromptInputProvider>
  );
};

export default ChatInput;