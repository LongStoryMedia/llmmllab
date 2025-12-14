import React from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { useTheme } from '@mui/material/styles';
import {
  Box,
  Typography,
  Switch,
  FormControlLabel
} from '@mui/material';
import { JsonEditor } from 'json-edit-react';

interface JsonSchemaEditorProps {
  readonly isOpen: boolean;
  readonly onOpenChange: (open: boolean) => void;
  readonly schema: string;
  readonly onSchemaChange: (schema: string) => void;
  readonly onSave: () => void;
}

export const JsonSchemaEditor: React.FC<JsonSchemaEditorProps> = ({
  isOpen,
  onOpenChange,
  schema,
  onSchemaChange,
  onSave
}) => {
  const [localSchema, setLocalSchema] = React.useState<Record<string, unknown>>({});
  const [rawMode, setRawMode] = React.useState(false);
  const [rawText, setRawText] = React.useState(schema);
  const [error, setError] = React.useState<string | undefined>();
  const theme = useTheme();

  React.useEffect(() => {
    if (isOpen) {
      try {
        const parsed = JSON.parse(schema) as Record<string, unknown>;
        setLocalSchema(parsed);
        setRawText(JSON.stringify(parsed, null, 2));
        setError(undefined);
      } catch (err) {
        setLocalSchema({});
        setRawText(schema);
        setError('Invalid JSON in current schema: ' + (err as Error).message);
      }
    }
  }, [schema, isOpen]);

  const validateAndSave = () => {
    try {
      let finalSchema: string;

      if (rawMode) {
        // Validate raw text
        JSON.parse(rawText);
        finalSchema = rawText;
      } else {
        // Convert object to JSON string
        finalSchema = JSON.stringify(localSchema, null, 2);
      }

      onSchemaChange(finalSchema);
      onSave();
      onOpenChange(false);
    } catch (err) {
      setError('Invalid JSON schema: ' + (err as Error).message);
    }
  };

  const handleCancel = () => {
    // Reset to original schema
    try {
      const parsed = JSON.parse(schema) as Record<string, unknown>;
      setLocalSchema(parsed);
      setRawText(JSON.stringify(parsed, null, 2));
    } catch {
      setLocalSchema({});
      setRawText(schema);
    }
    setError(undefined);
    setRawMode(false);
    onOpenChange(false);
  };

  const handleModeToggle = () => {
    if (rawMode) {
      // Switching to visual mode - parse raw text to object
      try {
        const parsed = JSON.parse(rawText) as Record<string, unknown>;
        setLocalSchema(parsed);

        setError(undefined);
      } catch (err) {
        setError('Invalid JSON text: ' + (err as Error).message);
        return;
      }
    } else {
      // Switching to raw mode - convert object to JSON string
      try {
        setRawText(JSON.stringify(localSchema, null, 2));
        setError(undefined);
      } catch (err) {
        setError('Cannot convert to raw text: ' + (err as Error).message);
        return;
      }
    }

    setRawMode(!rawMode);
  };

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent
        className='sm:max-w-[800px] max-h-[90vh]'
        style={{
          backgroundColor: theme.palette.background.paper,
          maxWidth: '800px',
          width: '95vw',
          height: 'auto',
          maxHeight: '90vh'
        }}
      >
        <DialogHeader>
          <DialogTitle>JSON Schema Editor</DialogTitle>
          <DialogDescription>
            Define a JSON schema that the AI response must follow. Use visual mode for easier editing or raw mode for direct JSON input.
          </DialogDescription>
        </DialogHeader>

        <Box sx={{
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
          flex: 1,
          minHeight: 0
        }}>
          {/* Mode Toggle */}
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant='caption' sx={{ color: theme.palette.text.secondary }}>
              JSON Schema Editor
            </Typography>
            <FormControlLabel
              control={
                <Switch
                  checked={rawMode}
                  size='small'
                  onChange={handleModeToggle}
                />
              }
              label={
                <Typography variant='caption'>
                  Raw JSON Mode
                </Typography>
              }
            />
          </Box>

          {/* Editor Container */}
          <Box
            sx={{
              flex: 1,
              minHeight: '400px',
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: 2,
              overflow: 'hidden',
              backgroundColor: theme.palette.background.default
            }}
          >
            {rawMode ? (
              <textarea
                value={rawText}
                placeholder={`{
  "type": "object",
  "properties": {
    "response": {
      "type": "string",
      "description": "The main response"
    }
  },
  "required": ["response"]
}`}
                style={{
                  width: '100%',
                  height: '100%',
                  minHeight: '400px',
                  padding: '16px',
                  border: 'none',
                  outline: 'none',
                  fontFamily: '"Fira Code", "JetBrains Mono", "Consolas", monospace',
                  fontSize: '14px',
                  lineHeight: '1.5',
                  backgroundColor: 'transparent',
                  color: theme.palette.text.primary,
                  resize: 'none'
                }}
                onChange={e => {
                  setRawText(e.target.value);
                  setError(undefined);
                }}
              />
            ) : (
              <Box sx={{ p: 1, height: '100%', overflow: 'scroll', maxHeight: '50vh' }}>
                <JsonEditor
                  data={localSchema}
                  setData={(data: unknown) => {
                    setLocalSchema(data as Record<string, unknown>);
                  }}
                  showErrorMessages={false}
                  minWidth={350}
                  maxWidth='100%'
                  theme={{
                    styles: {
                      container: {
                        backgroundColor: theme.palette.background.paper,
                        fontFamily: '"Fira Code", "JetBrains Mono", "Consolas", monospace',
                        overflow: 'scroll'
                      },
                      collection: {},
                      collectionInner: {},
                      collectionElement: {},
                      dropZone: {},
                      property: theme.palette.text.primary,
                      bracket: { color: theme.palette.divider, fontWeight: 'bold' },
                      itemCount: { color: theme.palette.text.secondary, fontStyle: 'italic' },
                      string: theme.palette.success.main,
                      number: theme.palette.info.main,
                      boolean: theme.palette.success.main,
                      null: { color: theme.palette.error.main, fontVariant: 'small-caps', fontWeight: 'bold' },
                      input: [theme.palette.text.primary, { fontSize: '90%' }],
                      inputHighlight: theme.palette.info.light,
                      error: { fontSize: '0.8em', color: theme.palette.error.main, fontWeight: 'bold' },
                      iconCollection: theme.palette.text.secondary,
                      iconEdit: theme.palette.info.main,
                      iconDelete: theme.palette.error.main,
                      iconAdd: theme.palette.info.main,
                      iconCopy: theme.palette.info.main,
                      iconOk: theme.palette.success.main,
                      iconCancel: theme.palette.error.main
                    }
                  }}
                />
              </Box>
            )}
          </Box>

          {/* Error Display */}
          {error && (
            <Box
              sx={{
                p: 2,
                borderRadius: 1,
                backgroundColor: theme.palette.error.light,
                color: theme.palette.error.contrastText,
                fontSize: '0.875rem'
              }}
            >
              {error}
            </Box>
          )}
        </Box>

        <DialogFooter className='gap-2'>
          <Button
            variant='outline'
            onClick={handleCancel}
          >
            Cancel
          </Button>
          <Button
            onClick={validateAndSave}
          >
            Save Schema
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

