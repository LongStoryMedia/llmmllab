import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  FormControlLabel,
  Checkbox,
  Grid,
  IconButton,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography,
  Alert,
  Chip,
  Stack,
  InputAdornment,
  Tooltip,
  Snackbar,
} from '@mui/material';
import {
  Delete as DeleteIcon,
  ContentCopy as CopyIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  Block as RevokeIcon,
} from '@mui/icons-material';
import { useAuth } from '../../auth';
import * as apiKeyService from '../../api/apiKey';
import { ApiKey, ApiKeyResponse, ApiKeyRequest } from '../../types';

const AVAILABLE_SCOPES = ['chat', 'generate', 'embed'];

interface CreateKeyDialogState {
  open: boolean;
  name: string;
  selectedScopes: string[];
  expiryDays: string;
  isLoading: boolean;
  createdKey?: ApiKeyResponse;
  showKey: boolean;
}

const ApiKeySettings: React.FC = () => {
  const auth = useAuth();
  const token = auth.user?.access_token;
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; type: 'success' | 'error' }>({
    open: false,
    message: '',
    type: 'success',
  });

  const [createDialog, setCreateDialog] = useState<CreateKeyDialogState>({
    open: false,
    name: '',
    selectedScopes: ['chat'],
    expiryDays: '',
    isLoading: false,
    showKey: false,
  });

  const [revokeDialog, setRevokeDialog] = useState<{ open: boolean; keyId: string; keyName: string }>({
    open: false,
    keyId: '',
    keyName: '',
  });

  const [deleteDialog, setDeleteDialog] = useState<{ open: boolean; keyId: string; keyName: string }>({
    open: false,
    keyId: '',
    keyName: '',
  });

  useEffect(() => {
    if (token) {
      loadKeys();
    }
  }, [token]);

  const loadKeys = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const fetchedKeys = await apiKeyService.listApiKeys(token!);
      setKeys(fetchedKeys);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load API keys';
      setError(errorMessage);
      showSnackbar(errorMessage, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const showSnackbar = (message: string, type: 'success' | 'error' = 'success') => {
    setSnackbar({ open: true, message, type });
  };

  const handleCreateClick = () => {
    setCreateDialog({
      open: true,
      name: '',
      selectedScopes: ['chat'],
      expiryDays: '',
      isLoading: false,
      showKey: false,
    });
  };

  const handleCreateClose = () => {
    setCreateDialog({ ...createDialog, open: false });
  };

  const handleScopeChange = (scope: string) => {
    setCreateDialog((prev) => ({
      ...prev,
      selectedScopes: prev.selectedScopes.includes(scope)
        ? prev.selectedScopes.filter((s) => s !== scope)
        : [...prev.selectedScopes, scope],
    }));
  };

  const handleCreateSubmit = async () => {
    if (!createDialog.name.trim()) {
      showSnackbar('Please enter a key name', 'error');
      return;
    }

    if (createDialog.selectedScopes.length === 0) {
      showSnackbar('Please select at least one scope', 'error');
      return;
    }

    try {
      setCreateDialog((prev) => ({ ...prev, isLoading: true }));

      const request: ApiKeyRequest = {
        name: createDialog.name,
        scopes: createDialog.selectedScopes,
        expires_in_days: createDialog.expiryDays ? parseInt(createDialog.expiryDays) : undefined,
      };

      const newKey = await apiKeyService.createApiKey(token!, request);
      setCreateDialog((prev) => ({
        ...prev,
        isLoading: false,
        createdKey: newKey,
        name: '',
        selectedScopes: ['chat'],
        expiryDays: '',
      }));

      showSnackbar(`API key "${newKey.name}" created successfully`, 'success');
      await loadKeys();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to create API key';
      setCreateDialog((prev) => ({ ...prev, isLoading: false }));
      showSnackbar(errorMessage, 'error');
    }
  };

  const handleCopyKey = () => {
    if (createDialog.createdKey?.key) {
      navigator.clipboard.writeText(createDialog.createdKey.key);
      showSnackbar('API key copied to clipboard', 'success');
    }
  };

  const handleRevokeClick = (key: ApiKey) => {
    setRevokeDialog({
      open: true,
      keyId: key.id,
      keyName: key.name,
    });
  };

  const handleRevokeConfirm = async () => {
    try {
      await apiKeyService.revokeApiKey(token!, revokeDialog.keyId);
      showSnackbar(`API key "${revokeDialog.keyName}" revoked successfully`, 'success');
      setRevokeDialog({ open: false, keyId: '', keyName: '' });
      await loadKeys();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to revoke API key';
      showSnackbar(errorMessage, 'error');
    }
  };

  const handleDeleteClick = (key: ApiKey) => {
    setDeleteDialog({
      open: true,
      keyId: key.id,
      keyName: key.name,
    });
  };

  const handleDeleteConfirm = async () => {
    try {
      await apiKeyService.deleteApiKey(token!, deleteDialog.keyId);
      showSnackbar(`API key "${deleteDialog.keyName}" deleted successfully`, 'success');
      setDeleteDialog({ open: false, keyId: '', keyName: '' });
      await loadKeys();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete API key';
      showSnackbar(errorMessage, 'error');
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const isKeyExpired = (expiresAt?: string) => {
    if (!expiresAt) return false;
    return new Date(expiresAt) < new Date();
  };

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', py: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h6" gutterBottom>
            API Keys
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Manage API keys for programmatic access to the API
          </Typography>
        </Box>
        <Button
          variant="contained"
          color="primary"
          onClick={handleCreateClick}
        >
          Create New Key
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Info Card */}
      <Card sx={{ mb: 3, backgroundColor: theme => theme.palette.background.paper }}>
        <CardContent>
          <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
            💡 Important
          </Typography>
          <Typography variant="body2" color="textSecondary">
            API keys grant full access to your account. Store them securely and never commit them to version control.
            Keys are shown only once when created - make sure to copy and save them immediately.
          </Typography>
        </CardContent>
      </Card>

      {/* Keys Table */}
      {keys.length === 0 ? (
        <Paper sx={{ p: 3, textAlign: 'center' }}>
          <Typography color="textSecondary">
            No API keys created yet. Create one to get started.
          </Typography>
        </Paper>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead sx={{ backgroundColor: '#fafafa' }}>
              <TableRow>
                <TableCell sx={{ fontWeight: 600 }}>Name</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Scopes</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Created</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Expires</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Status</TableCell>
                <TableCell sx={{ fontWeight: 600 }} align="right">
                  Actions
                </TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {keys.map((key) => (
                <TableRow key={key.id} hover>
                  {console.log('Rendering API key:', key)}
                  <TableCell>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      {key.name}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      {key.id.substring(0, 8)}...
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Stack direction="row" spacing={1}>
                      {key.scopes.map((scope) => (
                        <Chip key={scope} label={scope} size="small" variant="outlined" />
                      ))}
                    </Stack>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {formatDate(key.created_at)}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    {key.expires_at ? (
                      <Typography
                        variant="body2"
                        sx={{
                          color: isKeyExpired(key.expires_at) ? 'error.main' : 'inherit',
                        }}
                      >
                        {formatDate(key.expires_at)}
                      </Typography>
                    ) : (
                      <Typography variant="body2" color="textSecondary">
                        Never
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell>
                    {key.is_revoked ? (
                      <Chip label="Revoked" size="small" color="error" variant="outlined" />
                    ) : isKeyExpired(key.expires_at) ? (
                      <Chip label="Expired" size="small" color="warning" variant="outlined" />
                    ) : (
                      <Chip label="Active" size="small" color="success" variant="outlined" />
                    )}
                  </TableCell>
                  <TableCell align="right">
                    <Tooltip title="Revoke this key">
                      <IconButton
                        size="small"
                        onClick={() => handleRevokeClick(key)}
                        disabled={key.is_revoked}
                        color="warning"
                      >
                        <RevokeIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete this key permanently">
                      <IconButton
                        size="small"
                        onClick={() => handleDeleteClick(key)}
                        color="error"
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Create API Key Dialog */}
      <Dialog open={createDialog.open} onClose={handleCreateClose} maxWidth="sm" fullWidth>
        <DialogTitle>
          {createDialog.createdKey ? 'API Key Created' : 'Create New API Key'}
        </DialogTitle>
        <DialogContent sx={{ pt: 2 }}>
          {createDialog.createdKey ? (
            // Success screen showing the key
            <Box>
              <Alert severity="warning" sx={{ mb: 2 }}>
                ⚠️ This is the only time you'll see this key. Copy and store it securely!
              </Alert>

              <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
                API Key:
              </Typography>
              <TextField
                fullWidth
                value={createDialog.createdKey.key}
                type={createDialog.showKey ? 'text' : 'password'}
                InputProps={{
                  readOnly: true,
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        onClick={() =>
                          setCreateDialog((prev) => ({
                            ...prev,
                            showKey: !prev.showKey,
                          }))
                        }
                        edge="end"
                      >
                        {createDialog.showKey ? <VisibilityOffIcon /> : <VisibilityIcon />}
                      </IconButton>
                      <IconButton onClick={handleCopyKey} edge="end">
                        <CopyIcon />
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
                sx={{ mb: 2 }}
              />

              <Box sx={{ backgroundColor: '#f5f5f5', p: 2, borderRadius: 1 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                  Key Details
                </Typography>
                <Typography variant="body2">
                  <strong>Name:</strong> {createDialog.createdKey.name}
                </Typography>
                <Typography variant="body2">
                  <strong>Created:</strong> {formatDate(createDialog.createdKey.created_at)}
                </Typography>
                <Typography variant="body2">
                  <strong>Scopes:</strong> {createDialog.createdKey.scopes.join(', ')}
                </Typography>
                {createDialog.createdKey.expires_at && (
                  <Typography variant="body2">
                    <strong>Expires:</strong> {formatDate(createDialog.createdKey.expires_at)}
                  </Typography>
                )}
              </Box>
            </Box>
          ) : (
            // Creation form
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <TextField
                label="Key Name"
                placeholder="e.g., Production API Key"
                fullWidth
                value={createDialog.name}
                onChange={(e) =>
                  setCreateDialog((prev) => ({ ...prev, name: e.target.value }))
                }
                disabled={createDialog.isLoading}
              />

              <Box>
                <Typography variant="subtitle2" sx={{ mb: 1.5, fontWeight: 600 }}>
                  Scopes
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  {AVAILABLE_SCOPES.map((scope) => (
                    <FormControlLabel
                      key={scope}
                      control={
                        <Checkbox
                          checked={createDialog.selectedScopes.includes(scope)}
                          onChange={() => handleScopeChange(scope)}
                          disabled={createDialog.isLoading}
                        />
                      }
                      label={
                        <Box>
                          <Typography variant="body2" sx={{ fontWeight: 500 }}>
                            {scope.charAt(0).toUpperCase() + scope.slice(1)}
                          </Typography>
                          <Typography variant="caption" color="textSecondary">
                            {scope === 'chat' && 'Access chat completion endpoints'}
                            {scope === 'generate' && 'Access text generation endpoints'}
                            {scope === 'embed' && 'Access embedding endpoints'}
                          </Typography>
                        </Box>
                      }
                    />
                  ))}
                </Box>
              </Box>

              <TextField
                label="Expiration (days)"
                type="number"
                placeholder="Leave empty for no expiration"
                fullWidth
                value={createDialog.expiryDays}
                onChange={(e) =>
                  setCreateDialog((prev) => ({ ...prev, expiryDays: e.target.value }))
                }
                disabled={createDialog.isLoading}
                inputProps={{ min: 1, max: 36500 }}
                helperText="Maximum 100 years (36500 days)"
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          {createDialog.createdKey ? (
            <Button onClick={handleCreateClose} variant="contained">
              Close
            </Button>
          ) : (
            <>
              <Button onClick={handleCreateClose} disabled={createDialog.isLoading}>
                Cancel
              </Button>
              <Button
                onClick={handleCreateSubmit}
                variant="contained"
                disabled={createDialog.isLoading}
              >
                {createDialog.isLoading ? <CircularProgress size={20} /> : 'Create Key'}
              </Button>
            </>
          )}
        </DialogActions>
      </Dialog>

      {/* Revoke Confirmation Dialog */}
      <Dialog open={revokeDialog.open} onClose={() => setRevokeDialog({ ...revokeDialog, open: false })}>
        <DialogTitle>Revoke API Key?</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to revoke the API key <strong>"{revokeDialog.keyName}"</strong>?
          </Typography>
          <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
            Revoked keys cannot be used for authentication, but you can still view them.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRevokeDialog({ ...revokeDialog, open: false })}>
            Cancel
          </Button>
          <Button onClick={handleRevokeConfirm} color="warning" variant="contained">
            Revoke
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialog.open} onClose={() => setDeleteDialog({ ...deleteDialog, open: false })}>
        <DialogTitle>Delete API Key?</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to permanently delete the API key <strong>"{deleteDialog.keyName}"</strong>?
          </Typography>
          <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
            This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialog({ ...deleteDialog, open: false })}>
            Cancel
          </Button>
          <Button onClick={handleDeleteConfirm} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={() => setSnackbar({ ...snackbar, open: false })}
          severity={snackbar.type}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default ApiKeySettings;
