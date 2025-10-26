import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  IconButton,
  Collapse,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Chip,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Paper,
  Divider
} from '@mui/material';
import {
  ExpandMore,
  ExpandLess,
  Add,
  Edit,
  Delete,
  CheckCircle,
  RadioButtonUnchecked,
  PlayArrow,
  Cancel
} from '@mui/icons-material';
import { useAuth } from '../../auth/useAuth';
import { TodoItem } from '../../types/TodoItem';

interface ConversationTodosProps {
  conversationId: number;
}

const ConversationTodos: React.FC<ConversationTodosProps> = ({ conversationId }) => {
  const { getToken } = useAuth();
  const [todos, setTodos] = useState<TodoItem[]>([]);
  const [expanded, setExpanded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [editingTodo, setEditingTodo] = useState<TodoItem | null>(null);

  // Form state
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [status, setStatus] = useState<'not-started' | 'in-progress' | 'completed' | 'cancelled'>('not-started');
  const [priority, setPriority] = useState<'low' | 'medium' | 'high' | 'urgent'>('medium');

  const resetForm = () => {
    setTitle('');
    setDescription('');
    setStatus('not-started');
    setPriority('medium');
  };

  const fetchTodos = async () => {
    if (!conversationId) return;

    setLoading(true);
    try {
      const token = await getToken();
      const response = await fetch(`/api/todos/conversation/${conversationId}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const todoData = await response.json();
        setTodos(todoData);
      }
    } catch (error) {
      console.error('Failed to fetch conversation todos:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (expanded && conversationId) {
      fetchTodos();
    }
  }, [expanded, conversationId]);

  const createTodo = async () => {
    if (!title.trim()) return;

    try {
      const token = await getToken();
      const response = await fetch('/api/todos/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          title: title.trim(),
          description: description.trim() || null,
          status,
          priority,
          conversation_id: conversationId
        })
      });

      if (response.ok) {
        const newTodo = await response.json();
        setTodos(prev => [newTodo, ...prev]);
        setCreateDialogOpen(false);
        resetForm();
      }
    } catch (error) {
      console.error('Failed to create todo:', error);
    }
  };

  const updateTodo = async (todo: TodoItem) => {
    try {
      const token = await getToken();
      const response = await fetch(`/api/todos/${todo.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          title: todo.title,
          description: todo.description,
          status: todo.status,
          priority: todo.priority
        })
      });

      if (response.ok) {
        const updatedTodo = await response.json();
        setTodos(prev => prev.map(t => t.id === todo.id ? updatedTodo : t));
      }
    } catch (error) {
      console.error('Failed to update todo:', error);
    }
  };

  const deleteTodo = async (todoId: number) => {
    try {
      const token = await getToken();
      const response = await fetch(`/api/todos/${todoId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        setTodos(prev => prev.filter(t => t.id !== todoId));
      }
    } catch (error) {
      console.error('Failed to delete todo:', error);
    }
  };

  const toggleTodoStatus = async (todo: TodoItem) => {
    const newStatus = todo.status === 'completed' ? 'not-started' : 'completed';
    await updateTodo({ ...todo, status: newStatus });
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle color="success" />;
      case 'in-progress':
        return <PlayArrow color="primary" />;
      case 'cancelled':
        return <Cancel color="error" />;
      default:
        return <RadioButtonUnchecked color="disabled" />;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'urgent':
        return 'error';
      case 'high':
        return 'warning';
      case 'medium':
        return 'primary';
      case 'low':
        return 'default';
      default:
        return 'default';
    }
  };

  const handleEdit = (todo: TodoItem) => {
    setEditingTodo(todo);
    setTitle(todo.title);
    setDescription(todo.description || '');
    setStatus(todo.status as any);
    setPriority(todo.priority as any);
    setCreateDialogOpen(true);
  };

  const handleSave = async () => {
    if (!title.trim()) return;

    if (editingTodo) {
      // Update existing todo
      await updateTodo({
        ...editingTodo,
        title: title.trim(),
        description: description.trim() || null,
        status,
        priority
      });
      setEditingTodo(null);
    } else {
      // Create new todo
      await createTodo();
    }

    setCreateDialogOpen(false);
    resetForm();
  };

  const handleDialogClose = () => {
    setCreateDialogOpen(false);
    setEditingTodo(null);
    resetForm();
  };

  if (!conversationId) return null;

  return (
    <Paper elevation={1} sx={{ mb: 2, border: '1px solid #e0e0e0' }}>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          p: 1.5,
          cursor: 'pointer',
          '&:hover': {
            backgroundColor: 'action.hover'
          }
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="subtitle2" color="text.secondary">
            Conversation Todos
          </Typography>
          {todos.length > 0 && (
            <Chip
              size="small"
              label={`${todos.filter(t => t.status === 'completed').length}/${todos.length}`}
              color={todos.every(t => t.status === 'completed') ? 'success' : 'default'}
            />
          )}
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <IconButton
            size="small"
            onClick={(e) => {
              e.stopPropagation();
              setCreateDialogOpen(true);
            }}
          >
            <Add />
          </IconButton>
          <IconButton size="small">
            {expanded ? <ExpandLess /> : <ExpandMore />}
          </IconButton>
        </Box>
      </Box>

      <Collapse in={expanded}>
        <Divider />
        <Box sx={{ p: 1.5 }}>
          {loading ? (
            <Typography color="text.secondary" align="center">
              Loading todos...
            </Typography>
          ) : todos.length === 0 ? (
            <Typography color="text.secondary" align="center">
              No todos for this conversation yet.
            </Typography>
          ) : (
            <List dense>
              {todos.map((todo) => (
                <ListItem
                  key={todo.id}
                  sx={{
                    border: '1px solid #e0e0e0',
                    borderRadius: 1,
                    mb: 1,
                    backgroundColor: todo.status === 'completed' ? 'action.hover' : 'background.paper'
                  }}
                >
                  <IconButton
                    size="small"
                    onClick={() => toggleTodoStatus(todo)}
                    sx={{ mr: 1 }}
                  >
                    {getStatusIcon(todo.status)}
                  </IconButton>

                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography
                          variant="body2"
                          sx={{
                            textDecoration: todo.status === 'completed' ? 'line-through' : 'none',
                            opacity: todo.status === 'completed' ? 0.7 : 1
                          }}
                        >
                          {todo.title}
                        </Typography>
                        <Chip
                          size="small"
                          label={todo.priority}
                          color={getPriorityColor(todo.priority) as any}
                          variant="outlined"
                        />
                      </Box>
                    }
                    secondary={todo.description}
                  />

                  <ListItemSecondaryAction>
                    <IconButton
                      size="small"
                      onClick={() => handleEdit(todo)}
                      sx={{ mr: 0.5 }}
                    >
                      <Edit />
                    </IconButton>
                    <IconButton
                      size="small"
                      onClick={() => deleteTodo(todo.id)}
                      color="error"
                    >
                      <Delete />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          )}
        </Box>
      </Collapse>

      {/* Create/Edit Todo Dialog */}
      <Dialog open={createDialogOpen} onClose={handleDialogClose} maxWidth="sm" fullWidth>
        <DialogTitle>
          {editingTodo ? 'Edit Todo' : 'Create New Todo'}
        </DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Title"
            fullWidth
            variant="outlined"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="Description"
            fullWidth
            multiline
            rows={3}
            variant="outlined"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            sx={{ mb: 2 }}
          />
          <Box sx={{ display: 'flex', gap: 2 }}>
            <FormControl fullWidth>
              <InputLabel>Status</InputLabel>
              <Select
                value={status}
                label="Status"
                onChange={(e) => setStatus(e.target.value as any)}
              >
                <MenuItem value="not-started">Not Started</MenuItem>
                <MenuItem value="in-progress">In Progress</MenuItem>
                <MenuItem value="completed">Completed</MenuItem>
                <MenuItem value="cancelled">Cancelled</MenuItem>
              </Select>
            </FormControl>
            <FormControl fullWidth>
              <InputLabel>Priority</InputLabel>
              <Select
                value={priority}
                label="Priority"
                onChange={(e) => setPriority(e.target.value as any)}
              >
                <MenuItem value="low">Low</MenuItem>
                <MenuItem value="medium">Medium</MenuItem>
                <MenuItem value="high">High</MenuItem>
                <MenuItem value="urgent">Urgent</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDialogClose}>Cancel</Button>
          <Button onClick={handleSave} variant="contained" disabled={!title.trim()}>
            {editingTodo ? 'Update' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
};

export default ConversationTodos;