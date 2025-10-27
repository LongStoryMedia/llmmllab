import React, { useState } from 'react';
import {
  Box,
  Card,
  Typography,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  CardContent,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  styled
} from '@mui/material';
import { Edit, Delete, Add } from '@mui/icons-material';
import { TodoItem } from '../../types/TodoItem';
import { useTodos } from '../hooks/useTodos';
import { CreateTodoRequest, UpdateTodoRequest } from '../../api/todos';

interface TodoListProps {
  className?: string;
}

// Styled components
const TodoContainer = styled(Box)(({ theme }) => ({
  maxWidth: '1200px',
  margin: '0 auto',
  padding: theme.spacing(3)
}));

const HeaderSection = styled(Box)(({ theme }) => ({
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  marginBottom: theme.spacing(3),
  [theme.breakpoints.down('sm')]: {
    flexDirection: 'column',
    gap: theme.spacing(2),
    alignItems: 'stretch'
  }
}));

const FilterSection = styled(Box)(({ theme }) => ({
  marginBottom: theme.spacing(3)
}));

const EmptyState = styled(Box)(({ theme }) => ({
  textAlign: 'center',
  padding: theme.spacing(8)
}));

const TodoCard = styled(Card)<{ priority: string }>(({ theme, priority }) => {
  const borderColor = {
    low: theme.palette.grey[400],
    medium: theme.palette.warning.main,
    high: theme.palette.error.light,
    urgent: theme.palette.error.main
  }[priority] || theme.palette.grey[400];

  return {
    marginBottom: theme.spacing(2),
    borderLeft: `4px solid ${borderColor}`,
    '&:hover': {
      elevation: 2
    }
  };
});

const TodoHeader = styled(Box)({
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'flex-start',
  marginBottom: 8
});

const TodoMetadata = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  flexWrap: 'wrap'
});

const TodoActions = styled(Box)({
  display: 'flex',
  gap: 4
});

const statusOptions = [
  { value: '', label: 'All' },
  { value: 'not-started', label: 'Not Started' },
  { value: 'in-progress', label: 'In Progress' },
  { value: 'completed', label: 'Completed' },
  { value: 'cancelled', label: 'Cancelled' }
];

const priorityOptions = [
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
  { value: 'urgent', label: 'Urgent' }
];



export function TodoList({ className = '' }: TodoListProps) {
  const {
    todos,
    loading,
    error,
    createTodoItem,
    updateTodoItem,
    deleteTodoItem,
    filterByStatus
  } = useTodos();

  const [showCreateForm, setShowCreateForm] = useState(false);
  const [editingTodo, setEditingTodo] = useState<TodoItem | null>(null);
  const [statusFilter, setStatusFilter] = useState('');

  const [createForm, setCreateForm] = useState<CreateTodoRequest>({
    title: '',
    description: '',
    status: 'not-started',
    priority: 'medium'
  });

  const [editForm, setEditForm] = useState<UpdateTodoRequest>({
    title: '',
    description: '',
    status: 'not-started',
    priority: 'medium'
  });

  const handleStatusFilterChange = (status: string) => {
    setStatusFilter(status);
    filterByStatus(status || undefined);
  };

  const handleCreateSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!createForm.title.trim()) {
      return;
    }

    const success = await createTodoItem(createForm);
    if (success) {
      setCreateForm({
        title: '',
        description: '',
        status: 'not-started',
        priority: 'medium'
      });
      setShowCreateForm(false);
    }
  };

  const handleEditSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingTodo?.id || !editForm.title.trim()) {
      return;
    }

    const success = await updateTodoItem(editingTodo.id, editForm);
    if (success) {
      setEditingTodo(null);
      setEditForm({
        title: '',
        description: '',
        status: 'not-started',
        priority: 'medium'
      });
    }
  };

  const handleEdit = (todo: TodoItem) => {
    setEditingTodo(todo);
    setEditForm({
      title: todo.title,
      description: todo.description || '',
      status: todo.status,
      priority: todo.priority,
      due_date: todo.due_date
    });
  };

  const handleDelete = async (id: number) => {
    if (window.confirm('Are you sure you want to delete this todo?')) {
      await deleteTodoItem(id);
    }
  };

  const formatDate = (date: Date | undefined) => {
    if (!date) {
      return null;
    }
    return new Date(date).toLocaleDateString();
  };

  if (loading && todos.length === 0) {
    return (
      <div className={`flex items-center justify-center p-8 ${className}`}>
        <div className="text-gray-500">Loading todos...</div>
      </div>
    );
  }

  return (
    <TodoContainer className={className}>
      <HeaderSection>
        <Typography variant="h4" component="h1" color="primary">
          Todo List
        </Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setShowCreateForm(true)}
        >
          Add Todo
        </Button>
      </HeaderSection>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Status Filter */}
      <FilterSection>
        <FormControl sx={{ minWidth: 200 }}>
          <InputLabel>Filter by Status</InputLabel>
          <Select
            value={statusFilter}
            onChange={(e) => handleStatusFilterChange(e.target.value)}
            label="Filter by Status"
          >
            {statusOptions.map(option => (
              <MenuItem key={option.value} value={option.value}>
                {option.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </FilterSection>

      {/* Create Form Modal */}
      <Dialog
        open={showCreateForm}
        onClose={() => setShowCreateForm(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Create New Todo</DialogTitle>
        <form onSubmit={handleCreateSubmit}>
          <DialogContent>
            <TextField
              label="Title"
              value={createForm.title}
              onChange={(e) => setCreateForm({ ...createForm, title: e.target.value })}
              fullWidth
              required
              margin="normal"
            />
            <TextField
              label="Description"
              value={createForm.description}
              onChange={(e) => setCreateForm({ ...createForm, description: e.target.value })}
              fullWidth
              multiline
              rows={3}
              margin="normal"
            />
            <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
              <FormControl fullWidth>
                <InputLabel>Status</InputLabel>
                <Select
                  value={createForm.status}
                  onChange={(e) => setCreateForm({ ...createForm, status: e.target.value })}
                  label="Status"
                >
                  {statusOptions.slice(1).map(option => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <FormControl fullWidth>
                <InputLabel>Priority</InputLabel>
                <Select
                  value={createForm.priority}
                  onChange={(e) => setCreateForm({ ...createForm, priority: e.target.value })}
                  label="Priority"
                >
                  {priorityOptions.map(option => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setShowCreateForm(false)}>
              Cancel
            </Button>
            <Button type="submit" variant="contained">
              Create Todo
            </Button>
          </DialogActions>
        </form>
      </Dialog>

      {/* Edit Form Modal */}
      <Dialog
        open={!!editingTodo}
        onClose={() => setEditingTodo(null)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Edit Todo</DialogTitle>
        <form onSubmit={handleEditSubmit}>
          <DialogContent>
            <TextField
              label="Title"
              value={editForm.title}
              onChange={(e) => setEditForm({ ...editForm, title: e.target.value })}
              fullWidth
              required
              margin="normal"
            />
            <TextField
              label="Description"
              value={editForm.description}
              onChange={(e) => setEditForm({ ...editForm, description: e.target.value })}
              fullWidth
              multiline
              rows={3}
              margin="normal"
            />
            <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
              <FormControl fullWidth>
                <InputLabel>Status</InputLabel>
                <Select
                  value={editForm.status}
                  onChange={(e) => setEditForm({ ...editForm, status: e.target.value })}
                  label="Status"
                >
                  {statusOptions.slice(1).map(option => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <FormControl fullWidth>
                <InputLabel>Priority</InputLabel>
                <Select
                  value={editForm.priority}
                  onChange={(e) => setEditForm({ ...editForm, priority: e.target.value })}
                  label="Priority"
                >
                  {priorityOptions.map(option => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setEditingTodo(null)}>
              Cancel
            </Button>
            <Button type="submit" variant="contained">
              Update Todo
            </Button>
          </DialogActions>
        </form>
      </Dialog>

      {/* Todo List */}
      {todos.length === 0 ? (
        <EmptyState>
          <Typography variant="h6" color="text.secondary" gutterBottom>
            No todos found
          </Typography>
          <Button
            variant="text"
            color="primary"
            onClick={() => setShowCreateForm(true)}
          >
            Create your first todo
          </Button>
        </EmptyState>
      ) : (
        <Box>
          {todos.map((todo) => (
            <TodoCard
              key={todo.id}
              priority={todo.priority}
            >
              <CardContent>
                <TodoHeader>
                  <Box sx={{ flex: 1 }}>
                    <TodoMetadata>
                      <Typography variant="h6" component="h3">
                        {todo.title}
                      </Typography>
                      <Chip
                        label={statusOptions.find(s => s.value === todo.status)?.label}
                        size="small"
                        color={
                          todo.status === 'completed' ? 'success' :
                            todo.status === 'in-progress' ? 'primary' :
                              todo.status === 'cancelled' ? 'error' : 'default'
                        }
                      />
                      <Chip
                        label={`${todo.priority} priority`}
                        size="small"
                        variant="outlined"
                      />
                    </TodoMetadata>
                    {todo.description && (
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                        {todo.description}
                      </Typography>
                    )}
                    <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
                      {todo.due_date && (
                        <Typography variant="caption" color="text.secondary">
                          Due: {formatDate(todo.due_date)}
                        </Typography>
                      )}
                      {todo.created_at && (
                        <Typography variant="caption" color="text.secondary">
                          Created: {formatDate(todo.created_at)}
                        </Typography>
                      )}
                    </Box>
                  </Box>
                  <TodoActions>
                    <IconButton
                      size="small"
                      onClick={() => handleEdit(todo)}
                      color="primary"
                    >
                      <Edit />
                    </IconButton>
                    <IconButton
                      size="small"
                      onClick={() => todo.id && handleDelete(todo.id)}
                      color="error"
                    >
                      <Delete />
                    </IconButton>
                  </TodoActions>
                </TodoHeader>
              </CardContent>
            </TodoCard>
          ))}
        </Box>
      )}
    </TodoContainer>
  );
}