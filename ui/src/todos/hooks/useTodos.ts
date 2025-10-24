import { useState, useEffect, useCallback } from 'react';
import { TodoItem } from '../../types/TodoItem';
import { 
  getTodos, 
  getTodo, 
  createTodo, 
  updateTodo, 
  deleteTodo,
  CreateTodoRequest,
  UpdateTodoRequest 
} from '../../api/todos';
import { useAuth } from '../../auth/useAuth';

interface UseTodosResult {
  todos: TodoItem[];
  loading: boolean;
  error: string | null;
  createTodoItem: (todo: CreateTodoRequest) => Promise<TodoItem | null>;
  updateTodoItem: (id: number, todo: UpdateTodoRequest) => Promise<TodoItem | null>;
  deleteTodoItem: (id: number) => Promise<boolean>;
  refreshTodos: () => Promise<void>;
  getTodoById: (id: number) => Promise<TodoItem | null>;
  filterByStatus: (status?: string) => void;
}

export function useTodos(): UseTodosResult {
  const auth = useAuth();
  const [todos, setTodos] = useState<TodoItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string | undefined>();

  const getAccessToken = useCallback(() => {
    if (!auth.user?.access_token) {
      throw new Error('No access token available');
    }
    return auth.user.access_token;
  }, [auth.user]);

  const refreshTodos = useCallback(async () => {
    if (!auth.user) {
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const accessToken = getAccessToken();
      const todosData = await getTodos(accessToken, statusFilter);
      setTodos(todosData);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch todos';
      setError(errorMessage);
      console.error('Error fetching todos:', err);
    } finally {
      setLoading(false);
    }
  }, [auth.user, statusFilter, getAccessToken]);

  const createTodoItem = useCallback(async (todoRequest: CreateTodoRequest): Promise<TodoItem | null> => {
    try {
      setError(null);
      const accessToken = getAccessToken();
      const newTodo = await createTodo(accessToken, {
        ...todoRequest,
        status: todoRequest.status || 'not-started',
        priority: todoRequest.priority || 'medium'
      });
      
      // Add to local state
      setTodos(prev => [newTodo, ...prev]);
      return newTodo;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to create todo';
      setError(errorMessage);
      console.error('Error creating todo:', err);
      return null;
    }
  }, [getAccessToken]);

  const updateTodoItem = useCallback(async (id: number, todoRequest: UpdateTodoRequest): Promise<TodoItem | null> => {
    try {
      setError(null);
      const accessToken = getAccessToken();
      const updatedTodo = await updateTodo(accessToken, id, todoRequest);
      
      // Update local state
      setTodos(prev => prev.map(todo => 
        todo.id === id ? updatedTodo : todo
      ));
      return updatedTodo;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to update todo';
      setError(errorMessage);
      console.error('Error updating todo:', err);
      return null;
    }
  }, [getAccessToken]);

  const deleteTodoItem = useCallback(async (id: number): Promise<boolean> => {
    try {
      setError(null);
      const accessToken = getAccessToken();
      await deleteTodo(accessToken, id);
      
      // Remove from local state
      setTodos(prev => prev.filter(todo => todo.id !== id));
      return true;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete todo';
      setError(errorMessage);
      console.error('Error deleting todo:', err);
      return false;
    }
  }, [getAccessToken]);

  const getTodoById = useCallback(async (id: number): Promise<TodoItem | null> => {
    try {
      setError(null);
      const accessToken = getAccessToken();
      const todo = await getTodo(accessToken, id);
      return todo;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch todo';
      setError(errorMessage);
      console.error('Error fetching todo:', err);
      return null;
    }
  }, [getAccessToken]);

  const filterByStatus = useCallback((status?: string) => {
    setStatusFilter(status);
  }, []);

  // Load todos when component mounts or when filter changes
  useEffect(() => {
    if (auth.user) {
      refreshTodos();
    }
  }, [auth.user, refreshTodos]);

  return {
    todos,
    loading,
    error,
    createTodoItem,
    updateTodoItem,
    deleteTodoItem,
    refreshTodos,
    getTodoById,
    filterByStatus
  };
}