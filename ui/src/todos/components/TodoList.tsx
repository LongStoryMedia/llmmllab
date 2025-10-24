import React, { useState } from 'react';
import { TodoItem } from '../../types/TodoItem';
import { useTodos } from '../hooks/useTodos';
import { CreateTodoRequest, UpdateTodoRequest } from '../../api/todos';

interface TodoListProps {
  className?: string;
}

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

const statusColors = {
  'not-started': 'bg-gray-100 text-gray-800',
  'in-progress': 'bg-blue-100 text-blue-800',
  'completed': 'bg-green-100 text-green-800',
  'cancelled': 'bg-red-100 text-red-800'
};

const priorityColors = {
  'low': 'border-l-gray-400',
  'medium': 'border-l-yellow-400',
  'high': 'border-l-orange-400',
  'urgent': 'border-l-red-400'
};

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
    <div className={`max-w-4xl mx-auto p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Todo List</h1>
        <button
          onClick={() => setShowCreateForm(true)}
          className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors"
        >
          Add Todo
        </button>
      </div>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {/* Status Filter */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Filter by Status:
        </label>
        <select
          value={statusFilter}
          onChange={(e) => handleStatusFilterChange(e.target.value)}
          className="border border-gray-300 rounded-md px-3 py-2 bg-white"
        >
          {statusOptions.map(option => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      {/* Create Form Modal */}
      {showCreateForm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
            <h2 className="text-xl font-bold mb-4">Create New Todo</h2>
            <form onSubmit={handleCreateSubmit}>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Title *
                </label>
                <input
                  type="text"
                  value={createForm.title}
                  onChange={(e) => setCreateForm({ ...createForm, title: e.target.value })}
                  className="w-full border border-gray-300 rounded-md px-3 py-2"
                  required
                />
              </div>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Description
                </label>
                <textarea
                  value={createForm.description}
                  onChange={(e) => setCreateForm({ ...createForm, description: e.target.value })}
                  className="w-full border border-gray-300 rounded-md px-3 py-2 h-20"
                />
              </div>
              <div className="mb-4 flex gap-4">
                <div className="flex-1">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Status
                  </label>
                  <select
                    value={createForm.status}
                    onChange={(e) => setCreateForm({ ...createForm, status: e.target.value })}
                    className="w-full border border-gray-300 rounded-md px-3 py-2"
                  >
                    {statusOptions.slice(1).map(option => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="flex-1">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Priority
                  </label>
                  <select
                    value={createForm.priority}
                    onChange={(e) => setCreateForm({ ...createForm, priority: e.target.value })}
                    className="w-full border border-gray-300 rounded-md px-3 py-2"
                  >
                    {priorityOptions.map(option => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <div className="flex gap-3 justify-end">
                <button
                  type="button"
                  onClick={() => setShowCreateForm(false)}
                  className="px-4 py-2 text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                >
                  Create Todo
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Edit Form Modal */}
      {editingTodo && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
            <h2 className="text-xl font-bold mb-4">Edit Todo</h2>
            <form onSubmit={handleEditSubmit}>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Title *
                </label>
                <input
                  type="text"
                  value={editForm.title}
                  onChange={(e) => setEditForm({ ...editForm, title: e.target.value })}
                  className="w-full border border-gray-300 rounded-md px-3 py-2"
                  required
                />
              </div>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Description
                </label>
                <textarea
                  value={editForm.description}
                  onChange={(e) => setEditForm({ ...editForm, description: e.target.value })}
                  className="w-full border border-gray-300 rounded-md px-3 py-2 h-20"
                />
              </div>
              <div className="mb-4 flex gap-4">
                <div className="flex-1">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Status
                  </label>
                  <select
                    value={editForm.status}
                    onChange={(e) => setEditForm({ ...editForm, status: e.target.value })}
                    className="w-full border border-gray-300 rounded-md px-3 py-2"
                  >
                    {statusOptions.slice(1).map(option => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="flex-1">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Priority
                  </label>
                  <select
                    value={editForm.priority}
                    onChange={(e) => setEditForm({ ...editForm, priority: e.target.value })}
                    className="w-full border border-gray-300 rounded-md px-3 py-2"
                  >
                    {priorityOptions.map(option => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <div className="flex gap-3 justify-end">
                <button
                  type="button"
                  onClick={() => setEditingTodo(null)}
                  className="px-4 py-2 text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                >
                  Update Todo
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Todo List */}
      {todos.length === 0 ? (
        <div className="text-center py-12">
          <div className="text-gray-500 text-lg">No todos found</div>
          <button
            onClick={() => setShowCreateForm(true)}
            className="mt-4 text-blue-600 hover:text-blue-800"
          >
            Create your first todo
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          {todos.map((todo) => (
            <div
              key={todo.id}
              className={`bg-white border-l-4 rounded-lg shadow-sm p-4 ${
                priorityColors[todo.priority as keyof typeof priorityColors]
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <h3 className="font-semibold text-lg text-gray-900">
                      {todo.title}
                    </h3>
                    <span
                      className={`px-2 py-1 rounded-full text-xs font-medium ${
                        statusColors[todo.status as keyof typeof statusColors]
                      }`}
                    >
                      {statusOptions.find(s => s.value === todo.status)?.label}
                    </span>
                    <span className="text-xs text-gray-500 capitalize">
                      {todo.priority} priority
                    </span>
                  </div>
                  {todo.description && (
                    <p className="text-gray-600 mb-2">{todo.description}</p>
                  )}
                  <div className="flex items-center gap-4 text-sm text-gray-500">
                    {todo.due_date && (
                      <span>Due: {formatDate(todo.due_date)}</span>
                    )}
                    {todo.created_at && (
                      <span>Created: {formatDate(todo.created_at)}</span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2 ml-4">
                  <button
                    onClick={() => handleEdit(todo)}
                    className="text-blue-600 hover:text-blue-800 text-sm"
                  >
                    Edit
                  </button>
                  <button
                    onClick={() => todo.id && handleDelete(todo.id)}
                    className="text-red-600 hover:text-red-800 text-sm"
                  >
                    Delete
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}