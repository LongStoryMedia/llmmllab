import { TodoItem } from "../types/TodoItem";
import { getHeaders, req } from "./base";

export interface CreateTodoRequest {
  title: string;
  description?: string;
  status?: string;
  priority?: string;
  due_date?: Date;
}

export interface UpdateTodoRequest {
  title: string;
  description?: string;
  status: string;
  priority: string;
  due_date?: Date;
}

export const getTodos = async (accessToken: string, status?: string) =>
  req<TodoItem[]>({
    method: 'GET',
    headers: getHeaders(accessToken),
    path: `todos${status ? `?status=${encodeURIComponent(status)}` : ''}`
  });

export const getTodo = async (accessToken: string, todoId: number) =>
  req<TodoItem>({
    method: 'GET',
    headers: getHeaders(accessToken),
    path: `todos/${todoId}`
  });

export const createTodo = async (accessToken: string, todoRequest: CreateTodoRequest) =>
  req<TodoItem>({
    method: 'POST',
    headers: getHeaders(accessToken),
    path: 'todos',
    body: JSON.stringify(todoRequest)
  });

export const updateTodo = async (accessToken: string, todoId: number, todoRequest: UpdateTodoRequest) =>
  req<TodoItem>({
    method: 'PUT',
    headers: getHeaders(accessToken),
    path: `todos/${todoId}`,
    body: JSON.stringify(todoRequest)
  });

export const deleteTodo = async (accessToken: string, todoId: number) =>
  req<{ message: string }>({
    method: 'DELETE',
    headers: getHeaders(accessToken),
    path: `todos/${todoId}`
  });