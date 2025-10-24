import { TodoList } from '../todos';

export default function TodoPage() {
  return (
    <div className="h-full flex flex-col">
      <TodoList className="flex-1" />
    </div>
  );
}