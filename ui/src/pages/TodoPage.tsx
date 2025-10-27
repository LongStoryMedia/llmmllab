import { Box } from '@mui/material';
import { TodoList } from '../todos';

export default function TodoPage() {
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ flex: 1 }}>
        <TodoList />
      </Box>
    </Box>
  );
}