import './index.css';
import ChatPage from './pages/ChatPage';
import SettingsPage from './pages/SettingsPage';
import { Route, Routes } from 'react-router-dom';
import ImagePage from './pages/ImagePage';

function Router() {
  return (
    <Routes>
      <Route path="/" element={<ChatPage />} />
      <Route path="/chat/:conversationId" element={<ChatPage />} />
      <Route path="/images" element={<ImagePage />} />
      <Route path="/settings/:tab?" element={<SettingsPage />} />
    </Routes>
  );
}

export default Router;