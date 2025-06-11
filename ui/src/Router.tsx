import './App.css';
import ChatPage from './pages/ChatPage';
import SettingsPage from './pages/SettingsPage';
import { Route, Routes } from 'react-router-dom';
import ModelProfilesPage from './pages/ModelProfilesPage';

function Router() {
  return (
    <Routes>
      <Route path="/" element={<ChatPage />} />
      <Route path="/chat/:conversationId" element={<ChatPage />} />
      <Route path="/settings/:tab?" element={<SettingsPage />} />
      <Route path="/model-profiles" element={<ModelProfilesPage />} />
    </Routes>
  );
}

export default Router;