import { useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import SettingsTabs from '../components/Settings/SettingsTabs';



const SettingsPage = () => {
  const { tab } = useParams();
  const navigate = useNavigate();

  useEffect(() => {
    if (!tab) {
      navigate(`/settings/profile`, { replace: true });
    } 
  }, [tab, navigate]);
  
  return (
    <SettingsTabs onTabChange={tab => navigate(`/settings/${tab}`, { replace: true })} />
  );
};

export default SettingsPage;