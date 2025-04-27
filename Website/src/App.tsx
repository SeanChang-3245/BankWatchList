import { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import './App.css';
import Header from './components/Header';
import Footer from './components/Footer';
import ProjectOverview from './components/ProjectOverview';
import DataOverview from './components/DataOverview';
import Methodology from './components/Methodology';
import DataUpload from './components/DataUpload';

// This component will be inside the Router context
function AppContent() {
  const [activeSection, setActiveSection] = useState('overview');
  const navigate = useNavigate();

  useEffect(() => {
    switch (activeSection) {
      case 'overview':
        navigate('/');
        break;
      case 'data':
        navigate('/data');
        break;
      case 'methodology':
        navigate('/methodology');
        break;
      case 'upload':
        navigate('/upload');
        break;
      default:
        navigate('/');
    }
  }, [activeSection, navigate]);

  return (
    <div className="app-container">
      <Header activeSection={activeSection} setActiveSection={setActiveSection} />
      <div className="else-container">
        <div className="main-content-wrapper">
          <main className="main-content">
            <Routes>
              <Route path="/" element={<ProjectOverview />} />
              <Route path="/data" element={<DataOverview />} />
              <Route path="/methodology" element={<Methodology />} />
              <Route path="/upload" element={<DataUpload />} />
            </Routes>
          </main>
        </div>
        <Footer />
      </div>
      
    </div>
  );
}

// Main App just provides the Router context
function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App;