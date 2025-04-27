import { useState } from 'react';
import './App.css';
import Header from './components/Header';
import Footer from './components/Footer';
import ProjectOverview from './components/ProjectOverview';
import DataOverview from './components/DataOverview';
import Methodology from './components/Methodology';

function App() {
  const [activeSection, setActiveSection] = useState('overview');

  const renderSection = () => {
    switch (activeSection) {
      case 'overview':
        return <ProjectOverview />;
      case 'data':
        return <DataOverview />;
      case 'methodology':
        return <Methodology />;
      // Add these additional cases once the components are created
      /*
      case 'results':
        return <Results />;
      case 'insights':
        return <ModelInsights />;
      case 'implementation':
        return <Implementation />;
      case 'future':
        return <FutureImprovements />;
      */
      default:
        return <ProjectOverview />;
    }
  };

  return (
    <div className="app-container">
      <Header activeSection={activeSection} setActiveSection={setActiveSection} />
      <div className="main-content-wrapper">
        <main className="main-content">
          {renderSection()}
        </main>
      </div>
      <Footer />
    </div>
  );
}

export default App;
