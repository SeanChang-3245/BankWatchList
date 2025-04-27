import { useState } from 'react';
import '../styles/Header.css';

interface HeaderProps {
  activeSection: string;
  setActiveSection: (section: string) => void;
}

const Header: React.FC<HeaderProps> = ({ activeSection, setActiveSection }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { id: 'overview', label: 'Project Overview' },
    { id: 'data', label: 'Data Overview' },
    { id: 'methodology', label: 'Methodology' },
    { id: 'upload', label: 'Data Upload' },
  ];

  const handleNavClick = (sectionId: string) => {
    setActiveSection(sectionId);
    setMobileMenuOpen(false);
    
    // Scroll to section with proper offset
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <header className="header">
      <div className="header-container">
        <div className="logo">
          <h1>BankGuard<span className="highlight">ML</span></h1>
        </div>
        
        <button 
          className="mobile-menu-button" 
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          aria-label="Toggle menu"
        >
          <span className="menu-icon"></span>
        </button>
        
        <nav className={`nav-menu ${mobileMenuOpen ? 'open' : ''}`}>
          <ul>
            {navItems.map((item) => (
              <li key={item.id}>
                <button
                  className={activeSection === item.id ? 'active' : ''}
                  onClick={() => handleNavClick(item.id)}
                >
                  {item.label}
                </button>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </header>
  );
};

export default Header;
