import '../styles/Footer.css';

const Footer = () => {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-content">
          <div className="footer-section">
            <h3>BankGuard<span className="highlight">ML</span></h3>
            <p>Advanced machine learning solution for bank fraud detection and prevention.</p>
          </div>
          
          <div className="footer-section">
            <h4>Project Links</h4>
            <ul className="footer-links">
              <li><a href="#documentation">Documentation</a></li>
              <li><a href="#github">GitHub Repository</a></li>
              <li><a href="#contact">Contact Team</a></li>
            </ul>
          </div>
          
          <div className="footer-section">
            <h4>Resources</h4>
            <ul className="footer-links">
              <li><a href="#whitepaper">Methodology Whitepaper</a></li>
              <li><a href="#case-studies">Case Studies</a></li>
              <li><a href="#faq">FAQ</a></li>
            </ul>
          </div>
        </div>
        
        <div className="footer-bottom">
          <p>&copy; {currentYear} BankGuardML. All rights reserved.</p>
          <p>Machine Learning Hackathon Project</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
