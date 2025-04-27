import '../styles/Sections.css';

const ProjectOverview = () => {
  return (
    <div id="overview" className="section">
      <div className="section-anchor-target"></div>
      <h2 className="section-title">Project Overview</h2>
      
      <div className="overview-intro">
        <div className="overview-text">
          <h3>BankGuard<span className="highlight">ML</span>: Advanced Fraud Detection System</h3>
          <p>
            Banking fraud is a significant challenge for financial institutions, leading to trillions in annual losses
            and damaging customer trust. In Taiwan, fraud has cost more than 21 billion NTD in a single quarter. Our machine learning solution aims to address this critical business problem
            by providing early detection capabilities for potentially fraudulent accounts.
          </p>
        </div>
        <div className="overview-image">
          <img src="/images/fraud-detection.svg" alt="Bank Fraud Detection" />
        </div>
      </div>
      
      <div className="overview-cards">
        <div className="overview-card">
          <h4>Business Problem</h4>
          <p>
            Traditional fraud detection systems often identify fraud only <strong>AFTER</strong> significant damage has occurred.
            Financial institutions need proactive tools to flag suspicious accounts before fraudulent activity escalates.
          </p>
        </div>
        
        <div className="overview-card">
          <h4>Project Goal</h4>
          <p>
            Build a machine learning model that creates an early warning watchlist of potentially
            fraudulent accounts based on transaction patterns, account characteristics, and customer behavior.
            Detecting fraudulent activity <strong>BEFORE</strong> financial losses occur.
          </p>
        </div>
        
        <div className="overview-card">
          <h4>Value Proposition</h4>
          <p>
            Our solution provides three key benefits:
            <ul>
              <li>Reduced financial losses through early fraud detection</li>
              <li>Secure platform for analyzing bank account data</li>
              <li>Strengthened regulatory compliance and risk management</li>
            </ul>
          </p>
        </div>
      </div>
      
      <div className="overview-stats">
        <div className="stat-item">
          <h3>$5.127T+</h3>
          <p>Annual cost of banking fraud globally in 2024</p>
        </div>
        <div className="stat-item">
          <h3>1-2%</h3>
          <p>Fraud accounts in provided data</p>
        </div>
        <div className="stat-item">
          <h3>90%</h3>
          <p>Financial institutions that adopt ML techniques for fraud detection</p>
        </div>
        <div className="stat-item">
          <div className="metric-container">
            <h3>91.1%</h3>
          </div>
          <p>Model accuracy in identifying suspicious accounts</p>
          <span className="metric-note">Based on F1-score</span>
        </div>
      </div>
    </div>
  );
};

export default ProjectOverview;
