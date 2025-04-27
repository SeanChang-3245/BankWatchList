import '../styles/Sections.css';

const DataOverview = () => {
  return (
    <div className="section">
      <h2 className="section-title">Data Overview</h2>
      
      <div className="card">
        <h3>Data Sources</h3>
        <div className="data-sources">
          <div className="data-source-item">
            <h4>Account Data</h4>
            <p>Basic account information including age, type, balance, and status</p>
            <span className="badge badge-primary">Core Dataset</span>
          </div>
          <div className="data-source-item">
            <h4>Transaction History</h4>
            <p>Detailed records of financial transactions, including amounts, dates, types, and counterparties</p>
            <span className="badge badge-primary">Core Dataset</span>
          </div>
          <div className="data-source-item">
            <h4>Customer Data</h4>
            <p>Demographic and behavioral information about account holders</p>
            <span className="badge badge-secondary">Supplementary</span>
          </div>
        </div>
      </div>
      
      <div className="flex-container">
        <div className="flex-item card">
          <h3>Key Statistics</h3>
          <ul className="stat-list">
            <li><strong>Total Accounts:</strong> 100,000+</li>
            <li><strong>Fraud Rate:</strong> ~1-2% of accounts</li>
            <li><strong>Features:</strong> 45+ account-level variables</li>
            <li><strong>Time Period:</strong> 24 months of transaction data</li>
            <li><strong>Data Size:</strong> 1.5GB of structured data</li>
          </ul>
        </div>
        
        <div className="flex-item card">
          <h3>Data Quality</h3>
          <div className="quality-metrics">
            <div className="quality-metric">
              <div className="quality-bar" style={{ width: '92%' }}></div>
              <p>Completeness: 92%</p>
            </div>
            <div className="quality-metric">
              <div className="quality-bar" style={{ width: '89%' }}></div>
              <p>Consistency: 89%</p>
            </div>
            <div className="quality-metric">
              <div className="quality-bar" style={{ width: '94%' }}></div>
              <p>Accuracy: 94%</p>
            </div>
            <div className="quality-metric">
              <div className="quality-bar" style={{ width: '97%' }}></div>
              <p>Timeliness: 97%</p>
            </div>
          </div>
        </div>
      </div>
      
      <div className="card">
        <h3>Feature Engineering</h3>
        <p>The core challenge was transforming transaction-level data into account-level features that capture fraud patterns.</p>
        
        <div className="feature-categories">
          <div className="feature-category">
            <h4>Temporal Patterns</h4>
            <ul>
              <li>Transaction velocity (frequency over time)</li>
              <li>Time-of-day transaction patterns</li>
              <li>Weekend vs. weekday activity</li>
            </ul>
          </div>
          
          <div className="feature-category">
            <h4>Amount Patterns</h4>
            <ul>
              <li>Transaction amount statistics (mean, median, etc.)</li>
              <li>Large transaction frequency</li>
              <li>Unusual amount patterns</li>
            </ul>
          </div>
          
          <div className="feature-category">
            <h4>Network Features</h4>
            <ul>
              <li>Number of unique counterparties</li>
              <li>Geographic diversity of transactions</li>
              <li>New counterparty introduction rate</li>
            </ul>
          </div>
          
          <div className="feature-category">
            <h4>Account Behavior</h4>
            <ul>
              <li>Balance volatility</li>
              <li>Deposit-withdrawal patterns</li>
              <li>Account usage consistency</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataOverview;
