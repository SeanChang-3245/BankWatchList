import '../styles/Sections.css';

const DataOverview = () => {
  return (
    <div id="data" className="section">
      <div className="section-anchor-target"></div>
      <h2 className="section-title">Data Overview</h2>

      <div className="card">
        <h3>Data Sources</h3>
        <div className="data-sources">
          <div className="data-source-item">
            <h4>ECCUS Dataset</h4>
            <p>Contains confirmed fraud accounts with customer IDs, account numbers, and dates</p>
            <span className="badge badge-primary">Fraud Labels</span>
          </div>
          <div className="data-source-item">
            <h4>SAVTXN Dataset</h4>
            <p>Comprehensive transaction data with timestamps, amounts, types, and recipient details</p>
            <span className="badge badge-primary">Core Dataset</span>
          </div>
          <div className="data-source-item">
            <h4>Customer Profile Data</h4>
            <p>Demographic information and account histories</p>
            <span className="badge badge-secondary">Supplementary</span>
          </div>
        </div>
      </div>

      <div className="flex-container">
        <div className="flex-item card">
          <h3>Key Statistics</h3>
          <ul className="stat-list">
            <li><strong>Fraud Impact:</strong> 21+ billion NTD lost in a quarter</li>
            <li><strong>Fraud Rate:</strong> 1-2% of accounts identified as fraudulent</li>
            <li><strong>Detection Accuracy:</strong> 91.1% F1-score in identifying suspicious accounts</li>
            <li><strong>Data Coverage:</strong> 24+ months of transaction history</li>
            <li><strong>Industry Adoption:</strong> 90% of financial institutions use ML for fraud detection</li>
          </ul>
        </div>

        <div className="flex-item card">
          <h3>Feature Engineering</h3>
          <p>Detecting fraudulent activity <strong>BEFORE</strong> financial losses occur requires transforming transaction data into predictive features.</p>
          <div className="feature-categories">
            <div className="feature-category">
              <h4>Temporal Patterns</h4>
              <ul>
                <li>Sudden changes in transaction frequency</li>
                <li>Unusual hour-of-day activity</li>
                <li>Account dormancy followed by high activity</li>
              </ul>
            </div>
      
            <div className="feature-category">
              <h4>Amount Patterns</h4>
              <ul>
                <li>Sequential small transfers (structuring)</li>
                <li>Deviation from historical amount ranges</li>
                <li>Round-sum transactions</li>
              </ul>
            </div>
      
            <div className="feature-category">
              <h4>Network Features</h4>
              <ul>
                <li>Transactions to high-risk regions</li>
                <li>New beneficiary account patterns</li>
                <li>Shared recipient accounts across multiple customers</li>
              </ul>
            </div>
      
            <div className="feature-category">
              <h4>Account Behavior</h4>
              <ul>
                <li>Account emptying velocity</li>
                <li>Login location vs. transaction location</li>
                <li>Multi-channel transaction patterns</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataOverview;