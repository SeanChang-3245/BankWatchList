import '../styles/Sections.css';

const Methodology = () => {
  return (
    <div className="section">
      <h2 className="section-title">Methodology</h2>
      
      <div className="card">
        <h3>Approach Overview</h3>
        <p>
          We implemented a supervised learning approach to detect fraud, leveraging labeled data of known 
          fraudulent and legitimate accounts. Our methodology focuses on balancing precision and recall 
          to create an effective early warning system.
        </p>
        
        <div className="approach-diagram">
          <img src="/images/ml-pipeline.svg" alt="ML Pipeline Diagram" className="full-width-image" />
        </div>
      </div>
      
      <div className="flex-container">
        <div className="flex-item card">
          <h3>Model Selection</h3>
          <p>After evaluating several machine learning algorithms, we selected <strong>XGBoost</strong> as our primary model due to:</p>
          
          <ul className="feature-list">
            <li>
              <span className="feature-icon">✓</span>
              <div>
                <strong>Performance with Imbalanced Data</strong>
                <p>Handles the inherent class imbalance in fraud detection (1-2% fraud rate)</p>
              </div>
            </li>
            
            <li>
              <span className="feature-icon">✓</span>
              <div>
                <strong>Feature Importance</strong>
                <p>Provides clear visibility into which factors are most predictive of fraud</p>
              </div>
            </li>
            
            <li>
              <span className="feature-icon">✓</span>
              <div>
                <strong>Robustness</strong>
                <p>Less sensitive to outliers, which are common in financial transaction data</p>
              </div>
            </li>
            
            <li>
              <span className="feature-icon">✓</span>
              <div>
                <strong>Scalability</strong>
                <p>Efficiently handles large datasets with numerous features</p>
              </div>
            </li>
          </ul>
        </div>
        
        <div className="flex-item card">
          <h3>Evaluation Strategy</h3>
          
          <div className="eval-strategy">
            <div className="eval-item">
              <h4>Cross-Validation</h4>
              <p>5-fold cross-validation to ensure model robustness</p>
            </div>
            
            <div className="eval-item">
              <h4>Time-Based Splits</h4>
              <p>Training on historical data, validation on recent data to simulate real-world use</p>
            </div>
            
            <div className="eval-item">
              <h4>Primary Metrics</h4>
              <ul>
                <li>AUC-ROC: Overall ranking performance</li>
                <li>Precision: Minimizing false positives</li>
                <li>Recall at 5%: Catch fraud while limiting investigation volume</li>
              </ul>
            </div>
            
            <div className="eval-item">
              <h4>Threshold Optimization</h4>
              <p>Custom threshold selection based on business cost-benefit analysis</p>
            </div>
          </div>
        </div>
      </div>
      
      <div className="card">
        <h3>Training Pipeline</h3>
        <p>
          We implemented our training pipeline using AWS SageMaker to ensure reproducibility, 
          scalability, and efficient model deployment.
        </p>
        
        <div className="pipeline-steps">
          <div className="pipeline-step">
            <div className="step-number">1</div>
            <div className="step-content">
              <h4>Data Preprocessing</h4>
              <p>Feature normalization, encoding categorical variables, and handling missing values</p>
            </div>
          </div>
          
          <div className="pipeline-step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h4>Feature Selection</h4>
              <p>Identifying the most predictive features using SHAP values and correlation analysis</p>
            </div>
          </div>
          
          <div className="pipeline-step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h4>Handling Class Imbalance</h4>
              <p>Using SMOTE for synthetic minority oversampling and adjusted class weights</p>
            </div>
          </div>
          
          <div className="pipeline-step">
            <div className="step-number">4</div>
            <div className="step-content">
              <h4>Hyperparameter Tuning</h4>
              <p>Bayesian optimization to find optimal model parameters</p>
            </div>
          </div>
          
          <div className="pipeline-step">
            <div className="step-number">5</div>
            <div className="step-content">
              <h4>Model Evaluation</h4>
              <p>Comprehensive performance assessment on holdout test data</p>
            </div>
          </div>
          
          <div className="pipeline-step">
            <div className="step-number">6</div>
            <div className="step-content">
              <h4>Deployment</h4>
              <p>Packaging the model for real-time scoring in the production environment</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Methodology;
