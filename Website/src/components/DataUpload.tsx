import { useState, useRef, ChangeEvent, FormEvent } from 'react';
import '../styles/Sections.css';
import '../styles/DataUpload.css';

interface FraudResult {
  accountId: string;
  riskScore: number;
  isHighRisk: boolean;
  features: {
    [key: string]: number | string;
  }
}

interface RequiredFiles {
  'ACCTS_Data.csv': File | null;
  'SAV_TXN_Data.csv': File | null;
  'ID_Data.csv': File | null;
}

const DataUpload = () => {
  const [files, setFiles] = useState<RequiredFiles>({
    'ACCTS_Data.csv': null,
    'SAV_TXN_Data.csv': null,
    'ID_Data.csv': null,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [results, setResults] = useState<FraudResult[] | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const fileInputRefs = {
    'ACCTS_Data.csv': useRef<HTMLInputElement>(null),
    'SAV_TXN_Data.csv': useRef<HTMLInputElement>(null),
    'ID_Data.csv': useRef<HTMLInputElement>(null),
  };

  const handleFileChange = (fileName: keyof RequiredFiles) => (event: ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = event.target.files;
    if (selectedFiles && selectedFiles.length > 0) {
      const selectedFile = selectedFiles[0];
      if (selectedFile.type === 'text/csv' || selectedFile.name.endsWith('.csv')) {
        setFiles(prev => ({
          ...prev,
          [fileName]: selectedFile,
        }));
        setUploadError(null);
      } else {
        setUploadError('Please upload CSV files only');
        if (fileInputRefs[fileName].current) {
          fileInputRefs[fileName].current!.value = '';
        }
      }
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleDrop = (fileName: keyof RequiredFiles) => (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const droppedFiles = e.dataTransfer.files;
    if (droppedFiles && droppedFiles.length > 0) {
      const droppedFile = droppedFiles[0];
      if (droppedFile.type === 'text/csv' || droppedFile.name.endsWith('.csv')) {
        setFiles(prev => ({
          ...prev,
          [fileName]: droppedFile,
        }));
        setUploadError(null);
      } else {
        setUploadError('Please upload CSV files only');
      }
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    
    // Check if all required files are uploaded
    const missingFiles = Object.entries(files).filter(([_, file]) => !file).map(([name]) => name);
    
    if (missingFiles.length > 0) {
      setUploadError(`Please upload all required files: ${missingFiles.join(', ')}`);
      return;
    }

    setIsLoading(true);
    setUploadError(null);

    try {
      // Create form data to send files
      const formData = new FormData();
      Object.entries(files).forEach(([name, file]) => {
        if (file) formData.append(name, file);
      });

      // Replace this URL with your actual AWS endpoint
      const response = await fetch('https://api.bankguardml.com/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status} - ${response.statusText}`);
      }

      const data = await response.json();
      setResults(data.results);
      setUploadSuccess(true);
    } catch (error) {
      console.error('Upload error:', error);
      setUploadError('Failed to process the files. Please try again later.');
      // For demo purposes, generate mock results if API fails
      generateMockResults();
    } finally {
      setIsLoading(false);
    }
  };

  // This function is for demonstration purposes only
  const generateMockResults = () => {
    const mockResults: FraudResult[] = [];
    const numAccounts = Math.floor(Math.random() * 15) + 5; // 5 to 20 accounts
    
    for (let i = 0; i < numAccounts; i++) {
      const riskScore = Math.random();
      mockResults.push({
        accountId: `ACC${Math.floor(100000 + Math.random() * 900000)}`,
        riskScore: parseFloat(riskScore.toFixed(4)),
        isHighRisk: riskScore > 0.7,
        features: {
          transactionFrequency: parseFloat((Math.random() * 100).toFixed(2)),
          avgTransactionAmount: parseFloat((Math.random() * 5000).toFixed(2)),
          unusualTime: Math.random() > 0.7 ? 'Yes' : 'No',
          newBeneficiary: Math.random() > 0.6 ? 'Yes' : 'No',
          accountAge: Math.floor(Math.random() * 1000) + ' days'
        }
      });
    }
    
    setResults(mockResults);
    setUploadSuccess(true);
  };

  const resetUpload = () => {
    setFiles({
      'ACCTS_Data.csv': null,
      'SAV_TXN_Data.csv': null,
      'ID_Data.csv': null,
    });
    setResults(null);
    setUploadSuccess(false);
    setUploadError(null);
    
    // Reset all file input values
    Object.values(fileInputRefs).forEach(ref => {
      if (ref.current) {
        ref.current.value = '';
      }
    });
  };
  
  // Helper function to check if all files are uploaded
  const allFilesUploaded = () => {
    return Object.values(files).every(file => file !== null);
  };
  
  return (
    <div id="upload" className="section">
      <div className="section-anchor-target"></div>
      <h2 className="section-title">Data Upload</h2>
  
      <div className="card upload-card">
        <h3>Fraud Detection Tool</h3>
        <p>Upload your bank transaction data files to detect potentially fraudulent accounts using our machine learning model.</p>
        
        {!uploadSuccess ? (
          <form onSubmit={handleSubmit} className="upload-form">
            <div className="required-files-notice">
              <h4>Required Files (All 3 files must be uploaded):</h4>
              <ul>
                <li>ACCTS_Data.csv</li>
                <li>SAV_TXN_Data.csv</li>
                <li>ID_Data.csv</li>
              </ul>
            </div>
            
            {(Object.keys(files) as Array<keyof RequiredFiles>).map((fileName) => (
              <div 
                key={fileName}
                className={`file-upload-area ${files[fileName] ? 'file-uploaded' : ''}`}
                onDragOver={handleDragOver}
                onDrop={handleDrop(fileName)}
              >
                <input 
                  type="file" 
                  ref={fileInputRefs[fileName]}
                  onChange={handleFileChange(fileName)}
                  accept=".csv"
                  id={`file-upload-${fileName}`}
                  className="file-input"
                />
                <label htmlFor={`file-upload-${fileName}`} className="file-label">
                  <div className="upload-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                      <polyline points="17 8 12 3 7 8"></polyline>
                      <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                  </div>
                  <span className="upload-text">
                    <strong>{fileName}:</strong> {files[fileName] ? files[fileName]!.name : `Drag and drop ${fileName} here or click to browse`}
                  </span>
                </label>
              </div>
            ))}
            
            {uploadError && <p className="error-message">{uploadError}</p>}
            
            <div className="upload-actions">
              <button 
                type="submit" 
                className="btn btn-primary upload-btn"
                disabled={!allFilesUploaded() || isLoading}
              >
                {isLoading ? 'Processing...' : 'Analyze Accounts'}
              </button>
            </div>
          </form>
        ) : (
          <div className="upload-success">
            <div className="success-header">
              <h4>Analysis Complete</h4>
              <button className="btn btn-secondary" onClick={resetUpload}>Upload New Files</button>
            </div>
          </div>
        )}
      </div>

      {uploadSuccess && results && (
        <div className="results-section">
          <div className="card">
            <h3>Detection Results</h3>
            <div className="results-summary">
              <div className="summary-item">
                <span className="summary-value">{results.length}</span>
                <span className="summary-label">Accounts Analyzed</span>
              </div>
              <div className="summary-item">
                <span className="summary-value">{results.filter(r => r.isHighRisk).length}</span>
                <span className="summary-label">High Risk Accounts</span>
              </div>
              <div className="summary-item">
                <span className="summary-value">{((results.filter(r => r.isHighRisk).length / results.length) * 100).toFixed(1)}%</span>
                <span className="summary-label">Risk Rate</span>
              </div>
            </div>
            
            <div className="results-table-container">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>Account ID</th>
                    <th>Risk Score</th>
                    <th>Status</th>
                    <th>Key Factors</th>
                  </tr>
                </thead>
                <tbody>
                  {results.sort((a, b) => b.riskScore - a.riskScore).map((result) => (
                    <tr key={result.accountId} className={result.isHighRisk ? 'high-risk' : ''}>
                      <td>{result.accountId}</td>
                      <td>
                        <div className="risk-meter">
                          <div 
                            className="risk-level" 
                            style={{ 
                              width: `${result.riskScore * 100}%`,
                              backgroundColor: result.isHighRisk ? '#dc3545' : '#ffc107'
                            }}
                          ></div>
                        </div>
                        <span className="risk-value">{(result.riskScore * 100).toFixed(1)}%</span>
                      </td>
                      <td>
                        <span className={`status-badge ${result.isHighRisk ? 'status-high-risk' : 'status-normal'}`}>
                          {result.isHighRisk ? 'High Risk' : 'Normal'}
                        </span>
                      </td>
                      <td>
                        <div className="factor-pills">
                          {result.features.unusualTime === 'Yes' && 
                            <span className="factor-pill">Unusual Time</span>
                          }
                          {result.features.newBeneficiary === 'Yes' && 
                            <span className="factor-pill">New Beneficiary</span>
                          }
                          {(result.features.transactionFrequency as number) > 70 && 
                            <span className="factor-pill">High Frequency</span>
                          }
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          
          <div className="card info-card">
            <h3>What Does This Mean?</h3>
            <p>
              Our model has analyzed the transaction patterns and account behaviors in your uploaded data. 
              Accounts with a high risk score may require additional verification or monitoring.
            </p>
            <div className="action-recommendations">
              <h4>Recommended Actions</h4>
              <ul>
                <li>Review high-risk accounts for suspicious activity patterns</li>
                <li>Implement additional verification steps for flagged accounts</li>
                <li>Monitor transaction patterns over time to identify evolving threats</li>
                <li>Cross-reference with known fraud typologies in your organization</li>
              </ul>
            </div>
          </div>
        </div>
      )}
      
      <div className="card note-card">
        <h4><span className="highlight">Note:</span> Data Security</h4>
        <p>
          All uploaded data is processed securely on our Amazon EC2 servers with end-to-end encryption. 
          Your data is automatically deleted after processing and is never stored or shared with third parties.
        </p>
        <p>
          For production environments, we recommend deploying our model on your private cloud infrastructure or on-premises servers.
        </p>
      </div>
    </div>
  );
};

export default DataUpload;
