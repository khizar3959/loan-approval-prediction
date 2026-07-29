"use client";

import React, { useState, useEffect } from "react";

interface PredictionResult {
  prediction: "Approved" | "Rejected";
  probability: number;
  confidence: number;
  features_received: Record<string, string | number | null>;
}

export default function Home() {
  // Form input state
  const [formData, setFormData] = useState({
    Gender: "Male",
    Married: "No",
    Dependents: "0",
    Education: "Graduate",
    Self_Employed: "No",
    ApplicantIncome: "",
    CoapplicantIncome: "",
    LoanAmount: "",
    Loan_Amount_Term: "360",
    Credit_History: "1.0",
    Property_Area: "Semiurban",
  });

  // UI state
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Dynamic calculations
  const [metrics, setMetrics] = useState({
    totalIncome: 0,
    estMonthlyPayment: 0,
    paymentToIncomeRatio: 0,
  });

  // Calculate metrics in real time as inputs change
  useEffect(() => {
    const appIncome = parseFloat(formData.ApplicantIncome) || 0;
    const coIncome = parseFloat(formData.CoapplicantIncome) || 0;
    const loanAmt = parseFloat(formData.LoanAmount) || 0;
    const term = parseFloat(formData.Loan_Amount_Term) || 360;

    const totalIncome = appIncome + coIncome;
    
    // Estimate simple interest loan monthly payment (e.g. assuming ~6% annual interest rate for UI display)
    // Monthly payment factor roughly = 0.006 per dollar for 30 years
    const annualInterestRate = 0.06;
    const monthlyRate = annualInterestRate / 12;
    let estMonthlyPayment = 0;
    
    if (totalIncome > 0 && loanAmt > 0 && term > 0) {
      const loanDollars = loanAmt * 1000;
      estMonthlyPayment = (loanDollars * monthlyRate * Math.pow(1 + monthlyRate, term)) / 
                          (Math.pow(1 + monthlyRate, term) - 1);
      
      // Fallback to simple division if formula gets NaN or Infinite
      if (!isFinite(estMonthlyPayment)) {
        estMonthlyPayment = loanDollars / term;
      }
    }

    const ratio = totalIncome > 0 ? (estMonthlyPayment / totalIncome) * 100 : 0;

    setMetrics({
      totalIncome,
      estMonthlyPayment: Math.round(estMonthlyPayment),
      paymentToIncomeRatio: parseFloat(ratio.toFixed(1)),
    });
  }, [formData]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    // Prepare numerical values
    const payload = {
      ...formData,
      ApplicantIncome: formData.ApplicantIncome === "" ? null : parseFloat(formData.ApplicantIncome),
      CoapplicantIncome: formData.CoapplicantIncome === "" ? null : parseFloat(formData.CoapplicantIncome),
      LoanAmount: formData.LoanAmount === "" ? null : parseFloat(formData.LoanAmount),
      Loan_Amount_Term: formData.Loan_Amount_Term === "" ? null : parseFloat(formData.Loan_Amount_Term),
      Credit_History: formData.Credit_History === "" ? null : parseFloat(formData.Credit_History),
    };

    try {
      const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiBaseUrl}/api/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errDetail = await response.json();
        throw new Error(errDetail.detail || "Server error occurred");
      }

      const data = await response.json();
      setResult(data);
    } catch (err: unknown) {
      console.error(err);
      if (err instanceof Error) {
        setError(err.message || "Failed to connect to the backend server. Make sure your FastAPI backend is running.");
      } else {
        setError("Failed to connect to the backend server. Make sure your FastAPI backend is running.");
      }
    } finally {
      setLoading(false);
    }
  };

  // Circular progress calculations for confidence gauge
  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const confidencePercent = result ? Math.round(result.confidence * 100) : 0;
  const strokeDashoffset = circumference - (confidencePercent / 100) * circumference;

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <span className="app-title-badge">Machine Learning Underwriting</span>
        <h1 className="app-title">Loan Approval Predictor</h1>
        <p className="app-subtitle">
          Input applicant financials and demographics to predict loan approval using an optimized K-Nearest Neighbors (KNN) model.
        </p>
      </header>

      {/* Main Grid Layout */}
      <main className="app-grid">
        {/* Form Card */}
        <section className="glass-card primary-glow">
          <h2 className="card-title">
            <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
            </svg>
            Applicant Profiles & Financials
          </h2>

          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              {/* Gender */}
              <div className="form-group">
                <label className="form-label" htmlFor="Gender">Gender</label>
                <select
                  id="Gender"
                  name="Gender"
                  value={formData.Gender}
                  onChange={handleChange}
                  className="form-select"
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>

              {/* Married */}
              <div className="form-group">
                <label className="form-label" htmlFor="Married">Married</label>
                <select
                  id="Married"
                  name="Married"
                  value={formData.Married}
                  onChange={handleChange}
                  className="form-select"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>

              {/* Dependents */}
              <div className="form-group">
                <label className="form-label" htmlFor="Dependents">Dependents</label>
                <select
                  id="Dependents"
                  name="Dependents"
                  value={formData.Dependents}
                  onChange={handleChange}
                  className="form-select"
                >
                  <option value="0">0</option>
                  <option value="1">1</option>
                  <option value="2">2</option>
                  <option value="3+">3+</option>
                </select>
              </div>

              {/* Education */}
              <div className="form-group">
                <label className="form-label" htmlFor="Education">Education</label>
                <select
                  id="Education"
                  name="Education"
                  value={formData.Education}
                  onChange={handleChange}
                  className="form-select"
                >
                  <option value="Graduate">Graduate</option>
                  <option value="Not Graduate">Not Graduate</option>
                </select>
              </div>

              {/* Self Employed */}
              <div className="form-group">
                <label className="form-label" htmlFor="Self_Employed">Self Employed</label>
                <select
                  id="Self_Employed"
                  name="Self_Employed"
                  value={formData.Self_Employed}
                  onChange={handleChange}
                  className="form-select"
                >
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>

              {/* Property Area */}
              <div className="form-group">
                <label className="form-label" htmlFor="Property_Area">Property Area</label>
                <select
                  id="Property_Area"
                  name="Property_Area"
                  value={formData.Property_Area}
                  onChange={handleChange}
                  className="form-select"
                >
                  <option value="Rural">Rural</option>
                  <option value="Semiurban">Semiurban</option>
                  <option value="Urban">Urban</option>
                </select>
              </div>

              {/* Applicant Income */}
              <div className="form-group">
                <label className="form-label" htmlFor="ApplicantIncome">Applicant Income ($ / month)</label>
                <div className="input-wrapper">
                  <input
                    id="ApplicantIncome"
                    type="number"
                    name="ApplicantIncome"
                    value={formData.ApplicantIncome}
                    onChange={handleChange}
                    placeholder="e.g. 5000"
                    required
                    min="0"
                    className="form-input"
                  />
                </div>
              </div>

              {/* Coapplicant Income */}
              <div className="form-group">
                <label className="form-label" htmlFor="CoapplicantIncome">Coapplicant Income ($ / month)</label>
                <div className="input-wrapper">
                  <input
                    id="CoapplicantIncome"
                    type="number"
                    name="CoapplicantIncome"
                    value={formData.CoapplicantIncome}
                    onChange={handleChange}
                    placeholder="e.g. 1500 (0 if none)"
                    required
                    min="0"
                    className="form-input"
                  />
                </div>
              </div>

              {/* Loan Amount */}
              <div className="form-group">
                <label className="form-label" htmlFor="LoanAmount">Loan Amount ($ in Thousands)</label>
                <div className="input-wrapper">
                  <input
                    id="LoanAmount"
                    type="number"
                    name="LoanAmount"
                    value={formData.LoanAmount}
                    onChange={handleChange}
                    placeholder="e.g. 150 (for $150,000)"
                    required
                    min="1"
                    className="form-input"
                  />
                </div>
              </div>

              {/* Loan Amount Term */}
              <div className="form-group">
                <label className="form-label" htmlFor="Loan_Amount_Term">Loan Term (Months)</label>
                <select
                  id="Loan_Amount_Term"
                  name="Loan_Amount_Term"
                  value={formData.Loan_Amount_Term}
                  onChange={handleChange}
                  className="form-select"
                >
                  <option value="120">120 (10 Years)</option>
                  <option value="180">180 (15 Years)</option>
                  <option value="240">240 (20 Years)</option>
                  <option value="300">300 (25 Years)</option>
                  <option value="360">360 (30 Years)</option>
                  <option value="480">480 (40 Years)</option>
                </select>
              </div>

              {/* Credit History */}
              <div className="form-group full-width">
                <label className="form-label" htmlFor="Credit_History">Credit History Status</label>
                <select
                  id="Credit_History"
                  name="Credit_History"
                  value={formData.Credit_History}
                  onChange={handleChange}
                  className="form-select"
                >
                  <option value="1.0">Good Credit History (No past defaults, meets guidelines)</option>
                  <option value="0.0">Poor Credit History (Past default history or does not meet guidelines)</option>
                </select>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="submit-btn"
            >
              {loading ? (
                <>
                  <span className="spinner" style={{ width: "20px", height: "20px", borderWidth: "2px" }}></span>
                  Underwriting Analysis...
                </>
              ) : (
                <>
                  Evaluate Eligibility
                  <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                  </svg>
                </>
              )}
            </button>
          </form>
        </section>

        {/* Results Panel */}
        <section className="glass-card">
          <div className="result-card">
            <h2 className="card-title">
              <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
              </svg>
              Real-Time Evaluation Metrics
            </h2>

            {/* Helper metrics generated locally */}
            <div className="metrics-section">
              <div className="metric-row">
                <div className="metric-info">
                  <span>Total Monthly Income</span>
                  <span className="metric-value">${metrics.totalIncome.toLocaleString()}</span>
                </div>
                <div className="metric-bar-container">
                  <div 
                    className="metric-bar" 
                    style={{ width: `${Math.min(100, (metrics.totalIncome / 15000) * 100)}%` }}
                  ></div>
                </div>
              </div>

              <div className="metric-row">
                <div className="metric-info">
                  <span>Est. Monthly Principal & Interest</span>
                  <span className="metric-value">${metrics.estMonthlyPayment.toLocaleString()}</span>
                </div>
                <div className="metric-bar-container">
                  <div 
                    className="metric-bar" 
                    style={{ 
                      width: `${Math.min(100, (metrics.estMonthlyPayment / 5000) * 100)}%`,
                      backgroundColor: "#818cf8" 
                    }}
                  ></div>
                </div>
              </div>

              <div className="metric-row">
                <div className="metric-info">
                  <span>Debt-to-Income (DTI) Ratio</span>
                  <span className={`metric-value ${metrics.paymentToIncomeRatio > 40 ? 'color-rose' : ''}`}>
                    {metrics.paymentToIncomeRatio}%
                  </span>
                </div>
                <div className="metric-bar-container">
                  <div 
                    className={`metric-bar ${metrics.paymentToIncomeRatio > 40 ? 'rose' : 'emerald'}`}
                    style={{ width: `${Math.min(100, (metrics.paymentToIncomeRatio / 60) * 100)}%` }}
                  ></div>
                </div>
              </div>
            </div>

            {/* Dynamic result representation */}
            {loading && (
              <div className="spinner-wrapper">
                <div className="spinner"></div>
                <div className="loading-text">KNN Classifying Profile...</div>
              </div>
            )}

            {error && (
              <div className="prediction-box rejected" style={{ borderColor: "rgba(239, 68, 68, 0.3)" }}>
                <span className="badge rejected">Connection Error</span>
                <p className="pred-description" style={{ color: "#fca5a5" }}>
                  {error}
                </p>
                <div style={{ marginTop: "1rem", textAlign: "left", fontSize: "0.8rem", color: "rgba(255,255,255,0.4)" }}>
                  <p><strong>To start the FastAPI backend:</strong></p>
                  <ol style={{ paddingLeft: "1.2rem", marginTop: "0.4rem" }}>
                    <li>Open a terminal in the project directory.</li>
                    <li>Activate the virtual environment: <code>.\.venv\Scripts\activate</code></li>
                    <li>Run the training script (first time): <code>python backend/train_model.py</code></li>
                    <li>Start the API server: <code>uvicorn backend.main:app --reload</code></li>
                  </ol>
                </div>
              </div>
            )}

            {!loading && !error && !result && (
              <div className="result-placeholder">
                <svg className="result-placeholder-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path>
                </svg>
                <p>Awaiting submission. Enter applicant details and click <strong>Evaluate Eligibility</strong> to execute prediction.</p>
              </div>
            )}

            {!loading && !error && result && (
              <div className={`prediction-box ${result.prediction === "Approved" ? "approved" : "rejected"}`}>
                <span className={`badge ${result.prediction === "Approved" ? "approved" : "rejected"}`}>
                  {result.prediction}
                </span>

                <div className="gauge-container">
                  <svg className="gauge-svg">
                    <circle className="gauge-bg" cx="70" cy="70" r={radius}></circle>
                    <circle 
                      className={`gauge-fill ${result.prediction === "Approved" ? "approved" : "rejected"}`} 
                      cx="70" 
                      cy="70" 
                      r={radius}
                      strokeDasharray={circumference}
                      strokeDashoffset={strokeDashoffset}
                    ></circle>
                  </svg>
                  <div className="gauge-text">
                    <span className="gauge-percentage">{confidencePercent}%</span>
                    <span className="gauge-label">Confidence</span>
                  </div>
                </div>

                <p className="pred-description">
                  {result.prediction === "Approved" ? (
                    <>
                      The applicant profile matches underwriting metrics for <strong>loan approval</strong>. 
                      Model exhibits a high similarity score with approved historical loans based on credit history and stable DTI ratio.
                    </>
                  ) : (
                    <>
                      The applicant profile matches metrics for <strong>loan rejection</strong>. 
                      Model shows historical alignment with defaults or higher-risk profiles, typically caused by a poor credit history or high DTI.
                    </>
                  )}
                </p>
              </div>
            )}
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>Built with Next.js & FastAPI | KNN Model Accuracy: 78.86% (GridSearchCV tuned)</p>
      </footer>
    </div>
  );
}
