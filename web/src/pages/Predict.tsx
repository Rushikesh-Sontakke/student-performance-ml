import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Brain,
  ChevronLeft,
  GraduationCap,
  Users,
  School,
  User,
  Sparkles,
  Loader2,
  RefreshCw,
  TrendingUp,
  CheckCircle2,
  AlertCircle,
  BookOpen,
  Activity,
} from 'lucide-react'
import { Link } from 'react-router-dom'
import { getPrediction, inputConfig, StudentInput, PredictionResult } from '../services/predictor'
import './Predict.css'

const Predict = () => {
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [formData, setFormData] = useState<StudentInput>({
    hours_studied: inputConfig.hours_studied.default,
    attendance: inputConfig.attendance.default,
    previous_scores: inputConfig.previous_scores.default,
    parental_involvement: inputConfig.parental_involvement.default,
    access_to_resources: inputConfig.access_to_resources.default,
    tutoring_sessions: inputConfig.tutoring_sessions.default,
    teacher_quality: inputConfig.teacher_quality.default,
    school_type: inputConfig.school_type.default,
    peer_influence: inputConfig.peer_influence.default,
    sleep_hours: inputConfig.sleep_hours.default,
    physical_activity: inputConfig.physical_activity.default,
    motivation_level: inputConfig.motivation_level.default,
    gender: inputConfig.gender.default,
    family_income: inputConfig.family_income.default,
    parental_education: inputConfig.parental_education.default,
    distance_from_home: inputConfig.distance_from_home.default,
    extracurricular: inputConfig.extracurricular.default,
    internet_access: inputConfig.internet_access.default,
    learning_disabilities: inputConfig.learning_disabilities.default,
  })

  const handleInputChange = (field: keyof StudentInput, value: string | number) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setResult(null)

    try {
      const prediction = await getPrediction(formData)
      setResult(prediction)
    } catch (error) {
      console.error('Prediction failed:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleReset = () => {
    setResult(null)
    setFormData({
      hours_studied: inputConfig.hours_studied.default,
      attendance: inputConfig.attendance.default,
      previous_scores: inputConfig.previous_scores.default,
      parental_involvement: inputConfig.parental_involvement.default,
      access_to_resources: inputConfig.access_to_resources.default,
      tutoring_sessions: inputConfig.tutoring_sessions.default,
      teacher_quality: inputConfig.teacher_quality.default,
      school_type: inputConfig.school_type.default,
      peer_influence: inputConfig.peer_influence.default,
      sleep_hours: inputConfig.sleep_hours.default,
      physical_activity: inputConfig.physical_activity.default,
      motivation_level: inputConfig.motivation_level.default,
      gender: inputConfig.gender.default,
      family_income: inputConfig.family_income.default,
      parental_education: inputConfig.parental_education.default,
      distance_from_home: inputConfig.distance_from_home.default,
      extracurricular: inputConfig.extracurricular.default,
      internet_access: inputConfig.internet_access.default,
      learning_disabilities: inputConfig.learning_disabilities.default,
    })
  }

  const getScoreColor = (score: number) => {
    if (score >= 80) return '#22c55e'
    if (score >= 70) return '#00d4aa'
    if (score >= 60) return '#f59e0b'
    return '#ef4444'
  }

  const getScoreLabel = (score: number) => {
    if (score >= 80) return 'Excellent'
    if (score >= 70) return 'Good'
    if (score >= 60) return 'Average'
    return 'Needs Improvement'
  }

  return (
    <div className="predict-page">
      {/* Background */}
      <div className="background">
        <div className="gradient-orb orb-1" />
        <div className="gradient-orb orb-2" />
        <div className="grid-overlay" />
      </div>

      {/* Navigation */}
      <nav className="nav">
        <Link to="/" className="nav-back">
          <ChevronLeft size={20} />
          <span>Back</span>
        </Link>
        <div className="nav-logo">
          <Brain className="nav-icon" />
          <span>SPP</span>
        </div>
        <div className="nav-links">
          <Link to="/" className="nav-link">Home</Link>
          <Link to="/models" className="nav-link">Models</Link>
          <Link to="/predict" className="nav-link active">Predict</Link>
          <Link to="/report" className="nav-link">Report</Link>
        </div>
      </nav>

      {/* Main Content */}
      <main className="predict-content">
        <motion.div
          className="page-header"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h1>
            <GraduationCap className="header-icon" />
            Predict Student Performance
          </h1>
          <p>Enter student characteristics to predict exam scores using our ML models</p>
        </motion.div>

        <div className="predict-layout">
          {/* Input Form */}
          <motion.form
            className="input-form"
            onSubmit={handleSubmit}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            {/* Academic Section */}
            <div className="form-section">
              <div className="section-title">
                <BookOpen size={18} />
                <span>Academic Information</span>
              </div>
              <div className="form-grid">
                <div className="form-field">
                  <label>{inputConfig.hours_studied.label}</label>
                  <input
                    type="number"
                    min={inputConfig.hours_studied.min}
                    max={inputConfig.hours_studied.max}
                    value={formData.hours_studied}
                    onChange={(e) => handleInputChange('hours_studied', parseInt(e.target.value) || 0)}
                  />
                </div>
                <div className="form-field">
                  <label>{inputConfig.attendance.label}</label>
                  <input
                    type="number"
                    min={inputConfig.attendance.min}
                    max={inputConfig.attendance.max}
                    value={formData.attendance}
                    onChange={(e) => handleInputChange('attendance', parseInt(e.target.value) || 0)}
                  />
                </div>
                <div className="form-field">
                  <label>{inputConfig.previous_scores.label}</label>
                  <input
                    type="number"
                    min={inputConfig.previous_scores.min}
                    max={inputConfig.previous_scores.max}
                    value={formData.previous_scores}
                    onChange={(e) => handleInputChange('previous_scores', parseInt(e.target.value) || 0)}
                  />
                </div>
                <div className="form-field">
                  <label>{inputConfig.tutoring_sessions.label}</label>
                  <input
                    type="number"
                    min={inputConfig.tutoring_sessions.min}
                    max={inputConfig.tutoring_sessions.max}
                    value={formData.tutoring_sessions}
                    onChange={(e) => handleInputChange('tutoring_sessions', parseInt(e.target.value) || 0)}
                  />
                </div>
              </div>
            </div>

            {/* Support Section */}
            <div className="form-section">
              <div className="section-title">
                <Users size={18} />
                <span>Support & Resources</span>
              </div>
              <div className="form-grid">
                <div className="form-field">
                  <label>{inputConfig.parental_involvement.label}</label>
                  <select
                    value={formData.parental_involvement}
                    onChange={(e) => handleInputChange('parental_involvement', e.target.value)}
                  >
                    {inputConfig.parental_involvement.options.map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </div>
                <div className="form-field">
                  <label>{inputConfig.access_to_resources.label}</label>
                  <select
                    value={formData.access_to_resources}
                    onChange={(e) => handleInputChange('access_to_resources', e.target.value)}
                  >
                    {inputConfig.access_to_resources.options.map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </div>
                <div className="form-field">
                  <label>{inputConfig.family_income.label}</label>
                  <select
                    value={formData.family_income}
                    onChange={(e) => handleInputChange('family_income', e.target.value)}
                  >
                    {inputConfig.family_income.options.map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </div>
                <div className="form-field">
                  <label>{inputConfig.parental_education.label}</label>
                  <select
                    value={formData.parental_education}
                    onChange={(e) => handleInputChange('parental_education', e.target.value)}
                  >
                    {inputConfig.parental_education.options.map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            {/* School Section */}
            <div className="form-section">
              <div className="section-title">
                <School size={18} />
                <span>School Environment</span>
              </div>
              <div className="form-grid">
                <div className="form-field">
                  <label>{inputConfig.teacher_quality.label}</label>
                  <select
                    value={formData.teacher_quality}
                    onChange={(e) => handleInputChange('teacher_quality', e.target.value)}
                  >
                    {inputConfig.teacher_quality.options.map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </div>
                <div className="form-field">
                  <label>{inputConfig.school_type.label}</label>
                  <select
                    value={formData.school_type}
                    onChange={(e) => handleInputChange('school_type', e.target.value)}
                  >
                    {inputConfig.school_type.options.map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </div>
                <div className="form-field">
                  <label>{inputConfig.peer_influence.label}</label>
                  <select
                    value={formData.peer_influence}
                    onChange={(e) => handleInputChange('peer_influence', e.target.value)}
                  >
                    {inputConfig.peer_influence.options.map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </div>
                <div className="form-field">
                  <label>{inputConfig.distance_from_home.label}</label>
                  <select
                    value={formData.distance_from_home}
                    onChange={(e) => handleInputChange('distance_from_home', e.target.value)}
                  >
                    {inputConfig.distance_from_home.options.map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            {/* Personal Section */}
            <div className="form-section">
              <div className="section-title">
                <User size={18} />
                <span>Personal Factors</span>
              </div>
              <div className="form-grid">
                <div className="form-field">
                  <label>{inputConfig.sleep_hours.label}</label>
                  <input
                    type="number"
                    min={inputConfig.sleep_hours.min}
                    max={inputConfig.sleep_hours.max}
                    value={formData.sleep_hours}
                    onChange={(e) => handleInputChange('sleep_hours', parseInt(e.target.value) || 0)}
                  />
                </div>
                <div className="form-field">
                  <label>{inputConfig.physical_activity.label}</label>
                  <input
                    type="number"
                    min={inputConfig.physical_activity.min}
                    max={inputConfig.physical_activity.max}
                    value={formData.physical_activity}
                    onChange={(e) => handleInputChange('physical_activity', parseInt(e.target.value) || 0)}
                  />
                </div>
                <div className="form-field">
                  <label>{inputConfig.motivation_level.label}</label>
                  <select
                    value={formData.motivation_level}
                    onChange={(e) => handleInputChange('motivation_level', e.target.value)}
                  >
                    {inputConfig.motivation_level.options.map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </div>
                <div className="form-field">
                  <label>{inputConfig.gender.label}</label>
                  <select
                    value={formData.gender}
                    onChange={(e) => handleInputChange('gender', e.target.value)}
                  >
                    {inputConfig.gender.options.map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            {/* Additional Section */}
            <div className="form-section">
              <div className="section-title">
                <Activity size={18} />
                <span>Additional Factors</span>
              </div>
              <div className="form-grid">
                <div className="form-field">
                  <label>{inputConfig.extracurricular.label}</label>
                  <select
                    value={formData.extracurricular}
                    onChange={(e) => handleInputChange('extracurricular', e.target.value)}
                  >
                    {inputConfig.extracurricular.options.map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </div>
                <div className="form-field">
                  <label>{inputConfig.internet_access.label}</label>
                  <select
                    value={formData.internet_access}
                    onChange={(e) => handleInputChange('internet_access', e.target.value)}
                  >
                    {inputConfig.internet_access.options.map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </div>
                <div className="form-field">
                  <label>{inputConfig.learning_disabilities.label}</label>
                  <select
                    value={formData.learning_disabilities}
                    onChange={(e) => handleInputChange('learning_disabilities', e.target.value)}
                  >
                    {inputConfig.learning_disabilities.options.map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            {/* Submit Button */}
            <div className="form-actions">
              <button type="button" className="btn-reset" onClick={handleReset}>
                <RefreshCw size={18} />
                Reset
              </button>
              <button type="submit" className="btn-predict" disabled={isLoading}>
                {isLoading ? (
                  <>
                    <Loader2 size={18} className="spinning" />
                    Predicting...
                  </>
                ) : (
                  <>
                    <Sparkles size={18} />
                    Predict Score
                  </>
                )}
              </button>
            </div>
          </motion.form>

          {/* Results Panel */}
          <motion.div
            className="results-panel"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <AnimatePresence mode="wait">
              {!result && !isLoading ? (
                <motion.div
                  key="empty"
                  className="results-empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <div className="empty-icon">
                    <TrendingUp size={48} />
                  </div>
                  <h3>Ready to Predict</h3>
                  <p>Fill in the student characteristics and click "Predict Score" to see the results.</p>
                </motion.div>
              ) : isLoading ? (
                <motion.div
                  key="loading"
                  className="results-loading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <Loader2 size={48} className="spinning" />
                  <p>Analyzing student profile...</p>
                </motion.div>
              ) : result ? (
                <motion.div
                  key="results"
                  className="results-content"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0 }}
                >
                  {/* Main Score */}
                  <div className="main-score">
                    <div className="score-label">Ensemble Prediction</div>
                    <div
                      className="score-value"
                      style={{ color: getScoreColor(result.ensemble) }}
                    >
                      {result.ensemble.toFixed(1)}
                    </div>
                    <div
                      className="score-badge"
                      style={{ background: getScoreColor(result.ensemble) }}
                    >
                      {getScoreLabel(result.ensemble)}
                    </div>
                  </div>

                  {/* Model Predictions */}
                  <div className="model-predictions">
                    <h4>Predictions by Model</h4>
                    <div className="predictions-grid">
                      {[
                        { name: 'Linear Regression', score: result.linear_regression, color: '#7c3aed' },
                        { name: 'Neural Network', score: result.neural_network, color: '#ec4899' },
                        { name: 'XGBoost', score: result.xgboost, color: '#f59e0b' },
                        { name: 'Random Forest', score: result.random_forest, color: '#22c55e' },
                      ].map((model) => (
                        <div key={model.name} className="prediction-item">
                          <span className="pred-name">{model.name}</span>
                          <span className="pred-score" style={{ color: model.color }}>
                            {model.score.toFixed(1)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Analysis */}
                  <div className="analysis-section">
                    <h4>AI Analysis</h4>
                    <p className="analysis-text">{result.analysis}</p>
                  </div>

                  {/* Strengths & Improvements */}
                  <div className="feedback-grid">
                    <div className="feedback-card strengths">
                      <h4>
                        <CheckCircle2 size={18} />
                        Strengths
                      </h4>
                      <ul>
                        {result.strengths.map((s, i) => (
                          <li key={i}>{s}</li>
                        ))}
                      </ul>
                    </div>
                    <div className="feedback-card improvements">
                      <h4>
                        <AlertCircle size={18} />
                        Areas to Improve
                      </h4>
                      <ul>
                        {result.improvements.map((s, i) => (
                          <li key={i}>{s}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </motion.div>
              ) : null}
            </AnimatePresence>
          </motion.div>
        </div>
      </main>
    </div>
  )
}

export default Predict

