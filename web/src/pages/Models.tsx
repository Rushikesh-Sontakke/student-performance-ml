import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Brain, 
  ChevronLeft, 
  TrendingUp, 
  Layers, 
  TreeDeciduous, 
  Zap,
  Combine,
  ChevronDown,
  CheckCircle2,
  Settings2,
  BarChart3
} from 'lucide-react'
import { Link } from 'react-router-dom'
import './Models.css'

interface ModelConfig {
  id: string
  name: string
  icon: React.ReactNode
  color: string
  description: string
  parameters: { name: string; value: string }[]
  architecture?: string[]
  performance: {
    r2: number
    rmse: number
    mae: number
    mape: number
  }
  highlights: string[]
}

const models: ModelConfig[] = [
  {
    id: 'ensemble',
    name: 'Ensemble Model',
    icon: <Combine size={24} />,
    color: '#00d4aa',
    description: 'Weighted combination of all models for optimal performance',
    parameters: [
      { name: 'Linear Regression Weight', value: '90.3%' },
      { name: 'Random Forest Weight', value: '9.1%' },
      { name: 'XGBoost Weight', value: '0.5%' },
      { name: 'Neural Network Weight', value: '0.0%' },
    ],
    performance: {
      r2: 0.7705,
      rmse: 1.8011,
      mae: 0.4582,
      mape: 0.6380,
    },
    highlights: [
      'Best overall performance',
      'Combines strengths of multiple models',
      'Optimal weights learned from validation',
    ],
  },
  {
    id: 'linear-regression',
    name: 'Linear Regression',
    icon: <TrendingUp size={24} />,
    color: '#7c3aed',
    description: 'Simple yet effective baseline model using ordinary least squares',
    parameters: [
      { name: 'Algorithm', value: 'Ordinary Least Squares' },
      { name: 'Fit Intercept', value: 'True' },
      { name: 'Features', value: '20' },
      { name: 'Regularization', value: 'None' },
    ],
    performance: {
      r2: 0.7702,
      rmse: 1.8024,
      mae: 0.4486,
      mape: 0.6224,
    },
    highlights: [
      'Most interpretable model',
      'Fastest training time',
      'Lowest MAE among individual models',
    ],
  },
  {
    id: 'neural-network',
    name: 'Neural Network',
    icon: <Layers size={24} />,
    color: '#ec4899',
    description: 'Deep learning model with multiple hidden layers and batch normalization',
    parameters: [
      { name: 'Optimizer', value: 'Adam (lr=0.001)' },
      { name: 'Loss Function', value: 'MSE' },
      { name: 'Regularization', value: 'L2 (0.001)' },
      { name: 'Epochs', value: '300' },
      { name: 'Batch Size', value: '16' },
      { name: 'Early Stopping', value: 'patience=20' },
    ],
    architecture: [
      'Input Layer (20 features)',
      'Dense 256 + BatchNorm + ReLU',
      'Dense 128 + BatchNorm + ReLU',
      'Dense 64 + BatchNorm + ReLU',
      'Dense 32 + BatchNorm + ReLU',
      'Output Layer (1)',
    ],
    performance: {
      r2: 0.7547,
      rmse: 1.8622,
      mae: 0.5938,
      mape: 0.8373,
    },
    highlights: [
      'Can capture non-linear patterns',
      'Uses batch normalization',
      'Learning rate scheduling',
    ],
  },
  {
    id: 'xgboost',
    name: 'XGBoost',
    icon: <Zap size={24} />,
    color: '#f59e0b',
    description: 'Gradient boosting with regularization for robust predictions',
    parameters: [
      { name: 'n_estimators', value: '200' },
      { name: 'max_depth', value: '2' },
      { name: 'learning_rate', value: '0.15' },
      { name: 'min_child_weight', value: '3' },
      { name: 'subsample', value: '0.8' },
      { name: 'colsample_bytree', value: '0.8' },
      { name: 'reg_alpha (L1)', value: '0.5' },
      { name: 'reg_lambda (L2)', value: '2' },
    ],
    performance: {
      r2: 0.7036,
      rmse: 2.0469,
      mae: 0.8443,
      mape: 1.2136,
    },
    highlights: [
      'Hyperparameter tuned via GridSearchCV',
      'Strong regularization to prevent overfitting',
      'Feature importance analysis available',
    ],
  },
  {
    id: 'random-forest',
    name: 'Random Forest',
    icon: <TreeDeciduous size={24} />,
    color: '#22c55e',
    description: 'Ensemble of decision trees with Box-Cox transformation',
    parameters: [
      { name: 'n_estimators', value: '300' },
      { name: 'max_depth', value: '30' },
      { name: 'min_samples_split', value: '2' },
      { name: 'min_samples_leaf', value: '2' },
      { name: 'max_features', value: 'None (all)' },
      { name: 'Target Transform', value: 'Box-Cox (PowerTransformer)' },
    ],
    performance: {
      r2: 0.6869,
      rmse: 2.1038,
      mae: 1.0465,
      mape: 1.5216,
    },
    highlights: [
      'Tuned via RandomizedSearchCV',
      'Uses PowerTransformer on target',
      'Provides feature importance',
    ],
  },
]

const Models = () => {
  const [selectedModel, setSelectedModel] = useState<string>('ensemble')
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    parameters: true,
    architecture: true,
    performance: true,
  })

  const currentModel = models.find((m) => m.id === selectedModel) || models[0]

  const toggleSection = (section: string) => {
    setExpandedSections((prev) => ({ ...prev, [section]: !prev[section] }))
  }

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1 },
    },
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 },
  }

  return (
    <div className="models-page">
      {/* Background */}
      <div className="background">
        <div className="gradient-orb orb-1" style={{ background: currentModel.color }} />
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
          <Link to="/models" className="nav-link active">Models</Link>
          <Link to="/predict" className="nav-link">Predict</Link>
          <Link to="/report" className="nav-link">Report</Link>
        </div>
      </nav>

      {/* Main Content */}
      <main className="models-content">
        <motion.div
          className="page-header"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h1>Explore Models</h1>
          <p>Compare different machine learning models and their configurations</p>
        </motion.div>

        <div className="models-layout">
          {/* Model Selector */}
          <motion.div
            className="model-selector"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
          >
            {models.map((model) => (
              <motion.button
                key={model.id}
                className={`model-card ${selectedModel === model.id ? 'active' : ''}`}
                onClick={() => setSelectedModel(model.id)}
                variants={itemVariants}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                style={{
                  '--model-color': model.color,
                } as React.CSSProperties}
              >
                <div className="model-card-icon" style={{ color: model.color }}>
                  {model.icon}
                </div>
                <div className="model-card-info">
                  <h3>{model.name}</h3>
                  <span className="model-r2">R² = {model.performance.r2.toFixed(4)}</span>
                </div>
                {selectedModel === model.id && (
                  <CheckCircle2 className="model-check" size={18} />
                )}
              </motion.button>
            ))}
          </motion.div>

          {/* Model Details */}
          <AnimatePresence mode="wait">
            <motion.div
              key={currentModel.id}
              className="model-details"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
            >
              {/* Header */}
              <div className="details-header" style={{ '--model-color': currentModel.color } as React.CSSProperties}>
                <div className="details-icon" style={{ background: currentModel.color }}>
                  {currentModel.icon}
                </div>
                <div>
                  <h2>{currentModel.name}</h2>
                  <p>{currentModel.description}</p>
                </div>
              </div>

              {/* Highlights */}
              <div className="highlights">
                {currentModel.highlights.map((highlight, index) => (
                  <span key={index} className="highlight-tag">
                    {highlight}
                  </span>
                ))}
              </div>

              {/* Parameters Section */}
              <div className="details-section">
                <button
                  className="section-header"
                  onClick={() => toggleSection('parameters')}
                >
                  <Settings2 size={18} />
                  <span>Model Parameters</span>
                  <ChevronDown
                    size={18}
                    className={`chevron ${expandedSections.parameters ? 'expanded' : ''}`}
                  />
                </button>
                <AnimatePresence>
                  {expandedSections.parameters && (
                    <motion.div
                      className="section-content"
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                    >
                      <div className="params-grid">
                        {currentModel.parameters.map((param, index) => (
                          <div key={index} className="param-item">
                            <span className="param-name">{param.name}</span>
                            <span className="param-value">{param.value}</span>
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Architecture Section (for Neural Network) */}
              {currentModel.architecture && (
                <div className="details-section">
                  <button
                    className="section-header"
                    onClick={() => toggleSection('architecture')}
                  >
                    <Layers size={18} />
                    <span>Network Architecture</span>
                    <ChevronDown
                      size={18}
                      className={`chevron ${expandedSections.architecture ? 'expanded' : ''}`}
                    />
                  </button>
                  <AnimatePresence>
                    {expandedSections.architecture && (
                      <motion.div
                        className="section-content"
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                      >
                        <div className="architecture-diagram">
                          {currentModel.architecture.map((layer, index) => (
                            <div key={index} className="layer-item">
                              <div className="layer-connector" />
                              <div className="layer-box">{layer}</div>
                            </div>
                          ))}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              )}

              {/* Performance Section */}
              <div className="details-section">
                <button
                  className="section-header"
                  onClick={() => toggleSection('performance')}
                >
                  <BarChart3 size={18} />
                  <span>Performance Metrics</span>
                  <ChevronDown
                    size={18}
                    className={`chevron ${expandedSections.performance ? 'expanded' : ''}`}
                  />
                </button>
                <AnimatePresence>
                  {expandedSections.performance && (
                    <motion.div
                      className="section-content"
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                    >
                      <div className="metrics-grid">
                        <div className="metric-card">
                          <span className="metric-label">R² Score</span>
                          <span className="metric-value" style={{ color: currentModel.color }}>
                            {currentModel.performance.r2.toFixed(4)}
                          </span>
                          <div className="metric-bar">
                            <div
                              className="metric-fill"
                              style={{
                                width: `${currentModel.performance.r2 * 100}%`,
                                background: currentModel.color,
                              }}
                            />
                          </div>
                        </div>
                        <div className="metric-card">
                          <span className="metric-label">RMSE</span>
                          <span className="metric-value">{currentModel.performance.rmse.toFixed(4)}</span>
                          <span className="metric-note">Lower is better</span>
                        </div>
                        <div className="metric-card">
                          <span className="metric-label">MAE</span>
                          <span className="metric-value">{currentModel.performance.mae.toFixed(4)}</span>
                          <span className="metric-note">Lower is better</span>
                        </div>
                        <div className="metric-card">
                          <span className="metric-label">MAPE (%)</span>
                          <span className="metric-value">{currentModel.performance.mape.toFixed(4)}</span>
                          <span className="metric-note">Lower is better</span>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
    </div>
  )
}

export default Models

