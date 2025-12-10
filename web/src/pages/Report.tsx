import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Brain,
  ChevronLeft,
  FileText,
  BarChart3,
  GraduationCap,
  Users,
  School,
  Sparkles,
  Loader2,
  RefreshCw,
  Copy,
  Check,
  Languages,
} from 'lucide-react'
import { Link } from 'react-router-dom'
import { generateReportStream, generateReport, modelData, datasetInfo, Language, languageLabels } from '../services/gemini'
import './Report.css'

interface ReportTab {
  id: string
  label: string
  icon: React.ReactNode
  description: string
}

const reportTabs: ReportTab[] = [
  {
    id: 'overall',
    label: 'Overall Report',
    icon: <FileText size={20} />,
    description: 'Comprehensive analysis of all models and findings',
  },
  {
    id: 'by-model',
    label: 'By Model',
    icon: <BarChart3 size={20} />,
    description: 'Detailed breakdown of each model\'s performance',
  },
  {
    id: 'students',
    label: 'For Students',
    icon: <GraduationCap size={20} />,
    description: 'Actionable recommendations to improve performance',
  },
  {
    id: 'parents',
    label: 'For Parents',
    icon: <Users size={20} />,
    description: 'How parents can support academic success',
  },
  {
    id: 'teaching',
    label: 'For Educators',
    icon: <School size={20} />,
    description: 'Evidence-based strategies for teaching units',
  },
]

const Report = () => {
  const [activeTab, setActiveTab] = useState('overall')
  const [isGenerating, setIsGenerating] = useState(false)
  const [reportContent, setReportContent] = useState<Record<string, string>>({})
  const [copied, setCopied] = useState(false)
  const [language, setLanguage] = useState<Language>('en')
  const contentRef = useRef<HTMLDivElement>(null)

  const currentTab = reportTabs.find((t) => t.id === activeTab) || reportTabs[0]

  useEffect(() => {
    if (contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight
    }
  }, [reportContent[activeTab]])

  const handleGenerate = async () => {
    setIsGenerating(true)
    const cacheKey = `${activeTab}-${language}`
    setReportContent((prev) => ({ ...prev, [cacheKey]: '' }))

    try {
      // Try streaming first
      let fullText = ''
      try {
        for await (const chunk of generateReportStream(activeTab, language)) {
          fullText += chunk
          setReportContent((prev) => ({ ...prev, [cacheKey]: fullText }))
        }
      } catch {
        // Fallback to non-streaming
        console.log('Streaming failed, using fallback...')
        const result = await generateReport(activeTab, language)
        setReportContent((prev) => ({ ...prev, [cacheKey]: result }))
      }
    } catch (error) {
      console.error('Report generation failed:', error)
      const errorMsg = language === 'zh-TW' 
        ? '❌ 報告生成失敗，請重試。'
        : '❌ Failed to generate report. Please try again.'
      setReportContent((prev) => ({
        ...prev,
        [cacheKey]: errorMsg,
      }))
    } finally {
      setIsGenerating(false)
    }
  }

  const currentCacheKey = `${activeTab}-${language}`

  const handleCopy = async () => {
    const content = reportContent[currentCacheKey]
    if (content) {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  const renderMarkdown = (text: string) => {
    // Simple markdown rendering
    return text
      .split('\n')
      .map((line, index) => {
        // Headers
        if (line.startsWith('### ')) {
          return (
            <h3 key={index} className="md-h3">
              {line.slice(4)}
            </h3>
          )
        }
        if (line.startsWith('## ')) {
          return (
            <h2 key={index} className="md-h2">
              {line.slice(3)}
            </h2>
          )
        }
        if (line.startsWith('# ')) {
          return (
            <h1 key={index} className="md-h1">
              {line.slice(2)}
            </h1>
          )
        }
        // Bold
        const boldProcessed = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Lists
        if (line.startsWith('- ') || line.startsWith('* ')) {
          return (
            <li
              key={index}
              className="md-li"
              dangerouslySetInnerHTML={{ __html: boldProcessed.slice(2) }}
            />
          )
        }
        if (/^\d+\.\s/.test(line)) {
          return (
            <li
              key={index}
              className="md-li numbered"
              dangerouslySetInnerHTML={{ __html: boldProcessed.replace(/^\d+\.\s/, '') }}
            />
          )
        }
        // Empty line
        if (line.trim() === '') {
          return <br key={index} />
        }
        // Regular paragraph
        return (
          <p
            key={index}
            className="md-p"
            dangerouslySetInnerHTML={{ __html: boldProcessed }}
          />
        )
      })
  }

  return (
    <div className="report-page">
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
          <Link to="/predict" className="nav-link">Predict</Link>
          <Link to="/report" className="nav-link active">Report</Link>
        </div>
      </nav>

      {/* Main Content */}
      <main className="report-content">
        <motion.div
          className="page-header"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h1>
            <Sparkles className="header-icon" />
            AI-Generated Reports
          </h1>
          <p>Powered by Google Gemini AI for intelligent analysis</p>
        </motion.div>

        {/* Stats Bar */}
        <motion.div
          className="stats-bar"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="stat-item">
            <span className="stat-value">{datasetInfo.totalSamples.toLocaleString()}</span>
            <span className="stat-label">Samples</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{modelData.length}</span>
            <span className="stat-label">Models</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{(modelData[0].r2 * 100).toFixed(1)}%</span>
            <span className="stat-label">Best R²</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{datasetInfo.features}</span>
            <span className="stat-label">Features</span>
          </div>
        </motion.div>

        {/* Controls Row */}
        <motion.div
          className="controls-row"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          {/* Report Tabs */}
          <div className="report-tabs">
            {reportTabs.map((tab) => (
              <button
                key={tab.id}
                className={`report-tab ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.icon}
                <span>{tab.label}</span>
              </button>
            ))}
          </div>

          {/* Language Selector */}
          <div className="language-selector">
            <Languages size={18} className="lang-icon" />
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value as Language)}
              className="lang-select"
            >
              {(Object.keys(languageLabels) as Language[]).map((lang) => (
                <option key={lang} value={lang}>
                  {languageLabels[lang]}
                </option>
              ))}
            </select>
          </div>
        </motion.div>

        {/* Report Container */}
        <motion.div
          className="report-container"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          {/* Tab Info */}
          <div className="report-header">
            <div className="report-info">
              <h2>{currentTab.label}</h2>
              <p>{currentTab.description}</p>
            </div>
            <div className="report-actions">
              {reportContent[currentCacheKey] && (
                <button className="action-btn copy" onClick={handleCopy}>
                  {copied ? <Check size={16} /> : <Copy size={16} />}
                  <span>{copied ? (language === 'zh-TW' ? '已複製！' : 'Copied!') : (language === 'zh-TW' ? '複製' : 'Copy')}</span>
                </button>
              )}
              <button
                className="action-btn generate"
                onClick={handleGenerate}
                disabled={isGenerating}
              >
                {isGenerating ? (
                  <>
                    <Loader2 size={18} className="spinning" />
                    <span>{language === 'zh-TW' ? '生成中...' : 'Generating...'}</span>
                  </>
                ) : reportContent[currentCacheKey] ? (
                  <>
                    <RefreshCw size={18} />
                    <span>{language === 'zh-TW' ? '重新生成' : 'Regenerate'}</span>
                  </>
                ) : (
                  <>
                    <Sparkles size={18} />
                    <span>{language === 'zh-TW' ? '生成報告' : 'Generate Report'}</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Report Content */}
          <div className="report-body" ref={contentRef}>
            <AnimatePresence mode="wait">
              {!reportContent[currentCacheKey] && !isGenerating ? (
                <motion.div
                  key="empty"
                  className="empty-state"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <div className="empty-icon">{currentTab.icon}</div>
                  <h3>{language === 'zh-TW' ? '準備生成' : 'Ready to Generate'}</h3>
                  <p>
                    {language === 'zh-TW' 
                      ? `點擊「生成報告」按鈕，使用 AI 生成${currentTab.label}的分析報告。`
                      : `Click the "Generate Report" button to create an AI-powered analysis for ${currentTab.label.toLowerCase()}.`
                    }
                  </p>
                </motion.div>
              ) : (
                <motion.div
                  key="content"
                  className="report-text"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  {renderMarkdown(reportContent[currentCacheKey] || '')}
                  {isGenerating && (
                    <span className="cursor-blink">▊</span>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      </main>
    </div>
  )
}

export default Report

