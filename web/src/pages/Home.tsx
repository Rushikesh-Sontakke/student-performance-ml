import { motion } from 'framer-motion'
import { Brain, Target, Database, ChevronRight, Sparkles, Users } from 'lucide-react'
import { Link } from 'react-router-dom'
import './Home.css'

const teamMembers = [
  { name: 'Heainz Manoj', studentId: 'X1145017', task: 'Neural Network' },
  { name: 'Carrin', studentId: '112006201', task: 'Random Forest / Decision Tree' },
  { name: 'Chew', studentId: '111060062', task: 'XGBoost' },
  { name: 'Ray', studentId: '111062304', task: 'Random Forest / Decision Tree' },
  { name: 'Shi Jie', studentId: '111000263', task: 'Linear Regression, Website Develop & Deployment' },
  { name: 'Rushi', studentId: '111006244', task: 'Data Preprocessing, Ensemble Model, Git' },
]

const Home = () => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15,
        delayChildren: 0.2,
      },
    },
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.6,
        ease: [0.22, 1, 0.36, 1],
      },
    },
  }

  const floatingVariants = {
    animate: {
      y: [-10, 10, -10],
      transition: {
        duration: 4,
        repeat: Infinity,
        ease: 'easeInOut',
      },
    },
  }

  return (
    <div className="home">
      {/* Animated background */}
      <div className="background">
        <div className="gradient-orb orb-1" />
        <div className="gradient-orb orb-2" />
        <div className="gradient-orb orb-3" />
        <div className="grid-overlay" />
      </div>

      {/* Navigation */}
      <nav className="nav">
        <div className="nav-logo">
          <Brain className="nav-icon" />
          <span>SPP</span>
        </div>
        <div className="nav-links">
          <Link to="/" className="nav-link active">Home</Link>
          <Link to="/models" className="nav-link">Models</Link>
          <Link to="/predict" className="nav-link">Predict</Link>
          <Link to="/report" className="nav-link">Report</Link>
        </div>
      </nav>

      {/* Hero Section */}
      <motion.main
        className="hero"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <motion.div className="hero-badge" variants={itemVariants}>
          <Sparkles size={14} />
          <span>Machine Learning Final Project</span>
        </motion.div>

        <motion.h1 className="hero-title" variants={itemVariants}>
          <span className="title-gradient">Student Performance</span>
          <br />
          <span className="title-white">Predictor</span>
        </motion.h1>

        <motion.div className="hero-group" variants={itemVariants}>
          <div className="group-badge">
            <span className="group-label">Group</span>
            <span className="group-number">3</span>
          </div>
        </motion.div>

        <motion.div className="hero-goal" variants={itemVariants}>
          <div className="goal-card">
            <Target className="goal-icon" />
            <div className="goal-content">
              <span className="goal-label">Our Goal</span>
              <p className="goal-text">
                Predict Student Performance Data from a Kaggle Data Set
              </p>
            </div>
          </div>
        </motion.div>

        <motion.div className="hero-cta" variants={itemVariants}>
          <Link to="/models" className="cta-button primary">
            <span>Explore Models</span>
            <ChevronRight size={18} />
          </Link>
          <a 
            href="https://www.kaggle.com/datasets/lainguyn123/student-performance-factors" 
            target="_blank" 
            rel="noopener noreferrer"
            className="cta-button secondary"
          >
            <span>View Dataset</span>
            <Database size={16} />
          </a>
        </motion.div>

        {/* Floating elements */}
        <motion.div
          className="floating-element elem-1"
          variants={floatingVariants}
          animate="animate"
        >
          <div className="float-card">
            <span className="float-value">95.2%</span>
            <span className="float-label">Accuracy</span>
          </div>
        </motion.div>

        <motion.div
          className="floating-element elem-2"
          variants={floatingVariants}
          animate="animate"
          style={{ animationDelay: '1s' }}
        >
          <div className="float-card">
            <span className="float-value">5</span>
            <span className="float-label">Models</span>
          </div>
        </motion.div>

        <motion.div
          className="floating-element elem-3"
          variants={floatingVariants}
          animate="animate"
          style={{ animationDelay: '2s' }}
        >
          <div className="float-card">
            <span className="float-value">6.6K</span>
            <span className="float-label">Samples</span>
          </div>
        </motion.div>
      </motion.main>

      {/* Team Section */}
      <motion.section
        className="team-section"
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8, duration: 0.6 }}
      >
        <div className="team-header">
          <Users className="team-icon" />
          <h2>Team Members</h2>
        </div>
        <div className="team-grid">
          {teamMembers.map((member, index) => (
            <motion.div
              key={member.studentId}
              className="team-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.9 + index * 0.1 }}
            >
              <div className="member-avatar">
                {member.name.charAt(0)}
              </div>
              <div className="member-info">
                <h3 className="member-name">{member.name}</h3>
                <span className="member-id">{member.studentId}</span>
                <span className="member-task">{member.task}</span>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* Footer */}
      <footer className="footer">
        <p>© 2025 Group 3 • Machine Learning Final Project</p>
      </footer>
    </div>
  )
}

export default Home

