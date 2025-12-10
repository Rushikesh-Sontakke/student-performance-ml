import { Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import Models from './pages/Models'
import Predict from './pages/Predict'
import Report from './pages/Report'
import './App.css'

function App() {
  return (
    <div className="app">
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/models" element={<Models />} />
        <Route path="/predict" element={<Predict />} />
        <Route path="/report" element={<Report />} />
      </Routes>
    </div>
  )
}

export default App

