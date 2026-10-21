import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Navbar } from './components/Navbar';
import Dashboard from './pages/Dashboard';
import Analyze from './pages/Analyze';
import Insights from './pages/Insights';
import Training from './pages/Training';

export const App: React.FC = () => {
  return (
    <div className="app-container">
      <Navbar />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/insights" element={<Insights />} />
          <Route path="/training" element={<Training />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <footer className="footer">
        <p>EMOTISENSE AI • Real-time Speech Emotion Recognition System • MFCC + Hybrid CNN-BiLSTM</p>
      </footer>
    </div>
  );
};

export default App;
