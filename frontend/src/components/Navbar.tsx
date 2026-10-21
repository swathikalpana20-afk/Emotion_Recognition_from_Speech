import React from 'react';
import { NavLink } from 'react-router-dom';
import { Activity, Mic, BarChart2, Cpu, Sparkles } from 'lucide-react';

export const Navbar: React.FC = () => {
  return (
    <nav className="navbar">
      <div className="navbar-container">
        <NavLink to="/" className="navbar-brand">
          <div className="brand-icon">
            <Sparkles size={20} className="text-primary" />
          </div>
          <span className="brand-name">EMOTISENSE <span className="brand-accent">AI</span></span>
        </NavLink>

        <div className="navbar-links">
          <NavLink
            to="/"
            end
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
          >
            <Activity size={18} />
            <span>Dashboard</span>
          </NavLink>
          <NavLink
            to="/analyze"
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
          >
            <Mic size={18} />
            <span>Analyze Studio</span>
          </NavLink>
          <NavLink
            to="/insights"
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
          >
            <BarChart2 size={18} />
            <span>Model Insights</span>
          </NavLink>
          <NavLink
            to="/training"
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
          >
            <Cpu size={18} />
            <span>Training Lab</span>
          </NavLink>
        </div>
      </div>
    </nav>
  );
};
