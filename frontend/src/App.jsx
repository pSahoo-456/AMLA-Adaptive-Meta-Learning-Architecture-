import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation, Outlet } from 'react-router-dom';
import { LayoutDashboard, UploadCloud, BarChart2, Cpu, GitCompare, Lightbulb, History, Rocket } from 'lucide-react';
import LandingPage from './pages/LandingPage';
import Dashboard from './pages/Dashboard';
import Upload from './pages/Upload';
import Analysis from './pages/Analysis';
import AutoML from './pages/AutoML';
import Compare from './pages/Compare';
import Improve from './pages/Improve';
import Track from './pages/Track';
import Deploy from './pages/Deploy';

function Sidebar() {
  const location = useLocation();
  const navItems = [
    { path: '/dashboard', name: 'Dashboard', icon: <LayoutDashboard size={20} /> },
    { path: '/upload', name: 'Upload Data', icon: <UploadCloud size={20} /> },
    { path: '/analysis', name: 'EDA Analysis', icon: <BarChart2 size={20} /> },
    { path: '/automl', name: 'AutoML Engine', icon: <Cpu size={20} /> },
    { path: '/compare', name: 'Compare Models', icon: <GitCompare size={20} /> },
    { path: '/improve', name: 'Explain & Improve', icon: <Lightbulb size={20} /> },
    { path: '/track', name: 'Track History', icon: <History size={20} /> },
    { path: '/deploy', name: 'Deploy', icon: <Rocket size={20} /> }
  ];

  return (
    <div className="w-72 bg-surface/95 border-r border-slate-700/50 flex flex-col p-6 backdrop-blur-xl shrink-0">
      <div className="flex items-center gap-3 mb-10">
        <Link to="/" className="flex items-center gap-3 group">
          <div className="bg-primary/20 p-2 rounded-xl border border-primary/30 shadow-[0_0_15px_rgba(99,102,241,0.3)] group-hover:scale-110 transition-transform">
            <Cpu className="text-primary" size={26} />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
              AMLA
            </h1>
            <p className="text-xs text-slate-400 font-medium tracking-wide">ENTERPRISE EDITION</p>
          </div>
        </Link>
      </div>
      
      <nav className="flex flex-col gap-2 flex-1">
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 ml-2">Platform</div>
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <Link 
              key={item.path} 
              to={item.path}
              className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 group ${
                isActive 
                  ? 'bg-primary/10 text-primary font-semibold border border-primary/20 shadow-[inset_0_1px_1px_rgba(255,255,255,0.05)]' 
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <div className={`transition-transform duration-200 ${isActive ? 'scale-110' : 'group-hover:scale-110'}`}>
                {item.icon}
              </div>
              <span>{item.name}</span>
            </Link>
          );
        })}
      </nav>
      
      <div className="mt-auto glass-card p-5 text-sm">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-2 h-2 rounded-full bg-accent animate-pulse"></div>
          <span className="font-semibold text-slate-200">System Online</span>
        </div>
        <p className="text-xs text-slate-400">Adaptive Meta-Learning Architecture v1.0.0</p>
      </div>
    </div>
  );
}

function AppLayout() {
  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar />
      <main className="flex-1 overflow-y-auto p-8 lg:p-12 relative">
        {/* Subtle background glow effect */}
        <div className="absolute top-0 right-0 -mr-20 -mt-20 w-96 h-96 bg-primary/10 rounded-full blur-[100px] pointer-events-none"></div>
        <Outlet />
      </main>
    </div>
  );
}

function App() {
  return (
    <Router>
      <Routes>
        {/* Landing Page gets full screen without Sidebar */}
        <Route path="/" element={<LandingPage />} />
        
        {/* App components get the Sidebar Layout */}
        <Route element={<AppLayout />}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/analysis" element={<Analysis />} />
          <Route path="/automl" element={<AutoML />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/improve" element={<Improve />} />
          <Route path="/track" element={<Track />} />
          <Route path="/deploy" element={<Deploy />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
