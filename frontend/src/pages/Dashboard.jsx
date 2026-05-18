import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Activity, Server, Clock, Trophy, ArrowRight, Zap, Database, BarChart3 } from 'lucide-react';
import { Link } from 'react-router-dom';

function Dashboard() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    axios.get('http://localhost:8000/health')
      .then(res => setHealth(res.data))
      .catch(err => console.error("API offline", err));
  }, []);

  return (
    <div className="animate-fade-in max-w-7xl mx-auto">
      <div className="mb-12">
        <h1 className="text-4xl font-extrabold mb-4 tracking-tight">Welcome to AMLA</h1>
        <p className="text-slate-400 text-lg max-w-3xl leading-relaxed">
          The Adaptive Meta-Learning Architecture for end-to-end AutoML. Upload, analyze, train, compare, and deploy from one centralized, intelligent workspace.
        </p>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
        <div className="glass-card p-6 flex flex-col relative overflow-hidden group">
          <div className="absolute -right-4 -top-4 w-24 h-24 bg-primary/10 rounded-full blur-xl group-hover:bg-primary/20 transition-all"></div>
          <div className="flex items-center gap-3 mb-4 text-slate-400">
            <div className="p-2 bg-primary/10 rounded-lg text-primary">
              <Activity size={20} />
            </div>
            <span className="font-semibold tracking-wide text-sm uppercase">Engine Status</span>
          </div>
          <div className="text-3xl font-bold text-slate-100 mb-1">Live</div>
          <div className="text-sm text-accent font-medium flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-accent animate-pulse"></div> Ready to process
          </div>
        </div>
        
        <div className="glass-card p-6 flex flex-col relative overflow-hidden group">
          <div className="absolute -right-4 -top-4 w-24 h-24 bg-accent/10 rounded-full blur-xl group-hover:bg-accent/20 transition-all"></div>
          <div className="flex items-center gap-3 mb-4 text-slate-400">
            <div className={`p-2 rounded-lg ${health ? "bg-accent/10 text-accent" : "bg-red-500/10 text-red-500"}`}>
              <Server size={20} />
            </div>
            <span className="font-semibold tracking-wide text-sm uppercase">API Connection</span>
          </div>
          <div className="text-3xl font-bold text-slate-100 mb-1">{health ? "Healthy" : "Offline"}</div>
          <div className={`text-sm font-medium ${health ? 'text-accent' : 'text-red-500'}`}>
            {health ? "Latency: < 50ms" : "Cannot reach backend"}
          </div>
        </div>

        <div className="glass-card p-6 flex flex-col relative overflow-hidden group">
          <div className="absolute -right-4 -top-4 w-24 h-24 bg-secondary/10 rounded-full blur-xl group-hover:bg-secondary/20 transition-all"></div>
          <div className="flex items-center gap-3 mb-4 text-slate-400">
            <div className="p-2 bg-secondary/10 rounded-lg text-secondary">
              <Clock size={20} />
            </div>
            <span className="font-semibold tracking-wide text-sm uppercase">Experiments</span>
          </div>
          <div className="text-3xl font-bold text-slate-100 mb-1">{health?.total_experiments || 0}</div>
          <div className="text-sm text-slate-400 font-medium">Historical tracked runs</div>
        </div>

        <div className="glass-card p-6 flex flex-col relative overflow-hidden group">
          <div className="absolute -right-4 -top-4 w-24 h-24 bg-amber-500/10 rounded-full blur-xl group-hover:bg-amber-500/20 transition-all"></div>
          <div className="flex items-center gap-3 mb-4 text-slate-400">
            <div className="p-2 bg-amber-500/10 rounded-lg text-amber-500">
              <Trophy size={20} />
            </div>
            <span className="font-semibold tracking-wide text-sm uppercase">Current Leader</span>
          </div>
          <div className="text-3xl font-bold text-slate-100 mb-1">None</div>
          <div className="text-sm text-slate-400 font-medium">Awaiting new training data</div>
        </div>
      </div>

      <div className="mb-10 flex items-center justify-between">
        <h2 className="text-2xl font-bold tracking-tight">Platform Modules</h2>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <Link to="/upload" className="glass-card p-8 group hover:border-primary/50 transition-all cursor-pointer flex flex-col">
          <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center text-primary mb-6 group-hover:scale-110 transition-transform shadow-[0_0_15px_rgba(99,102,241,0.2)]">
            <Database size={28} />
          </div>
          <h3 className="text-xl font-bold mb-3 text-slate-100 group-hover:text-primary transition-colors">1. Data Ingestion</h3>
          <p className="text-slate-400 mb-6 flex-1 leading-relaxed">
            Securely import structured datasets. AMLA automatically infers schemas, profiles the data, and identifies the target predictive task.
          </p>
          <div className="flex items-center text-primary font-medium mt-auto group-hover:gap-3 transition-all gap-2">
            Upload Data <ArrowRight size={18} />
          </div>
        </Link>

        <Link to="/analysis" className="glass-card p-8 group hover:border-secondary/50 transition-all cursor-pointer flex flex-col">
          <div className="w-14 h-14 rounded-2xl bg-secondary/10 flex items-center justify-center text-secondary mb-6 group-hover:scale-110 transition-transform shadow-[0_0_15px_rgba(139,92,246,0.2)]">
            <BarChart3 size={28} />
          </div>
          <h3 className="text-xl font-bold mb-3 text-slate-100 group-hover:text-secondary transition-colors">2. Advanced EDA</h3>
          <p className="text-slate-400 mb-6 flex-1 leading-relaxed">
            Explore deep statistical insights. Visualize missingness, feature distributions, and complex correlations before modeling.
          </p>
          <div className="flex items-center text-secondary font-medium mt-auto group-hover:gap-3 transition-all gap-2">
            Analyze Metrics <ArrowRight size={18} />
          </div>
        </Link>

        <Link to="/automl" className="glass-card p-8 group hover:border-accent/50 transition-all cursor-pointer flex flex-col">
          <div className="w-14 h-14 rounded-2xl bg-accent/10 flex items-center justify-center text-accent mb-6 group-hover:scale-110 transition-transform shadow-[0_0_15px_rgba(16,185,129,0.2)]">
            <Zap size={28} />
          </div>
          <h3 className="text-xl font-bold mb-3 text-slate-100 group-hover:text-accent transition-colors">3. Meta-Learning Engine</h3>
          <p className="text-slate-400 mb-6 flex-1 leading-relaxed">
            Leverage the core meta-knowledge base to recommend and parallel-train the optimal algorithm stack for your specific dataset DNA.
          </p>
          <div className="flex items-center text-accent font-medium mt-auto group-hover:gap-3 transition-all gap-2">
            Launch AutoML <ArrowRight size={18} />
          </div>
        </Link>
      </div>
    </div>
  );
}

export default Dashboard;
