import React, { useState } from 'react';
import axios from 'axios';
import { useAppContext } from '../context/AppContext';
import { useNavigate } from 'react-router-dom';
import { Play, Trophy, Activity, Loader2, Award, Zap, CheckCircle2, ChevronRight } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from 'recharts';

function AutoML() {
  const { file, targetColumn, taskType, setAutomlResults, setHistory, history } = useAppContext();
  const navigate = useNavigate();
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const startTraining = async () => {
    if (!file || !targetColumn || !taskType) {
      setError("Missing dataset or target column. Please navigate back to 'Upload Data' to configure your dataset.");
      return;
    }

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_column', targetColumn);
    formData.append('task_type', taskType);

    try {
      const res = await axios.post('http://localhost:8000/api/automl/train', formData);
      setResults(res.data);
      setAutomlResults(res.data);
      
      const score = res.data.ranked[0]?.f1 || res.data.ranked[0]?.r2 || 0;
      setHistory(prev => [...prev, {
        time: new Date().toLocaleTimeString(),
        task: taskType,
        target: targetColumn,
        best_model: res.data.best_model_name,
        score: score,
        dataset_name: file.name
      }]);
    } catch (err) {
      console.error(err);
      const backendError = err.response?.data?.error;
      if (backendError) {
        setError(`AutoML training failed: ${backendError}`);
      } else {
        setError('AutoML training failed. Please check the backend server logs for more details.');
      }
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[70vh] animate-fade-in">
        <div className="relative mb-10">
          <div className="absolute inset-0 bg-primary/20 blur-3xl rounded-full scale-150"></div>
          <div className="w-32 h-32 relative bg-surface/50 backdrop-blur-xl border border-primary/30 rounded-2xl flex items-center justify-center shadow-[0_0_50px_rgba(99,102,241,0.2)]">
            <Loader2 className="animate-spin text-primary" size={64} />
          </div>
        </div>
        <h2 className="text-3xl font-extrabold mb-4 tracking-tight bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
          Synthesizing Models
        </h2>
        <div className="max-w-md w-full bg-slate-900/50 rounded-xl p-6 border border-slate-700/50">
          <ul className="space-y-3">
            <li className="flex items-center gap-3 text-slate-300">
              <CheckCircle2 className="text-success" size={20} /> Extracting Dataset Fingerprint
            </li>
            <li className="flex items-center gap-3 text-slate-300">
              <CheckCircle2 className="text-success" size={20} /> Querying Meta-Knowledge Base
            </li>
            <li className="flex items-center gap-3 text-slate-300 animate-pulse">
              <Loader2 className="animate-spin text-primary" size={20} /> Parallel Training Algorithms
            </li>
            <li className="flex items-center gap-3 text-slate-500">
              <div className="w-5 h-5 rounded-full border-2 border-slate-700"></div> Ranking Predictions
            </li>
          </ul>
        </div>
      </div>
    );
  }

  return (
    <div className="animate-fade-in max-w-7xl mx-auto">
      <div className="mb-10">
        <h1 className="text-3xl font-bold mb-3 tracking-tight">AutoML Engine</h1>
        <p className="text-slate-400 text-lg max-w-3xl">
          Deploy the Adaptive Meta-Learning Architecture to automatically select, train, and rank the optimal machine learning algorithms for your dataset.
        </p>
      </div>

      {!results && !error && (
        <div className="glass-card p-12 flex flex-col items-center text-center max-w-3xl mx-auto mt-16 relative overflow-hidden group">
          <div className="absolute top-0 right-0 w-64 h-64 bg-primary/10 rounded-full blur-[80px] -mr-20 -mt-20 transition-all duration-700 group-hover:bg-primary/20"></div>
          <div className="absolute bottom-0 left-0 w-64 h-64 bg-accent/10 rounded-full blur-[80px] -ml-20 -mb-20 transition-all duration-700 group-hover:bg-accent/20"></div>
          
          <div className="w-24 h-24 bg-gradient-to-br from-primary/20 to-secondary/20 border border-primary/30 rounded-full flex items-center justify-center mb-8 relative z-10 shadow-[0_0_30px_rgba(99,102,241,0.2)]">
            <Zap size={40} className="text-primary animate-pulse" />
          </div>
          
          <h2 className="text-3xl font-bold mb-4 tracking-tight relative z-10">System Ready for Training</h2>
          
          <div className="flex items-center gap-6 mb-10 bg-slate-900/50 px-8 py-4 rounded-full border border-slate-700/50 relative z-10">
            <div className="flex flex-col items-center">
              <span className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">Target</span>
              <span className="text-lg font-semibold text-slate-200">{targetColumn || 'Not Selected'}</span>
            </div>
            <div className="w-px h-10 bg-slate-700"></div>
            <div className="flex flex-col items-center">
              <span className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">Task</span>
              <span className="text-lg font-semibold text-accent capitalize">{taskType || 'Unknown'}</span>
            </div>
          </div>
          
          <button 
            className="btn btn-primary px-10 py-4 text-lg font-bold rounded-xl shadow-[0_0_40px_rgba(99,102,241,0.4)] hover:shadow-[0_0_60px_rgba(99,102,241,0.6)] relative z-10 hover:-translate-y-1 transition-all duration-300"
            onClick={startTraining}
          >
            <Play fill="currentColor" size={20} className="mr-2" />
            Initialize Pipeline
          </button>
        </div>
      )}

      {error && (
        <div className="glass-card border-red-500/30 bg-red-500/5 p-8 flex flex-col items-center gap-6 max-w-2xl mx-auto">
          <div className="bg-red-500/10 p-4 rounded-full">
            <Activity className="text-red-500" size={40} />
          </div>
          <span className="text-red-400 font-bold text-xl text-center">{error}</span>
          <button className="btn btn-secondary px-8 py-3" onClick={() => navigate('/upload')}>Return to Upload</button>
        </div>
      )}

      {results && (
        <div className="animate-fade-in space-y-8">
          {/* Winner Banner */}
          <div className="glass-card overflow-hidden relative border border-accent/30 shadow-[0_0_30px_rgba(16,185,129,0.15)]">
            <div className="absolute top-0 right-0 p-12 opacity-[0.03] pointer-events-none transform translate-x-10 -translate-y-10">
              <Trophy size={250} className="text-accent" />
            </div>
            
            <div className="p-8 lg:p-10 relative z-10">
              <div className="flex items-center gap-3 mb-6">
                <div className="bg-amber-500/20 p-2 rounded-lg border border-amber-500/30">
                  <Award className="text-amber-500" size={24} />
                </div>
                <h2 className="text-sm font-bold text-slate-400 uppercase tracking-widest">Recommended Algorithm</h2>
              </div>
              
              <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-8">
                <div>
                  <h3 className="text-4xl lg:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-accent to-emerald-300 mb-2">
                    {results.best_model_name}
                  </h3>
                  <p className="text-slate-400 text-lg">Top performer across cross-validation folds.</p>
                </div>
                
                <div className="flex gap-4">
                  <div className="bg-slate-900/60 backdrop-blur-md border border-slate-700 rounded-xl p-5 min-w-[140px]">
                    <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1">Status</div>
                    <div className="font-bold text-xl text-success capitalize flex items-center gap-2">
                      <CheckCircle2 size={18} /> {results.status}
                    </div>
                  </div>
                  <div className="bg-slate-900/60 backdrop-blur-md border border-slate-700 rounded-xl p-5 min-w-[140px]">
                    <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1">Confidence</div>
                    <div className="font-bold text-xl text-slate-200">
                      {(results.algorithm_recommendation?.confidence * 100)?.toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
            {/* Chart */}
            <div className="glass-card p-8 flex flex-col">
              <h3 className="text-xl font-bold tracking-tight mb-8">Performance Leaderboard (F1 Score)</h3>
              <div className="flex-1 min-h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={results.ranked} layout="vertical" margin={{ top: 0, right: 30, left: 40, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                    <XAxis type="number" stroke="#94a3b8" domain={[0, 1]} tick={{ fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                    <YAxis dataKey="model" type="category" stroke="#94a3b8" width={110} tick={{ fill: '#94a3b8', fontWeight: 500 }} axisLine={false} tickLine={false} />
                    <Tooltip 
                      cursor={{ fill: '#1e293b' }}
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '0.75rem', boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5)' }}
                      itemStyle={{ color: '#10b981', fontWeight: 'bold' }}
                    />
                    <Bar dataKey="f1" radius={[0, 6, 6, 0]} maxBarSize={40}>
                      {results.ranked.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={index === 0 ? '#10b981' : '#4f46e5'} className="transition-all duration-300" />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Table */}
            <div className="glass-card p-0 overflow-hidden flex flex-col">
              <div className="p-8 border-b border-slate-700/50 bg-slate-800/20">
                <h3 className="text-xl font-bold tracking-tight">Detailed Metrics</h3>
              </div>
              <div className="table-container border-0 rounded-none bg-transparent flex-1">
                <table className="data-table">
                  <thead className="bg-slate-900/80">
                    <tr>
                      <th className="w-16 text-center">Rank</th>
                      <th>Algorithm</th>
                      <th className="text-right">F1 Score</th>
                      <th className="text-right">Accuracy</th>
                      <th className="text-right">ROC AUC</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/50">
                    {results.ranked.map((row, idx) => (
                      <tr key={idx} className={`transition-colors ${idx === 0 ? "bg-accent/10 hover:bg-accent/20" : "hover:bg-slate-800/40 group"}`}>
                        <td className="text-center">
                          {idx === 0 ? (
                            <div className="w-8 h-8 rounded-full bg-accent text-slate-900 font-bold flex items-center justify-center mx-auto shadow-[0_0_10px_rgba(16,185,129,0.5)]">
                              1
                            </div>
                          ) : (
                            <span className="font-bold text-slate-500">{idx + 1}</span>
                          )}
                        </td>
                        <td className={`font-semibold ${idx === 0 ? "text-accent" : "text-slate-300 group-hover:text-primary"}`}>
                          {row.model}
                        </td>
                        <td className={`text-right font-mono font-medium ${idx === 0 ? "text-accent" : "text-slate-300"}`}>
                          {row.f1?.toFixed(4)}
                        </td>
                        <td className="text-right font-mono text-sm text-slate-400">{row.accuracy?.toFixed(4)}</td>
                        <td className="text-right font-mono text-sm text-slate-400">{row.roc_auc?.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default AutoML;
