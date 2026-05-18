import React from 'react';
import { useAppContext } from '../context/AppContext';
import { History, LineChart as LineChartIcon, Activity } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts';

function Track() {
  const { history } = useAppContext();

  if (!history || history.length === 0) {
    return (
      <div className="glass-card border-slate-500/30 bg-slate-500/5 p-10 flex flex-col items-center text-center max-w-2xl mx-auto mt-20">
        <div className="w-16 h-16 bg-slate-500/10 rounded-full flex items-center justify-center mb-6">
          <History className="text-slate-400" size={32} />
        </div>
        <h2 className="text-2xl font-bold text-slate-100 mb-2">No History Yet</h2>
        <span className="text-slate-400 mb-8">Run the AutoML pipeline on one or more datasets to track performance changes over time.</span>
      </div>
    );
  }

  return (
    <div className="animate-fade-in max-w-7xl mx-auto">
      <div className="mb-10">
        <h1 className="text-3xl font-bold mb-3 tracking-tight">Performance Tracking</h1>
        <p className="text-slate-400 text-lg">Track model performance across different datasets and configurations over time.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="glass-card p-6 border-l-4 border-l-primary flex flex-col">
          <div className="text-slate-400 text-sm font-medium mb-1">Total Runs</div>
          <div className="text-3xl font-bold text-slate-100">{history.length}</div>
        </div>
        <div className="glass-card p-6 border-l-4 border-l-success flex flex-col">
          <div className="text-slate-400 text-sm font-medium mb-1">Highest Score</div>
          <div className="text-3xl font-bold text-slate-100">
            {Math.max(...history.map(h => h.score)).toFixed(4)}
          </div>
        </div>
        <div className="glass-card p-6 border-l-4 border-l-secondary flex flex-col">
          <div className="text-slate-400 text-sm font-medium mb-1">Latest Best Model</div>
          <div className="text-xl font-bold text-slate-100 truncate mt-2">{history[history.length - 1].best_model}</div>
        </div>
      </div>

      <div className="glass-card p-8 mb-8">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 bg-primary/10 rounded-xl">
            <LineChartIcon className="text-primary" size={24} />
          </div>
          <h3 className="text-xl font-bold tracking-tight">Timeline: Performance by Run</h3>
        </div>
        
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 20, right: 30, left: 20, bottom: 25 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
              <XAxis dataKey="time" stroke="#94a3b8" tick={{ fill: '#94a3b8' }} axisLine={false} tickLine={false} />
              <YAxis domain={['auto', 'auto']} stroke="#94a3b8" tick={{ fill: '#94a3b8' }} axisLine={false} tickLine={false} />
              <Tooltip 
                cursor={{ stroke: '#334155', strokeWidth: 1, strokeDasharray: '5 5' }}
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '0.75rem' }}
                labelStyle={{ color: '#94a3b8', marginBottom: '0.5rem' }}
              />
              <Legend wrapperStyle={{ paddingTop: '20px' }} />
              <Line 
                type="monotone" 
                dataKey="score" 
                name="Score (F1/R2)" 
                stroke="#6366f1" 
                strokeWidth={3} 
                dot={{ r: 6, fill: '#0f172a', stroke: '#6366f1', strokeWidth: 2 }}
                activeDot={{ r: 8, fill: '#6366f1', stroke: '#fff', strokeWidth: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="glass-card overflow-hidden">
        <div className="px-8 py-6 border-b border-slate-700/50 bg-slate-800/30">
          <h3 className="text-xl font-bold tracking-tight">Experiment History Log</h3>
        </div>
        <div className="table-container border-0 rounded-none bg-transparent">
          <table className="data-table">
            <thead className="bg-slate-900/80">
              <tr>
                <th>Time</th>
                <th>Dataset</th>
                <th>Target</th>
                <th>Task</th>
                <th>Best Model</th>
                <th className="text-right">Score</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {history.map((row, idx) => (
                <tr key={idx} className="hover:bg-slate-800/40 transition-colors">
                  <td className="text-slate-400 font-mono text-sm">{row.time}</td>
                  <td className="font-medium text-slate-200">{row.dataset_name}</td>
                  <td><span className="badge badge-primary">{row.target}</span></td>
                  <td><span className="badge badge-secondary">{row.task}</span></td>
                  <td className="font-semibold text-accent">{row.best_model}</td>
                  <td className="text-right font-mono text-sm font-bold text-success">{row.score?.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default Track;
