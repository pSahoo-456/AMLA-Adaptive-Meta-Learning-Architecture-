import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useAppContext } from '../context/AppContext';
import { AlertTriangle, Lightbulb, Activity, BarChart2, ShieldCheck, Loader2 } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from 'recharts';

function Improve() {
  const { automlResults } = useAppContext();
  const [explainData, setExplainData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!automlResults) {
      setLoading(false);
      return;
    }

    const fetchExplanation = async () => {
      try {
        const res = await axios.post('http://localhost:8000/api/automl/explain');
        setExplainData(res.data);
      } catch (err) {
        console.error(err);
        setError(err.response?.data?.error || "Failed to fetch Explainable AI insights.");
      } finally {
        setLoading(false);
      }
    };

    fetchExplanation();
  }, [automlResults]);

  if (!automlResults) {
    return (
      <div className="glass-card border-amber-500/30 bg-amber-500/5 p-10 flex flex-col items-center text-center max-w-2xl mx-auto mt-20">
        <div className="w-16 h-16 bg-amber-500/10 rounded-full flex items-center justify-center mb-6">
          <AlertTriangle className="text-amber-500" size={32} />
        </div>
        <h2 className="text-2xl font-bold text-slate-100 mb-2">No Model Found</h2>
        <span className="text-slate-400 mb-8">Train a model using the AutoML Engine first to unlock Explainable AI insights.</span>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] animate-fade-in">
        <Loader2 className="animate-spin text-primary mb-4" size={48} />
        <h3 className="text-xl font-bold">Generating AI Explanations...</h3>
        <p className="text-slate-400">Computing SHAP and LIME feature attributions.</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-card border-red-500/30 bg-red-500/5 p-8 flex flex-col items-center text-center max-w-2xl mx-auto mt-20">
        <AlertTriangle className="text-red-500 mb-4" size={48} />
        <h2 className="text-xl font-bold mb-2">Explainable AI Error</h2>
        <p className="text-red-400/80">{error}</p>
      </div>
    );
  }

  return (
    <div className="animate-fade-in max-w-7xl mx-auto">
      <div className="mb-10">
        <h1 className="text-3xl font-bold mb-3 tracking-tight">Explain & Improve</h1>
        <p className="text-slate-400 text-lg">Understand your model's decisions and learn how to improve accuracy.</p>
      </div>

      {/* Suggestions Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div className="glass-card p-8">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-amber-500/10 rounded-xl">
              <Lightbulb className="text-amber-500" size={24} />
            </div>
            <h3 className="text-xl font-bold tracking-tight">Data & Model Suggestions</h3>
          </div>
          <div className="space-y-4">
            {explainData.suggestions.length > 0 ? (
              explainData.suggestions.map((suggestion, idx) => (
                <div key={idx} className="p-4 bg-slate-800/50 rounded-xl border border-slate-700/50 flex gap-4 items-start">
                  <div className="mt-1 text-primary"><ShieldCheck size={20} /></div>
                  <p className="text-slate-300 leading-relaxed text-sm">{suggestion}</p>
                </div>
              ))
            ) : (
              <p className="text-slate-400">Your dataset and model look healthy. No immediate suggestions.</p>
            )}
          </div>
        </div>

        <div className="glass-card p-8">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-emerald-500/10 rounded-xl">
              <Activity className="text-emerald-500" size={24} />
            </div>
            <h3 className="text-xl font-bold tracking-tight">What-If Scenarios</h3>
          </div>
          <div className="space-y-4">
            {explainData.what_if.length > 0 ? (
              explainData.what_if.map((scenario, idx) => (
                <div key={idx} className="p-4 bg-emerald-500/5 rounded-xl border border-emerald-500/20">
                  <p className="text-emerald-400/90 leading-relaxed text-sm font-medium">{scenario}</p>
                </div>
              ))
            ) : (
              <p className="text-slate-400">No what-if scenarios available.</p>
            )}
          </div>
        </div>
      </div>

      {/* XAI Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div className="glass-card p-8">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-primary/10 rounded-xl">
              <BarChart2 className="text-primary" size={24} />
            </div>
            <h3 className="text-xl font-bold tracking-tight">SHAP Feature Importance</h3>
          </div>
          {explainData.shap_error ? (
            <p className="text-amber-500 text-sm">{explainData.shap_error}</p>
          ) : (
            <div className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={explainData.shap.slice(0, 15)} layout="vertical" margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                  <XAxis type="number" stroke="#94a3b8" tick={{ fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                  <YAxis type="category" dataKey="feature" stroke="#94a3b8" tick={{ fill: '#e2e8f0', fontSize: 12 }} axisLine={false} tickLine={false} />
                  <Tooltip 
                    cursor={{ fill: '#1e293b' }}
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '0.75rem' }}
                    itemStyle={{ color: '#818cf8', fontWeight: 'bold' }}
                    formatter={(value) => [value.toFixed(4), 'SHAP Value']}
                  />
                  <Bar dataKey="shap_importance" fill="#6366f1" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        <div className="glass-card p-8">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-secondary/10 rounded-xl">
              <BarChart2 className="text-secondary" size={24} />
            </div>
            <h3 className="text-xl font-bold tracking-tight">LIME Local Explanation</h3>
          </div>
          {explainData.lime_error ? (
            <p className="text-amber-500 text-sm">{explainData.lime_error}</p>
          ) : (
            <div className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={explainData.lime} layout="vertical" margin={{ top: 5, right: 30, left: 150, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                  <XAxis type="number" stroke="#94a3b8" tick={{ fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                  <YAxis type="category" dataKey="feature_rule" stroke="#94a3b8" tick={{ fill: '#e2e8f0', fontSize: 12 }} axisLine={false} tickLine={false} />
                  <Tooltip 
                    cursor={{ fill: '#1e293b' }}
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '0.75rem' }}
                    formatter={(value) => [value.toFixed(4), 'Contribution']}
                  />
                  <Bar dataKey="contribution" radius={[0, 4, 4, 0]}>
                    {explainData.lime.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.contribution > 0 ? '#10b981' : '#ef4444'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Improve;
