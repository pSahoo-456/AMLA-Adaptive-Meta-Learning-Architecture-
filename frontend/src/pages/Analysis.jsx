import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useAppContext } from '../context/AppContext';
import { useNavigate } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell, Legend } from 'recharts';
import { AlertTriangle, Loader2, Info, ArrowRight, CheckCircle, Database, BarChart3, Activity } from 'lucide-react';

function Analysis() {
  const { file, targetColumn, taskType } = useAppContext();
  const navigate = useNavigate();
  const [profile, setProfile] = useState(null);
  const [signal, setSignal] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    if (!file || !targetColumn) {
      navigate('/upload');
      return;
    }
    fetchData();
  }, [file, targetColumn, navigate]);

  const fetchData = async () => {
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_column', targetColumn);

    try {
      // Fetch Profile
      const resProfile = await axios.post('http://localhost:8000/api/eda/profile', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setProfile(resProfile.data);
      setLoading(false); // Unblock the UI immediately

      // Fetch Feature Signal in the background
      if (taskType) {
        const signalFormData = new FormData();
        signalFormData.append('file', file);
        signalFormData.append('target_column', targetColumn);
        signalFormData.append('task_type', taskType);
        
        axios.post('http://localhost:8000/api/eda/feature-signal', signalFormData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        }).then(resSignal => {
          setSignal(resSignal.data.signal);
        }).catch(err => {
          console.error("Feature signal failed to load:", err);
        });
      }
    } catch (err) {
      console.error(err);
      setError('Failed to fetch dataset profile. Ensure your dataset is valid and matches the target.');
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] animate-fade-in max-w-7xl mx-auto">
        <div className="w-24 h-24 bg-primary/10 rounded-full flex items-center justify-center mb-8 relative">
          <div className="absolute inset-0 rounded-full border-t-2 border-primary animate-spin"></div>
          <Loader2 className="text-primary animate-pulse" size={40} />
        </div>
        <h2 className="text-3xl font-bold mb-3 tracking-tight">Profiling Dataset</h2>
        <p className="text-slate-400 text-lg">Extracting statistical moments, missingness matrices, and correlations...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-card border-red-500/30 bg-red-500/5 p-10 flex flex-col items-center text-center max-w-2xl mx-auto mt-20">
        <div className="w-16 h-16 bg-red-500/10 rounded-full flex items-center justify-center mb-6">
          <AlertTriangle className="text-red-500" size={32} />
        </div>
        <h2 className="text-2xl font-bold text-slate-100 mb-2">Analysis Failed</h2>
        <span className="text-slate-400 mb-8">{error}</span>
        <button className="btn btn-secondary px-6" onClick={() => navigate('/upload')}>Return to Upload</button>
      </div>
    );
  }

  if (!profile) return null;

  const missingData = Object.entries(profile.missing_by_column || {}).map(([key, value]) => ({
    name: key,
    missing: value
  })).filter(item => item.missing > 0);

  const outlierData = Object.entries(profile.outlier_counts || {}).map(([key, value]) => ({
    name: key,
    outliers: value
  })).filter(item => item.outliers > 0).sort((a, b) => b.outliers - a.outliers);

  const tabs = [
    { id: 'overview', name: 'Overview' },
    { id: 'missing', name: 'Missing Values' },
    { id: 'distributions', name: 'Distributions' },
    { id: 'correlation', name: 'Correlation Heatmap' },
    { id: 'boxplots', name: 'Box Plots (Outliers)' },
    { id: 'stats', name: 'Statistical Moments' },
    { id: 'signal', name: 'Feature Signal' }
  ];

  return (
    <div className="animate-fade-in max-w-7xl mx-auto pb-12">
      <div className="flex justify-between items-end mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-3 tracking-tight">Exploratory Data Analysis</h1>
          <p className="text-slate-400 text-lg flex items-center gap-2">
            Target Variable: 
            <span className="badge badge-primary px-3 py-1 text-sm">{targetColumn}</span>
          </p>
        </div>
        <button className="btn btn-primary px-6 py-2.5 shadow-[0_0_20px_rgba(99,102,241,0.3)]" onClick={() => navigate('/automl')}>
          Launch AutoML <ArrowRight size={18} />
        </button>
      </div>

      {/* Primary KPI Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="glass-card p-6 relative overflow-hidden group hover:border-slate-600">
          <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Total Rows</div>
          <div className="text-3xl font-extrabold text-slate-100">{(profile.rows || 0).toLocaleString()}</div>
          <div className="absolute -bottom-6 -right-6 w-24 h-24 bg-slate-800 rounded-full opacity-50 group-hover:bg-slate-700 transition-colors"></div>
        </div>
        <div className="glass-card p-6 relative overflow-hidden group hover:border-slate-600">
          <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Total Features</div>
          <div className="text-3xl font-extrabold text-slate-100">{profile.columns || 0}</div>
          <div className="absolute -bottom-6 -right-6 w-24 h-24 bg-slate-800 rounded-full opacity-50 group-hover:bg-slate-700 transition-colors"></div>
        </div>
        <div className="glass-card p-6 relative overflow-hidden group hover:border-red-900/30">
          <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Missing Cells</div>
          <div className={`text-3xl font-extrabold ${(profile.missing_total || 0) > 0 ? 'text-red-400' : 'text-accent'}`}>
            {(profile.missing_total || 0).toLocaleString()}
          </div>
          <div className="absolute -bottom-6 -right-6 w-24 h-24 bg-slate-800 rounded-full opacity-50 group-hover:bg-red-900/20 transition-colors"></div>
        </div>
        <div className="glass-card p-6 relative overflow-hidden group hover:border-amber-900/30">
          <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Duplicate Rows</div>
          <div className={`text-3xl font-extrabold ${(profile.duplicate_rows || 0) > 0 ? 'text-amber-400' : 'text-accent'}`}>
            {(profile.duplicate_rows || 0).toLocaleString()}
          </div>
          <div className="absolute -bottom-6 -right-6 w-24 h-24 bg-slate-800 rounded-full opacity-50 group-hover:bg-amber-900/20 transition-colors"></div>
        </div>
      </div>

      {/* Tabs Navigation */}
      <div className="flex border-b border-slate-700/50 mb-8 overflow-x-auto">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-6 py-3 font-semibold text-sm whitespace-nowrap transition-colors border-b-2 ${
              activeTab === tab.id 
                ? 'border-primary text-primary bg-primary/5' 
                : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-800/30'
            }`}
          >
            {tab.name}
          </button>
        ))}
      </div>

      {/* Tab Contents */}
      <div className="animate-fade-in">
        {/* OVERVIEW TAB */}
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="glass-card p-8 flex flex-col">
              <h3 className="text-xl font-bold tracking-tight mb-8">Feature Composition</h3>
              <div className="flex gap-6 mb-8">
                <div className="flex-1 bg-gradient-to-br from-primary/20 to-primary/5 border border-primary/20 p-6 rounded-2xl relative overflow-hidden">
                  <div className="text-sm font-semibold text-primary uppercase tracking-wider mb-2">Numerical</div>
                  <div className="text-5xl font-extrabold text-white">{profile.numeric_columns}</div>
                  <div className="absolute right-[-20px] bottom-[-20px] opacity-10">
                    <BarChart3 size={120} />
                  </div>
                </div>
                <div className="flex-1 bg-gradient-to-br from-secondary/20 to-secondary/5 border border-secondary/20 p-6 rounded-2xl relative overflow-hidden">
                  <div className="text-sm font-semibold text-secondary uppercase tracking-wider mb-2">Categorical</div>
                  <div className="text-5xl font-extrabold text-white">{profile.categorical_columns}</div>
                  <div className="absolute right-[-20px] bottom-[-20px] opacity-10">
                    <Database size={120} />
                  </div>
                </div>
              </div>
              <div className="mt-auto bg-slate-800/50 p-5 rounded-xl border border-slate-700">
                <h4 className="text-sm font-bold text-slate-300 mb-2">AutoML Preprocessing Plan:</h4>
                <ul className="text-sm text-slate-400 space-y-2 list-disc pl-5 marker:text-primary">
                  <li>Numerical features will be imputed (median) and standard scaled.</li>
                  <li>Categorical features will be checked for cardinality and one-hot encoded.</li>
                  <li>Target variable "{targetColumn}" will be isolated for model training.</li>
                </ul>
              </div>
            </div>

            <div className="glass-card p-8">
              <h3 className="text-xl font-bold tracking-tight mb-6">Target Distribution</h3>
              <div className="flex flex-col items-center justify-center h-[300px] bg-slate-800/30 rounded-xl border border-slate-700/50">
                <div className="text-center">
                  <div className="text-4xl font-extrabold text-slate-200 mb-2">{profile.target_unique}</div>
                  <div className="text-slate-400 font-medium">Unique Target Values</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* MISSING VALUES TAB */}
        {activeTab === 'missing' && (
          <div className="glass-card p-8 flex flex-col h-[500px]">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-bold tracking-tight">Missing Values Matrix</h3>
              <div className="p-1.5 bg-slate-800 rounded-lg text-slate-400" title="Columns containing NaN values">
                <Info size={16} />
              </div>
            </div>
            
            <div className="flex-1 min-h-0">
              {missingData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={missingData} layout="vertical" margin={{ top: 10, right: 30, left: 80, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                    <XAxis type="number" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false} />
                    <YAxis type="category" dataKey="name" stroke="#94a3b8" tick={{ fill: '#e2e8f0', fontSize: 12 }} axisLine={false} tickLine={false} />
                    <Tooltip 
                      cursor={{ fill: '#1e293b' }}
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '0.5rem' }}
                      itemStyle={{ color: '#ec4899', fontWeight: 'bold' }}
                    />
                    <Bar dataKey="missing" fill="#ec4899" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex flex-col items-center justify-center border-2 border-dashed border-slate-700/50 rounded-xl bg-slate-800/20">
                  <div className="w-16 h-16 bg-accent/10 rounded-full flex items-center justify-center mb-4 text-accent">
                    <CheckCircle size={32} />
                  </div>
                  <span className="text-slate-300 font-semibold text-lg">Clean Dataset</span>
                  <span className="text-slate-500 text-sm mt-1">No missing values detected</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* DISTRIBUTIONS TAB */}
        {activeTab === 'distributions' && profile.histograms && (
          <div className="glass-card p-8">
            <h3 className="text-xl font-bold tracking-tight mb-6">Feature Distributions</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Object.entries(profile.histograms).map(([feature, hist], idx) => {
                const data = hist.bins.slice(0, -1).map((bin, i) => ({
                  bin: bin.toFixed(2),
                  count: hist.counts[i]
                }));
                return (
                  <div key={idx} className="bg-slate-800/30 border border-slate-700/50 rounded-xl p-4 h-[250px] flex flex-col">
                    <h4 className="text-sm font-bold text-slate-300 mb-2 truncate" title={feature}>{feature}</h4>
                    <div className="flex-1 min-h-0">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={data} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
                          <XAxis dataKey="bin" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} tickLine={false} />
                          <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} tickLine={false} />
                          <Tooltip 
                            cursor={{ fill: '#1e293b' }}
                            contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '0.5rem', fontSize: '12px' }}
                          />
                          <Bar dataKey="count" fill="#3b82f6" radius={[2, 2, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* CORRELATION TAB */}
        {activeTab === 'correlation' && profile.correlation_matrix && (
          <div className="glass-card p-8 overflow-x-auto">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-bold tracking-tight">Correlation Heatmap</h3>
              <span className="text-sm text-slate-500">Pearson Correlation Coefficient</span>
            </div>
            <div className="inline-block min-w-max border border-slate-700/50 rounded-xl overflow-hidden">
              <table className="border-collapse">
                <thead>
                  <tr>
                    <th className="bg-slate-900/80 p-3 text-left text-xs font-bold text-slate-400 sticky left-0 z-20">Feature</th>
                    {Object.keys(profile.correlation_matrix).map(col => (
                      <th key={col} className="bg-slate-900/80 p-2 text-xs font-bold text-slate-400 rotate-[-45deg] whitespace-nowrap h-24 align-bottom w-12">
                        <div className="w-4">{col}</div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(profile.correlation_matrix).map(([rowKey, rowDict]) => (
                    <tr key={rowKey}>
                      <td className="bg-slate-900/90 p-3 text-xs font-bold text-slate-300 sticky left-0 z-10 border-t border-r border-slate-700/50 whitespace-nowrap">
                        {rowKey}
                      </td>
                      {Object.keys(profile.correlation_matrix).map(colKey => {
                        const val = rowDict[colKey];
                        // Color scale from red (-1) to transparent (0) to green (1)
                        const bgColor = val > 0 
                          ? `rgba(16, 185, 129, ${Math.abs(val)})` 
                          : `rgba(239, 68, 68, ${Math.abs(val)})`;
                        return (
                          <td key={colKey} className="w-12 h-12 text-center text-[10px] border-t border-slate-700/20 transition-colors hover:border-slate-400" style={{ backgroundColor: bgColor }}>
                            <span className={Math.abs(val) > 0.5 ? 'text-white font-bold' : 'text-slate-300'}>
                              {val.toFixed(2)}
                            </span>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* BOX PLOTS TAB */}
        {activeTab === 'boxplots' && profile.box_plot_stats && (
          <div className="glass-card p-8">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-bold tracking-tight">Box Plots & Outliers</h3>
              <div className="p-1.5 bg-slate-800 rounded-lg text-slate-400" title="Distribution, IQR, and Outliers">
                <Info size={16} />
              </div>
            </div>
            
            <div className="space-y-8">
              {Object.entries(profile.box_plot_stats).map(([feature, stats], idx) => {
                // Calculate percentages for horizontal positioning (0% to 100%)
                const range = stats.max - stats.min || 1;
                const getPct = (val) => `${((val - stats.min) / range) * 100}%`;
                
                return (
                  <div key={idx} className="bg-slate-800/20 p-5 rounded-xl border border-slate-700/50">
                    <div className="flex justify-between items-end mb-4">
                      <h4 className="font-bold text-slate-200">{feature}</h4>
                      <span className="text-xs text-slate-500">
                        {stats.outliers.length > 0 ? <span className="text-amber-500 font-bold">{stats.outliers.length} Outliers</span> : 'No Outliers'}
                      </span>
                    </div>
                    
                    {/* Visual Box Plot */}
                    <div className="relative h-12 mt-6 mb-2">
                      {/* Range Line (Whisker) */}
                      <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-slate-600 -translate-y-1/2 rounded"></div>
                      
                      {/* Min / Max bounds */}
                      <div className="absolute top-2 bottom-2 w-0.5 bg-slate-500" style={{ left: '0%' }}></div>
                      <div className="absolute top-2 bottom-2 w-0.5 bg-slate-500" style={{ left: '100%' }}></div>
                      
                      {/* IQR Box */}
                      <div 
                        className="absolute top-1 bottom-1 bg-primary/30 border border-primary/60 rounded-sm"
                        style={{ left: getPct(stats.q1), width: `${((stats.q3 - stats.q1) / range) * 100}%` }}
                      ></div>
                      
                      {/* Median Line */}
                      <div 
                        className="absolute top-0 bottom-0 w-1 bg-white shadow-[0_0_10px_rgba(255,255,255,0.5)] z-10"
                        style={{ left: getPct(stats.median), transform: 'translateX(-50%)' }}
                      ></div>
                      
                      {/* Outliers */}
                      {stats.outliers.map((val, i) => (
                        <div 
                          key={i}
                          className="absolute top-1/2 w-1.5 h-1.5 rounded-full bg-amber-500 -translate-y-1/2 -translate-x-1/2 opacity-60"
                          style={{ left: getPct(val) }}
                          title={`Outlier: ${val}`}
                        ></div>
                      ))}
                    </div>
                    
                    {/* Stat Labels */}
                    <div className="flex justify-between text-[10px] text-slate-400 font-mono mt-1">
                      <span>Min: {stats.min.toFixed(2)}</span>
                      <span>Q1: {stats.q1.toFixed(2)}</span>
                      <span className="text-white font-bold">Med: {stats.median.toFixed(2)}</span>
                      <span>Q3: {stats.q3.toFixed(2)}</span>
                      <span>Max: {stats.max.toFixed(2)}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* STATS TAB */}
        {activeTab === 'stats' && (
          <div className="glass-card overflow-hidden">
            <div className="px-8 py-6 border-b border-slate-700/50 flex justify-between items-center bg-slate-800/30">
              <h3 className="text-xl font-bold tracking-tight">Statistical Moments</h3>
              <span className="text-sm text-slate-500">Numerical Features Only</span>
            </div>
            <div className="table-container border-0 rounded-none bg-transparent max-h-[600px] overflow-y-auto">
              <table className="data-table">
                <thead className="sticky top-0 z-10 shadow-md">
                  <tr>
                    <th className="bg-slate-900">Feature</th>
                    <th className="bg-slate-900 text-right">Mean</th>
                    <th className="bg-slate-900 text-right">Median</th>
                    <th className="bg-slate-900 text-right">Std Dev</th>
                    <th className="bg-slate-900 text-right">Min</th>
                    <th className="bg-slate-900 text-right">Max</th>
                    <th className="bg-slate-900 text-right">Skew</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50">
                  {profile.stats_table && profile.stats_table.map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-800/40 transition-colors group">
                      <td className="font-semibold text-slate-200 group-hover:text-primary transition-colors">{row.feature}</td>
                      <td className="text-right font-mono text-sm">{row.mean?.toFixed(3) || '-'}</td>
                      <td className="text-right font-mono text-sm">{row.median?.toFixed(3) || '-'}</td>
                      <td className="text-right font-mono text-sm">{row.std?.toFixed(3) || '-'}</td>
                      <td className="text-right font-mono text-sm text-slate-500">{row.min?.toFixed(3) || '-'}</td>
                      <td className="text-right font-mono text-sm text-slate-500">{row.max?.toFixed(3) || '-'}</td>
                      <td className={`text-right font-mono text-sm ${Math.abs(row.skew || 0) > 1 ? 'text-amber-400 font-bold' : ''}`}>
                        {row.skew?.toFixed(3) || '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* SIGNAL TAB */}
        {activeTab === 'signal' && (
          <div className="glass-card p-8 flex flex-col h-[600px]">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-bold tracking-tight">Feature Signal (Mutual Information)</h3>
              <div className="p-1.5 bg-slate-800 rounded-lg text-slate-400" title="Mutual information between feature and target">
                <Activity size={16} />
              </div>
            </div>
            
            <div className="flex-1 min-h-0">
              {signal && signal.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={signal.slice(0, 20)} layout="vertical" margin={{ top: 10, right: 30, left: 100, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                    <XAxis type="number" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false} />
                    <YAxis type="category" dataKey="feature" stroke="#94a3b8" tick={{ fill: '#e2e8f0', fontSize: 12 }} axisLine={false} tickLine={false} />
                    <Tooltip 
                      cursor={{ fill: '#1e293b' }}
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '0.5rem' }}
                      itemStyle={{ color: '#10b981', fontWeight: 'bold' }}
                      formatter={(value) => [value.toFixed(4), 'MI Score']}
                    />
                    <Bar dataKey="score" fill="#10b981" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex flex-col items-center justify-center border-2 border-dashed border-slate-700/50 rounded-xl bg-slate-800/20">
                  {signal === null ? (
                    <div className="flex items-center gap-2">
                      <Loader2 className="animate-spin text-primary" size={24} />
                      <span className="text-slate-400">Computing mutual information...</span>
                    </div>
                  ) : (
                    <span className="text-slate-400">No feature signal data available. Please ensure task type is selected.</span>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default Analysis;
