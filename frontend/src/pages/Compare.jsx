import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useAppContext } from '../context/AppContext';
import { useNavigate } from 'react-router-dom';
import { GitCompare, AlertTriangle, ArrowRight, Target, Loader2, Activity } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell, Legend, LineChart, Line, ScatterChart, Scatter, ZAxis } from 'recharts';

function Compare() {
  const { automlResults, taskType } = useAppContext();
  const navigate = useNavigate();
  const [deepDiveData, setDeepDiveData] = useState(null);
  const [loadingMetrics, setLoadingMetrics] = useState(true);

  useEffect(() => {
    if (!automlResults) return;
    
    const fetchBestModelMetrics = async () => {
      try {
        const res = await axios.post('http://localhost:8000/api/automl/best-model-metrics');
        setDeepDiveData(res.data);
      } catch (err) {
        console.error("Failed to fetch best model deep dive metrics", err);
      } finally {
        setLoadingMetrics(false);
      }
    };
    
    fetchBestModelMetrics();
  }, [automlResults]);

  if (!automlResults) {
    return (
      <div className="glass-card border-amber-500/30 bg-amber-500/5 p-10 flex flex-col items-center text-center max-w-2xl mx-auto mt-20">
        <div className="w-16 h-16 bg-amber-500/10 rounded-full flex items-center justify-center mb-6">
          <AlertTriangle className="text-amber-500" size={32} />
        </div>
        <h2 className="text-2xl font-bold text-slate-100 mb-2">No Models to Compare</h2>
        <span className="text-slate-400 mb-8">You need to train models using the AutoML engine before you can compare them.</span>
        <button className="btn btn-primary px-6" onClick={() => navigate('/automl')}>Go to AutoML Engine</button>
      </div>
    );
  }

  const ranked = automlResults.ranked;
  
  // Prepare data for recharts (Grouped Bar Chart)
  const chartData = ranked.map(row => ({
    name: row.model,
    f1: row.f1 || 0,
    accuracy: row.accuracy || 0,
    roc_auc: row.roc_auc || 0,
    r2: row.r2 || 0,
    rmse: row.rmse || 0,
    mae: row.mae || 0
  }));

  const isClassification = taskType === 'classification';

  return (
    <div className="animate-fade-in max-w-7xl mx-auto">
      <div className="flex justify-between items-end mb-10">
        <div>
          <h1 className="text-3xl font-bold mb-3 tracking-tight">Compare Models</h1>
          <p className="text-slate-400 text-lg">Side-by-side performance benchmarking.</p>
        </div>
        <button className="btn btn-primary px-6 py-2.5 shadow-[0_0_20px_rgba(99,102,241,0.3)]" onClick={() => navigate('/improve')}>
          Explain & Improve <ArrowRight size={18} />
        </button>
      </div>

      <div className="glass-card p-8 mb-8">
        <h3 className="text-xl font-bold tracking-tight mb-8">Metrics Comparison</h3>
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
              <XAxis dataKey="name" stroke="#94a3b8" tick={{ fill: '#94a3b8' }} axisLine={false} tickLine={false} />
              <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8' }} axisLine={false} tickLine={false} />
              <Tooltip 
                cursor={{ fill: '#1e293b' }}
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '0.75rem' }}
                itemStyle={{ fontWeight: 'bold' }}
              />
              <Legend wrapperStyle={{ paddingTop: '20px' }} />
              {isClassification ? (
                <>
                  <Bar dataKey="f1" name="F1 Score" fill="#10b981" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="accuracy" name="Accuracy" fill="#6366f1" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="roc_auc" name="ROC AUC" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                </>
              ) : (
                <>
                  <Bar dataKey="r2" name="R2 Score" fill="#10b981" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="rmse" name="RMSE" fill="#ef4444" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="mae" name="MAE" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                </>
              )}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="glass-card overflow-hidden mb-12">
        <div className="px-8 py-6 border-b border-slate-700/50 bg-slate-800/30">
          <h3 className="text-xl font-bold tracking-tight">Leaderboard Summary</h3>
        </div>
        <div className="table-container border-0 rounded-none bg-transparent">
          <table className="data-table">
            <thead className="bg-slate-900/80">
              <tr>
                <th>Model</th>
                <th className="text-right">Status</th>
                {isClassification ? (
                  <>
                    <th className="text-right">F1 Score</th>
                    <th className="text-right">Accuracy</th>
                    <th className="text-right">ROC AUC</th>
                    <th className="text-right">Precision</th>
                    <th className="text-right">Recall</th>
                  </>
                ) : (
                  <>
                    <th className="text-right">R2 Score</th>
                    <th className="text-right">RMSE</th>
                    <th className="text-right">MAE</th>
                  </>
                )}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {ranked.map((row, idx) => (
                <tr key={idx} className={`transition-colors ${idx === 0 ? "bg-accent/10" : "hover:bg-slate-800/40"}`}>
                  <td className={`font-semibold ${idx === 0 ? "text-accent" : "text-slate-200"}`}>{row.model} {idx === 0 && <span className="ml-2 text-xs bg-accent/20 text-accent px-2 py-0.5 rounded-full border border-accent/30">Best</span>}</td>
                  <td className="text-right text-success">{row.status}</td>
                  {isClassification ? (
                    <>
                      <td className="text-right font-mono text-sm">{row.f1?.toFixed(4)}</td>
                      <td className="text-right font-mono text-sm text-slate-400">{row.accuracy?.toFixed(4)}</td>
                      <td className="text-right font-mono text-sm text-slate-400">{row.roc_auc?.toFixed(4)}</td>
                      <td className="text-right font-mono text-sm text-slate-400">{row.precision?.toFixed(4)}</td>
                      <td className="text-right font-mono text-sm text-slate-400">{row.recall?.toFixed(4)}</td>
                    </>
                  ) : (
                    <>
                      <td className="text-right font-mono text-sm">{row.r2?.toFixed(4)}</td>
                      <td className="text-right font-mono text-sm text-slate-400">{row.rmse?.toFixed(4)}</td>
                      <td className="text-right font-mono text-sm text-slate-400">{row.mae?.toFixed(4)}</td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* BEST MODEL DEEP DIVE */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold mb-6 tracking-tight flex items-center gap-3">
          <Target className="text-accent" /> Best Model Deep Dive
        </h2>
        
        {loadingMetrics ? (
          <div className="glass-card p-12 flex flex-col items-center justify-center text-center">
            <Loader2 className="animate-spin text-primary mb-4" size={40} />
            <span className="text-slate-400">Loading evaluation matrices...</span>
          </div>
        ) : !deepDiveData ? (
          <div className="glass-card p-8 text-center text-slate-400">
            Deep dive metrics unavailable for this model.
          </div>
        ) : isClassification ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Confusion Matrix */}
            <div className="glass-card p-8">
              <h3 className="text-xl font-bold tracking-tight mb-6">Confusion Matrix</h3>
              <div className="flex flex-col items-center justify-center min-h-[300px]">
                {deepDiveData.confusion_matrix && deepDiveData.confusion_matrix.length > 0 ? (
                  <div className="grid gap-1 bg-slate-700/30 p-1 rounded-xl" 
                       style={{ gridTemplateColumns: `repeat(${deepDiveData.confusion_matrix[0].length}, minmax(0, 1fr))` }}>
                    {deepDiveData.confusion_matrix.map((row, i) => (
                      <React.Fragment key={`row-${i}`}>
                        {row.map((val, j) => {
                          const maxVal = Math.max(...deepDiveData.confusion_matrix.flat());
                          const intensity = val / maxVal;
                          // Green for true positives/negatives (diagonal), Red for errors
                          const isCorrect = i === j;
                          const bgColor = isCorrect 
                            ? `rgba(16, 185, 129, ${Math.max(0.1, intensity)})`
                            : `rgba(239, 68, 68, ${Math.max(0.1, intensity)})`;
                            
                          return (
                            <div key={`cell-${i}-${j}`} 
                                 className="w-16 h-16 sm:w-20 sm:h-20 flex items-center justify-center rounded-lg text-white font-bold text-lg shadow-sm border border-slate-700/50"
                                 style={{ backgroundColor: bgColor }}
                                 title={`True: ${i}, Predicted: ${j}`}>
                              {val}
                            </div>
                          );
                        })}
                      </React.Fragment>
                    ))}
                  </div>
                ) : (
                  <span className="text-slate-400">Confusion matrix not available.</span>
                )}
                <div className="mt-6 flex justify-between w-full max-w-[200px] text-xs font-bold text-slate-400 uppercase tracking-widest">
                  <span>Actual</span>
                  <span>Predicted</span>
                </div>
              </div>
            </div>
            
            {/* ROC Curve */}
            <div className="glass-card p-8">
              <h3 className="text-xl font-bold tracking-tight mb-6">ROC Curve (Binary)</h3>
              <div className="h-[300px]">
                {deepDiveData.roc_curve ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={deepDiveData.roc_curve} margin={{ top: 5, right: 20, left: -20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="fpr" type="number" domain={[0, 1]} stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                      <YAxis dataKey="tpr" type="number" domain={[0, 1]} stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                      <Tooltip 
                        cursor={{ stroke: '#475569', strokeWidth: 1, strokeDasharray: '3 3' }}
                        contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '0.5rem' }}
                        formatter={(value) => [value.toFixed(4), '']}
                        labelFormatter={(label) => `FPR: ${label.toFixed(4)}`}
                      />
                      {/* Random guess line */}
                      <Line dataKey="fpr" stroke="#64748b" strokeWidth={1} strokeDasharray="5 5" dot={false} activeDot={false} name="Random" />
                      {/* Actual ROC */}
                      <Line dataKey="tpr" type="stepAfter" stroke="#10b981" strokeWidth={3} dot={false} activeDot={{ r: 6, fill: '#10b981' }} name="ROC" />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center border-2 border-dashed border-slate-700/50 rounded-xl bg-slate-800/20">
                    <Activity className="text-slate-500 mb-2" size={32} />
                    <span className="text-slate-400 text-sm">ROC Curve only available for binary classification.</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Regression Scatter */}
            <div className="glass-card p-8">
              <h3 className="text-xl font-bold tracking-tight mb-6">Actual vs Predicted</h3>
              <div className="h-[300px]">
                {deepDiveData.actual_vs_predicted ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: -10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis type="number" dataKey="actual" name="Actual" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                      <YAxis type="number" dataKey="predicted" name="Predicted" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '0.5rem' }} />
                      <Scatter name="Predictions" data={deepDiveData.actual_vs_predicted} fill="#6366f1" opacity={0.7} />
                    </ScatterChart>
                  </ResponsiveContainer>
                ) : (
                  <span className="text-slate-400">Data unavailable.</span>
                )}
              </div>
            </div>
            
            {/* Residuals */}
            <div className="glass-card p-8">
              <h3 className="text-xl font-bold tracking-tight mb-6">Residuals (Error)</h3>
              <div className="h-[300px]">
                {deepDiveData.residuals ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={deepDiveData.residuals} margin={{ top: 10, right: 20, bottom: 10, left: -20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                      <XAxis dataKey="residual" tick={false} axisLine={false} />
                      <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false} />
                      <Tooltip cursor={{ fill: '#1e293b' }} contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '0.5rem' }} />
                      <Bar dataKey="residual">
                        {deepDiveData.residuals.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.residual > 0 ? '#ef4444' : '#10b981'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <span className="text-slate-400">Data unavailable.</span>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default Compare;
