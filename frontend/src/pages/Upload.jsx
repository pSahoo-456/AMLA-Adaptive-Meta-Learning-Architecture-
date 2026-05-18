import React, { useState, useCallback } from 'react';
import axios from 'axios';
import { UploadCloud, CheckCircle, AlertTriangle, FileSpreadsheet, ArrowRight, Loader2 } from 'lucide-react';
import { useAppContext } from '../context/AppContext';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';

function Upload() {
  const { 
    setFile, file, 
    setPreviewData, previewData,
    targetColumn, setTargetColumn,
    taskType, setTaskType
  } = useAppContext();
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const onDrop = useCallback(async (acceptedFiles) => {
    const selectedFile = acceptedFiles[0];
    if (!selectedFile) return;
    
    setFile(selectedFile);
    setLoading(true);
    setError(null);
    setTargetColumn('');
    setTaskType('');
    
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await axios.post('http://localhost:8000/api/eda/preview', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setPreviewData(res.data);
      if (res.data.columns && res.data.columns.length > 0) {
        const lastCol = res.data.columns[res.data.columns.length - 1].name;
        handleTargetSelection(lastCol, selectedFile);
      }
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || 'Failed to upload and parse file');
    } finally {
      setLoading(false);
    }
  }, [setFile, setPreviewData, setTargetColumn, setTaskType]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls']
    },
    maxFiles: 1
  });

  const handleTargetSelection = async (colName, currentFile = file) => {
    setTargetColumn(colName);
    if (!currentFile || !colName) return;
    
    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('target_column', colName);
    
    try {
      const res = await axios.post('http://localhost:8000/api/eda/infer-task', formData);
      setTaskType(res.data.task_type);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="animate-fade-in max-w-6xl mx-auto">
      <div className="mb-10">
        <h1 className="text-3xl font-bold mb-3 tracking-tight">Dataset Ingestion</h1>
        <p className="text-slate-400 max-w-2xl">
          Upload your raw dataset. AMLA will automatically parse the schema, infer data types, and identify the predictive modeling task.
        </p>
      </div>
      
      {!previewData && !loading && (
        <div 
          {...getRootProps()} 
          className={`border-2 border-dashed rounded-2xl p-16 text-center cursor-pointer transition-all duration-300 ease-in-out ${
            isDragActive 
              ? 'border-primary bg-primary/10 shadow-[0_0_30px_rgba(99,102,241,0.2)]' 
              : 'border-slate-700 bg-surface/50 hover:border-primary/50 hover:bg-surface'
          }`}
        >
          <input {...getInputProps()} />
          <div className={`w-20 h-20 mx-auto rounded-full flex items-center justify-center mb-6 transition-colors duration-300 ${
            isDragActive ? 'bg-primary/20 text-primary scale-110' : 'bg-slate-800 text-slate-400'
          }`}>
            <UploadCloud size={40} />
          </div>
          <h3 className="text-2xl font-bold mb-3 text-slate-200">
            {isDragActive ? "Drop the dataset here..." : "Drag & drop your dataset"}
          </h3>
          <p className="text-slate-500 mb-6 max-w-md mx-auto">
            Supports standard tabular formats including .CSV and .XLSX files up to 25MB.
          </p>
          <button className="btn btn-secondary px-6 py-2">
            Browse Files
          </button>
        </div>
      )}

      {loading && (
        <div className="glass-card p-16 flex flex-col items-center justify-center text-center">
          <Loader2 className="animate-spin text-primary mb-6" size={56} />
          <h2 className="text-2xl font-bold mb-2">Analyzing Schema</h2>
          <p className="text-slate-400">Parsing columns and inferring data types...</p>
        </div>
      )}
      
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-5 mb-8 flex items-start gap-4">
          <AlertTriangle className="text-red-500 shrink-0 mt-0.5" />
          <div>
            <h4 className="text-red-500 font-bold mb-1">Ingestion Failed</h4>
            <span className="text-red-400/90 text-sm">{error}</span>
          </div>
          <button onClick={() => setError(null)} className="ml-auto text-red-500 hover:text-red-400 text-sm font-medium">Dismiss</button>
        </div>
      )}

      {previewData && !loading && (
        <div className="space-y-6">
          <div className="glass-card overflow-hidden">
            <div className="px-6 py-5 border-b border-slate-700/50 flex flex-wrap justify-between items-center bg-slate-800/20">
              <div className="flex items-center gap-3">
                <FileSpreadsheet className="text-primary" />
                <h2 className="text-xl font-bold tracking-tight">Dataset Schema</h2>
              </div>
              <div className="flex gap-3">
                <span className="badge badge-primary px-3 py-1 text-sm bg-primary/20 text-primary border-primary/30">
                  {previewData.total_rows.toLocaleString()} Rows
                </span>
                <span className="badge px-3 py-1 text-sm bg-slate-800 text-slate-300 border border-slate-700">
                  {previewData.total_cols} Columns
                </span>
              </div>
            </div>
            
            <div className="table-container border-0 rounded-none bg-transparent">
              <table className="data-table">
                <thead>
                  <tr>
                    {previewData.columns.map(col => (
                      <th key={col.name} className="bg-slate-900/40">
                        <div className="font-semibold text-slate-200">{col.name}</div>
                        <div className="text-xs text-primary font-mono mt-1 opacity-80">{col.dtype}</div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50">
                  {previewData.preview.map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                      {previewData.columns.map(col => (
                        <td key={col.name} className="py-3 font-mono text-sm">
                          {row[col.name] !== null ? String(row[col.name]) : <span className="text-slate-600 italic">NaN</span>}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="glass-card p-6">
              <label className="block text-sm font-bold text-slate-400 uppercase tracking-wider mb-3">
                1. Select Target Variable
              </label>
              <select 
                className="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-slate-200 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-colors appearance-none" 
                value={targetColumn} 
                onChange={(e) => handleTargetSelection(e.target.value)}
              >
                <option value="">-- Choose target column --</option>
                {previewData.columns.map(col => (
                  <option key={col.name} value={col.name}>{col.name}</option>
                ))}
              </select>
              <p className="text-xs text-slate-500 mt-3">
                The column you want the machine learning model to predict.
              </p>
            </div>
            
            <div className="glass-card p-6">
              <label className="block text-sm font-bold text-slate-400 uppercase tracking-wider mb-3">
                2. Inferred Engine Task
              </label>
              <div className="bg-slate-900/30 border border-slate-700/50 rounded-xl p-4 flex items-center justify-between h-[50px] mt-1">
                {taskType ? (
                  <>
                    <span className="text-lg font-bold capitalize text-accent flex items-center gap-2">
                      <CheckCircle size={20} />
                      {taskType}
                    </span>
                    <span className="text-xs text-slate-500 font-medium bg-slate-800 px-2 py-1 rounded">Auto-detected</span>
                  </>
                ) : (
                  <span className="text-slate-500 italic text-sm">Awaiting target selection...</span>
                )}
              </div>
            </div>
          </div>

          {targetColumn && taskType && (
            <div className="flex justify-end pt-6 gap-4 border-t border-slate-700/50 mt-8">
              <button className="btn btn-secondary px-6 py-2.5" onClick={() => { setPreviewData(null); setFile(null); }}>
                Change Dataset
              </button>
              <button className="btn btn-primary px-8 py-2.5 shadow-[0_0_20px_rgba(99,102,241,0.4)] flex items-center gap-2" onClick={() => navigate('/analysis')}>
                Proceed to Analysis <ArrowRight size={18} />
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default Upload;
