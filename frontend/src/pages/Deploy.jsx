import React, { useState } from 'react';
import axios from 'axios';
import { useAppContext } from '../context/AppContext';
import { Rocket, Download, Code, FileText, Server, AlertTriangle, Loader2 } from 'lucide-react';

function Deploy() {
  const { automlResults } = useAppContext();
  const [deployFiles, setDeployFiles] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const generateDeployment = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post('http://localhost:8000/api/automl/deploy');
      setDeployFiles(res.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || "Failed to generate deployment package.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = (filename, content) => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  if (!automlResults) {
    return (
      <div className="glass-card border-amber-500/30 bg-amber-500/5 p-10 flex flex-col items-center text-center max-w-2xl mx-auto mt-20">
        <div className="w-16 h-16 bg-amber-500/10 rounded-full flex items-center justify-center mb-6">
          <AlertTriangle className="text-amber-500" size={32} />
        </div>
        <h2 className="text-2xl font-bold text-slate-100 mb-2">No Model Ready for Deployment</h2>
        <span className="text-slate-400 mb-8">You must successfully train a model using the AutoML Engine before you can deploy it.</span>
      </div>
    );
  }

  return (
    <div className="animate-fade-in max-w-7xl mx-auto pb-12">
      <div className="flex justify-between items-end mb-10">
        <div>
          <h1 className="text-3xl font-bold mb-3 tracking-tight">Production Deployment</h1>
          <p className="text-slate-400 text-lg">Generate production-ready serving code and container definitions for your best model.</p>
        </div>
        <button 
          className="btn btn-primary px-6 shadow-[0_0_20px_rgba(99,102,241,0.3)] flex items-center gap-2"
          onClick={generateDeployment}
          disabled={loading}
        >
          {loading ? <Loader2 className="animate-spin" size={20} /> : <Rocket size={20} />}
          {loading ? "Generating Package..." : "Generate Deployment Package"}
        </button>
      </div>

      {error && (
        <div className="p-4 mb-8 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 flex items-center gap-3">
          <AlertTriangle size={20} />
          {error}
        </div>
      )}

      {deployFiles ? (
        <div className="space-y-8 animate-fade-in">
          <div className="p-6 bg-success/10 border border-success/30 rounded-xl flex items-center gap-4 text-success">
            <div className="p-3 bg-success/20 rounded-full">
              <Rocket size={24} />
            </div>
            <div>
              <h3 className="text-lg font-bold">Deployment Package Ready</h3>
              <p className="text-sm opacity-90">Model artifacts have been exported and serving files are generated. Download the files below to deploy via Docker.</p>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-8">
            {/* FastAPI Server Code */}
            <div className="glass-card overflow-hidden">
              <div className="flex justify-between items-center px-6 py-4 border-b border-slate-700/50 bg-slate-800/40">
                <div className="flex items-center gap-3">
                  <Server className="text-primary" size={20} />
                  <h3 className="font-bold tracking-wide">serve_model_fastapi.py</h3>
                </div>
                <button 
                  onClick={() => handleDownload("serve_model_fastapi.py", deployFiles.api_file)}
                  className="btn bg-slate-700 hover:bg-slate-600 px-4 py-1.5 text-sm flex items-center gap-2 rounded-lg"
                >
                  <Download size={16} /> Download
                </button>
              </div>
              <div className="p-0 bg-[#0d1117] overflow-x-auto max-h-[500px]">
                <pre className="p-6 text-sm font-mono text-slate-300">
                  <code>{deployFiles.api_file}</code>
                </pre>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Dockerfile */}
              <div className="glass-card overflow-hidden">
                <div className="flex justify-between items-center px-6 py-4 border-b border-slate-700/50 bg-slate-800/40">
                  <div className="flex items-center gap-3">
                    <Code className="text-accent" size={20} />
                    <h3 className="font-bold tracking-wide">Dockerfile</h3>
                  </div>
                  <button 
                    onClick={() => handleDownload("Dockerfile", deployFiles.dockerfile)}
                    className="btn bg-slate-700 hover:bg-slate-600 px-4 py-1.5 text-sm flex items-center gap-2 rounded-lg"
                  >
                    <Download size={16} /> Download
                  </button>
                </div>
                <div className="p-0 bg-[#0d1117] overflow-x-auto h-[350px]">
                  <pre className="p-6 text-sm font-mono text-slate-300">
                    <code>{deployFiles.dockerfile}</code>
                  </pre>
                </div>
              </div>

              {/* Requirements */}
              <div className="glass-card overflow-hidden">
                <div className="flex justify-between items-center px-6 py-4 border-b border-slate-700/50 bg-slate-800/40">
                  <div className="flex items-center gap-3">
                    <FileText className="text-secondary" size={20} />
                    <h3 className="font-bold tracking-wide">requirements-serving.txt</h3>
                  </div>
                  <button 
                    onClick={() => handleDownload("requirements-serving.txt", deployFiles.requirements)}
                    className="btn bg-slate-700 hover:bg-slate-600 px-4 py-1.5 text-sm flex items-center gap-2 rounded-lg"
                  >
                    <Download size={16} /> Download
                  </button>
                </div>
                <div className="p-0 bg-[#0d1117] overflow-x-auto h-[350px]">
                  <pre className="p-6 text-sm font-mono text-slate-300">
                    <code>{deployFiles.requirements}</code>
                  </pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="glass-card p-12 flex flex-col items-center justify-center text-center border-dashed border-2 border-slate-700">
          <Rocket className="text-slate-600 mb-4" size={48} />
          <h3 className="text-xl font-bold text-slate-300 mb-2">Ready for Hand-off</h3>
          <p className="text-slate-500 max-w-md">
            Click the button above to export your trained `best_pipeline` and auto-generate the FastAPI inference server and Docker build specs.
          </p>
        </div>
      )}
    </div>
  );
}

export default Deploy;
