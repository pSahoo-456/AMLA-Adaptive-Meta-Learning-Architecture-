import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { ChevronRight, Cpu, Database, Zap, Activity, ShieldCheck, GitCompare, Rocket, UploadCloud } from 'lucide-react';

const PlatformMockup = () => {
  const lineVariants = {
    hidden: { width: 0 },
    visible: { width: "100%", transition: { duration: 1, ease: "easeInOut", delay: 0.5 } }
  };

  return (
    <div className="w-full relative h-[450px] rounded-2xl border border-slate-700/50 bg-slate-900/60 backdrop-blur-xl shadow-2xl shadow-indigo-500/10 overflow-hidden flex items-center justify-center p-8 perspective-1000">
      
      {/* Background Decorative */}
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMSIgY3k9IjEiIHI9IjEiIGZpbGw9InJnYmEoMjU1LDI1NSwyNTUsMC4wMykiLz48L3N2Zz4=')] opacity-50"></div>
      
      <div className="flex flex-col sm:flex-row items-center justify-between w-full max-w-4xl relative z-10 group">
        
        {/* Node 1: Upload */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.2 }}
          className="flex flex-col items-center gap-4 z-20 group-hover:-translate-y-2 transition-transform duration-500"
        >
          <div className="w-20 h-20 rounded-full border-2 border-indigo-500/50 bg-indigo-500/10 flex items-center justify-center shadow-[0_0_20px_rgba(99,102,241,0.3)] backdrop-blur-sm relative">
            <div className="absolute inset-0 rounded-full border border-indigo-400 animate-ping opacity-20"></div>
            <UploadCloud className="text-indigo-400" size={32} />
          </div>
          <div className="text-center bg-slate-900/80 px-4 py-2 rounded-xl border border-slate-800">
            <h3 className="font-bold text-slate-200">1. Upload Dataset</h3>
            <p className="text-xs text-slate-400 mt-1">Raw structured data</p>
          </div>
        </motion.div>

        {/* Path 1 */}
        <div className="hidden sm:block flex-1 mx-4 relative h-0.5 bg-slate-700">
           <motion.div 
             variants={lineVariants} initial="hidden" animate="visible"
             className="absolute top-0 left-0 h-full bg-gradient-to-r from-indigo-500 to-lime-500 shadow-[0_0_10px_rgba(99,102,241,0.5)]" 
           />
        </div>

        {/* Node 2: Engine */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 1.5 }}
          className="flex flex-col items-center gap-4 z-20 my-8 sm:my-0 group-hover:-translate-y-2 transition-transform duration-500 delay-75"
        >
          <div className="w-24 h-24 rounded-2xl border border-lime-500/50 bg-lime-500/10 flex items-center justify-center shadow-[0_0_30px_rgba(132,204,22,0.3)] relative backdrop-blur-sm">
            <div className="absolute inset-0 rounded-2xl border border-lime-400 animate-ping opacity-20" style={{ animationDuration: '2s' }}></div>
            <Cpu className="text-lime-400" size={40} />
          </div>
          <div className="text-center bg-slate-900/80 px-4 py-2 rounded-xl border border-slate-800">
            <h3 className="font-bold text-slate-200">2. Meta-Learning Engine</h3>
            <p className="text-xs text-slate-400 mt-1">Intelligent profiling</p>
          </div>
        </motion.div>

        {/* Path 2 */}
        <div className="hidden sm:block flex-1 mx-4 relative h-0.5 bg-slate-700">
           <motion.div 
             variants={lineVariants} initial="hidden" animate="visible" transition={{ delay: 2.2, duration: 1 }}
             className="absolute top-0 left-0 h-full bg-gradient-to-r from-lime-500 to-accent shadow-[0_0_10px_rgba(16,185,129,0.5)]" 
           />
        </div>

        {/* Node 3: Output */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 3.2 }}
          className="flex flex-col items-center gap-4 z-20 group-hover:-translate-y-2 transition-transform duration-500 delay-150"
        >
          <div className="w-20 h-20 rounded-full border-2 border-accent/50 bg-accent/10 flex items-center justify-center shadow-[0_0_20px_rgba(16,185,129,0.3)] backdrop-blur-sm relative">
             <div className="absolute inset-0 rounded-full border border-accent animate-ping opacity-20" style={{ animationDuration: '3s' }}></div>
            <ShieldCheck className="text-accent" size={32} />
          </div>
          <div className="text-center bg-slate-900/80 px-4 py-2 rounded-xl border border-slate-800">
            <h3 className="font-bold text-slate-200">3. Optimal Output</h3>
            <p className="text-xs text-slate-400 mt-1">&amp; Recommendations</p>
          </div>
        </motion.div>

      </div>
    </div>
  );
};

const LandingPage = () => {
  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { staggerChildren: 0.2, delayChildren: 0.1 }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { 
      y: 0, 
      opacity: 1,
      transition: { type: "spring", stiffness: 100 }
    }
  };

  const glowVariants = {
    initial: { opacity: 0.5, scale: 0.8 },
    animate: { 
      opacity: [0.5, 0.8, 0.5], 
      scale: [0.8, 1.1, 0.8],
      transition: { duration: 8, repeat: Infinity, ease: "easeInOut" }
    }
  };

  return (
    <div className="min-h-screen bg-background overflow-x-hidden text-slate-200 selection:bg-primary/30 font-sans">
      
      {/* Background Ambient Glows */}
      <motion.div 
        variants={glowVariants} initial="initial" animate="animate"
        className="fixed top-[-10%] left-[-10%] w-[40vw] h-[40vw] bg-primary/20 rounded-full blur-[120px] pointer-events-none"
      />
      <motion.div 
        variants={glowVariants} initial="initial" animate="animate"
        className="fixed bottom-[-10%] right-[-10%] w-[40vw] h-[40vw] bg-secondary/20 rounded-full blur-[120px] pointer-events-none"
        style={{ animationDelay: '2s' }}
      />

      {/* Navigation Bar */}
      <nav className="w-full relative z-10 px-8 py-6 flex justify-between items-center max-w-7xl mx-auto">
        <motion.div 
          initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
          className="flex items-center gap-3"
        >
          <div className="bg-primary/20 p-2 rounded-xl border border-primary/30 shadow-[0_0_15px_rgba(99,102,241,0.4)]">
            <Cpu className="text-primary" size={28} />
          </div>
          <div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent tracking-tight">
              AMLA
            </h1>
          </div>
        </motion.div>
        
        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
          <Link to="/dashboard" className="btn btn-primary px-6 py-2.5 rounded-full font-semibold shadow-[0_0_20px_rgba(99,102,241,0.4)] hover:shadow-[0_0_30px_rgba(99,102,241,0.6)]">
            Launch Workspace <ChevronRight size={18} />
          </Link>
        </motion.div>
      </nav>

      {/* Hero Section */}
      <section className="relative z-10 w-full min-h-screen flex flex-col justify-center py-24 px-6 md:px-10">
        <div className="max-w-7xl mx-auto grid grid-cols-1 gap-16 items-center w-full">
          
          {/* Value Proposition */}
          <motion.div 
            className="text-center max-w-4xl mx-auto"
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
          >
            <span className="text-lime-300 uppercase tracking-[2px] text-sm font-bold block mb-4">
              Next-Gen Automated Machine Learning
            </span>
            <h1 className="text-5xl md:text-7xl font-extrabold mt-2 leading-[1.1] bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
              Instantly Characterize Datasets & Predict Optimal Algorithms
            </h1>
            <p className="text-slate-400 text-lg md:text-xl mt-6 leading-relaxed">
              Skip weeks of trial-and-error computational experiments. AMLA maps your data structural blueprints straight to elite configurations using advanced meta-learning features.
            </p>
            
            <div className="mt-10 flex flex-col sm:flex-row gap-4 justify-center">
              <Link to="/dashboard" className="bg-indigo-600 text-white border-none py-3.5 px-8 rounded-xl text-lg font-semibold cursor-pointer shadow-[0_4px_14px_rgba(79,70,229,0.4)] hover:bg-indigo-500 transition-colors">
                Launch Pipeline Studio
              </Link>
              <a href="#features" className="bg-transparent text-slate-50 border border-slate-700 py-3.5 px-8 rounded-xl text-lg font-semibold cursor-pointer hover:bg-slate-800 transition-colors">
                Read Technical Documentation
              </a>
            </div>
          </motion.div>

          {/* The Solution Dynamic Showcase */}
          <motion.div 
            className="w-full relative"
            initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5, duration: 0.8 }}
          >
            <div className="absolute inset-0 bg-gradient-to-t from-background via-transparent to-transparent z-10 pointer-events-none"></div>
            <PlatformMockup />
          </motion.div>

        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="relative z-10 max-w-7xl mx-auto px-8 py-24 border-t border-slate-800/50">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Unleash the Power of Meta-Learning</h2>
          <p className="text-slate-400 max-w-2xl mx-auto">AMLA isn't just an AutoML platform. It's a self-improving intelligence engine that learns from every dataset it encounters.</p>
        </div>

        <motion.div 
          className="grid md:grid-cols-2 lg:grid-cols-4 gap-8"
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
        >
          {[
            { icon: <Database size={24} />, title: "Dataset DNA", desc: "Extracts 21 deep meta-features across 4 layers to create a unique fingerprint of your data.", color: "text-blue-400", bg: "bg-blue-400/10" },
            { icon: <GitCompare size={24} />, title: "Smart Selection", desc: "A Random Forest meta-learner recommends the precise algorithm stack that will dominate.", color: "text-primary", bg: "bg-primary/10" },
            { icon: <ShieldCheck size={24} />, title: "Health Advisor", desc: "Instantly detects data leakage, class imbalance, and 5 types of critical data quality issues.", color: "text-accent", bg: "bg-accent/10" },
            { icon: <Activity size={24} />, title: "Self-Improving", desc: "Feeds experimental results back into the Knowledge Base. AMLA gets smarter every day.", color: "text-secondary", bg: "bg-secondary/10" }
          ].map((feat, i) => (
            <motion.div key={i} variants={itemVariants} className="glass-card p-6 glass-card-hover group">
              <div className={`w-12 h-12 rounded-xl ${feat.bg} flex items-center justify-center ${feat.color} mb-6 transition-transform group-hover:scale-110`}>
                {feat.icon}
              </div>
              <h3 className="text-xl font-bold mb-3 text-slate-200 group-hover:text-white">{feat.title}</h3>
              <p className="text-slate-400 text-sm leading-relaxed">{feat.desc}</p>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-slate-800/50 py-12 text-center text-slate-500 text-sm">
        <div className="flex justify-center items-center gap-2 mb-4">
           <Cpu size={20} className="text-primary/50" />
           <span className="font-semibold text-slate-400">AMLA Dashboard</span>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
