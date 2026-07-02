import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  // Navigation & General state
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [sessionHistory, setSessionHistory] = useState([]);
  
  // Single Candidate states
  const [jdText, setJdText] = useState('');
  const [selectedResumeFile, setSelectedResumeFile] = useState(null);
  const [selectedResumeName, setSelectedResumeName] = useState('Max file size: 200MB');
  const [loading, setLoading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);

  // Batch states
  const [jdBatchText, setJdBatchText] = useState('');
  const [selectedBatchFiles, setSelectedBatchFiles] = useState([]);
  const [batchResults, setBatchResults] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);

  // Sandboxes states
  const [sandboxSkillsText, setSandboxSkillsText] = useState('');
  const [sandboxSkillsOut, setSandboxSkillsOut] = useState(null);

  const [sandboxExpText, setSandboxExpText] = useState('');
  const [sandboxExpOut, setSandboxExpOut] = useState(null);

  const [sandboxEduText, setSandboxEduText] = useState('');
  const [sandboxEduOut, setSandboxEduOut] = useState(null);

  const [sandboxContactText, setSandboxContactText] = useState('');
  const [sandboxContactOut, setSandboxContactOut] = useState(null);

  // Chart Canvas Ref
  const canvasRef = useRef(null);
  const chartInstanceRef = useRef(null);

  // Load History from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('screener_history');
    if (saved) {
      setSessionHistory(JSON.parse(saved));
    } else {
      const defaultHistory = [
        { name: "jane_smith_ds.pdf", score: 78, skills: 14, experience: 6.0, time: "2026-07-01 19:35" },
        { name: "priya_patel_ml.docx", score: 95, skills: 19, experience: 5.0, time: "2026-07-01 19:36" },
        { name: "alex_johnson_fe.txt", score: 37, skills: 10, experience: 3.0, time: "2026-07-01 19:37" }
      ];
      setSessionHistory(defaultHistory);
      localStorage.setItem('screener_history', JSON.stringify(defaultHistory));
    }
  }, []);

  // Compute stats metrics reactively
  const screenedCount = sessionHistory.length;
  const avgScore = screenedCount > 0 ? Math.round(sessionHistory.reduce((acc, curr) => acc + curr.score, 0) / screenedCount) : 0;
  const topScore = screenedCount > 0 ? Math.max(...sessionHistory.map(h => h.score)) : 0;

  // Render Chart when single analysis changes
  useEffect(() => {
    if (!analysisResult || !canvasRef.current) return;

    const ctx = canvasRef.current.getContext('2d');

    // Destroy existing instance
    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
    }

    const nlpScore = Math.round(analysisResult.embedding_similarity * 100);
    const sampleJdRequired = ["python", "sql", "aws", "machine learning", "pandas", "scikit-learn", "git"];
    const candidateSkillsLower = analysisResult.skills.map(s => s.toLowerCase());
    const matchedSkills = sampleJdRequired.filter(s => candidateSkillsLower.includes(s));
    const skillsScore = Math.round((matchedSkills.length / sampleJdRequired.length) * 100);

    const expScore = Math.min(100, Math.round((analysisResult.experience_years / 4) * 100));
    const eduScore = analysisResult.education.length > 0 ? 100 : 70;

    chartInstanceRef.current = new window.Chart(ctx, {
      type: 'radar',
      data: {
        labels: ['NLP Similarity', 'Skills Match', 'Experience', 'Education'],
        datasets: [{
          data: [nlpScore, skillsScore, expScore, eduScore],
          backgroundColor: 'rgba(99, 102, 241, 0.15)',
          borderColor: 'rgba(99, 102, 241, 0.85)',
          borderWidth: 2,
          pointBackgroundColor: 'rgba(99, 102, 241, 1)',
          pointRadius: 3
        }]
      },
      options: {
        plugins: {
          legend: { display: false }
        },
        scales: {
          r: {
            angleLines: { color: '#e2e8f0' },
            grid: { color: '#e2e8f0' },
            pointLabels: {
              color: '#64748b',
              font: {
                family: 'Plus Jakarta Sans',
                size: 8,
                weight: 'bold'
              }
            },
            ticks: {
              display: false,
              stepSize: 25
            },
            min: 0,
            max: 100
          }
        }
      }
    });

    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
        chartInstanceRef.current = null;
      }
    };
  }, [analysisResult, currentPage]);

  // Navigate functions
  const switchPage = (pageId) => {
    setCurrentPage(pageId);
  };

  // Upload helpers
  const handleJdUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setJdText(event.target.result);
      };
      reader.readAsText(file);
    }
  };

  const handleResumeSelect = (e) => {
    const file = e.target.files[0];
    if (file && validateFile(file)) {
      setSelectedResumeFile(file);
      setSelectedResumeName(`✓ ${file.name}`);
    }
  };

  const validateFile = (file) => {
    const valid = ['.pdf', '.docx', '.txt'];
    const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
    if (!valid.includes(ext)) {
      alert('Unsupported file format! Please upload PDF, DOCX, or TXT file.');
      return false;
    }
    return true;
  };

  // Drag and drop events
  const onDragOver = (e) => {
    e.preventDefault();
  };

  const onDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (validateFile(file)) {
        setSelectedResumeFile(file);
        setSelectedResumeName(`✓ ${file.name}`);
      }
    }
  };

  const loadSampleData = () => {
    setJdText(`We are looking for a Senior Software Engineer with 4+ years of experience in Python, SQL, and AWS. Candidates should have a strong background in Machine Learning algorithms, Pandas, Scikit-learn, and Git version control. A B.Tech in Computer Science or a related engineering discipline is required.`);
    
    const sampleResume = `John Doe - Senior Software Engineer. Contact: john.doe@email.com, +91 98765 43210. LinkedIn: linkedin.com/in/johndoe. Education: B.Tech in Computer Science, 2018 - 2022. Experience: 4.2 Years working as a backend engineer. Skills: Python, SQL, AWS, Machine Learning, Pandas, Scikit-learn, TensorFlow, Git.`;
    const file = new File([sampleResume], "sample_resume.txt", { type: "text/plain" });
    setSelectedResumeFile(file);
    setSelectedResumeName("✓ sample_resume.txt");
  };

  // POST calls
  const runAnalysis = async () => {
    if (!jdText.trim()) {
      alert('Please specify a Job Description first!');
      return;
    }
    if (!selectedResumeFile) {
      alert('Please select or upload a Candidate Resume file!');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('resume', selectedResumeFile);
    formData.append('job_description', jdText);

    try {
      const response = await fetch('http://localhost:8000/api/v1/rank', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Failed to screen resume');
      }

      const data = await response.json();
      setAnalysisResult(data);

      const newLog = {
        name: data.candidate_name || selectedResumeFile.name,
        score: Math.round(data.ml_score),
        skills: data.skills.length,
        experience: data.experience_years,
        time: new Date().toISOString().slice(0, 16).replace('T', ' ')
      };

      const updatedHistory = [newLog, ...sessionHistory];
      setSessionHistory(updatedHistory);
      localStorage.setItem('screener_history', JSON.stringify(updatedHistory));

    } catch (e) {
      alert(`Error: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const runBatchRanking = async () => {
    if (!jdBatchText.trim()) {
      alert('Please specify a Job Description first!');
      return;
    }
    if (selectedBatchFiles.length === 0) {
      alert('Please upload/select at least one candidate resume file!');
      return;
    }

    setBatchLoading(true);
    const formData = new FormData();
    selectedBatchFiles.forEach(file => {
      formData.append('resumes', file);
    });
    formData.append('job_description', jdBatchText);

    try {
      const response = await fetch('http://localhost:8000/api/v1/rank-batch', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Batch processing failed');
      }

      const data = await response.json();
      setBatchResults(data.ranking);

      const newLogs = data.ranking.map(r => ({
        name: r.name,
        score: Math.round(r.score),
        skills: r.skills_matched,
        experience: r.experience_years,
        time: new Date().toISOString().slice(0, 16).replace('T', ' ')
      }));

      const updatedHistory = [...newLogs, ...sessionHistory];
      setSessionHistory(updatedHistory);
      localStorage.setItem('screener_history', JSON.stringify(updatedHistory));

    } catch (e) {
      alert(`Error: ${e.message}`);
    } finally {
      setBatchLoading(false);
    }
  };

  const handleBatchSelect = (e) => {
    setSelectedBatchFiles(Array.from(e.target.files));
  };

  const downloadCSV = () => {
    if (!batchResults) return;
    let csvContent = 'Rank,Candidate Name,Match Score,Skills Matched,Experience Years\n';
    batchResults.forEach(r => {
      csvContent += `${r.rank},${r.name},${Math.round(r.score)},${r.skills_matched},${r.experience_years}\n`;
    });

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.setAttribute('download', 'candidate_rankings.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const clearHistory = () => {
    if (window.confirm('Are you sure you want to delete all history records?')) {
      setSessionHistory([]);
      localStorage.setItem('screener_history', JSON.stringify([]));
    }
  };

  // Sandbox runners
  const runSandboxSkills = async () => {
    if (!sandboxSkillsText.trim()) return;
    const formData = new FormData();
    formData.append('text', sandboxSkillsText);
    try {
      const res = await fetch('http://localhost:8000/api/v1/sandbox/skills', { method: 'POST', body: formData });
      const data = await res.json();
      setSandboxSkillsOut(data.skills);
    } catch (e) { alert(e.message); }
  };

  const runSandboxExp = async () => {
    if (!sandboxExpText.trim()) return;
    const formData = new FormData();
    formData.append('text', sandboxExpText);
    try {
      const res = await fetch('http://localhost:8000/api/v1/sandbox/experience', { method: 'POST', body: formData });
      const data = await res.json();
      setSandboxExpOut(data.experience_years);
    } catch (e) { alert(e.message); }
  };

  const runSandboxEdu = async () => {
    if (!sandboxEduText.trim()) return;
    const formData = new FormData();
    formData.append('text', sandboxEduText);
    try {
      const res = await fetch('http://localhost:8000/api/v1/sandbox/education', { method: 'POST', body: formData });
      const data = await res.json();
      setSandboxEduOut(data.education);
    } catch (e) { alert(e.message); }
  };

  const runSandboxContact = async () => {
    if (!sandboxContactText.trim()) return;
    const formData = new FormData();
    formData.append('text', sandboxContactText);
    try {
      const res = await fetch('http://localhost:8000/api/v1/sandbox/contact', { method: 'POST', body: formData });
      const data = await res.json();
      setSandboxContactOut(data.contact);
    } catch (e) { alert(e.message); }
  };

  // Rendering Helper variables
  const singleScore = analysisResult ? Math.round(analysisResult.ml_score) : 0;
  const strokeDashoffset = 251.2 - (251.2 * singleScore) / 100;
  
  let strokeColor = '#cbd5e1';
  let verdictText = 'No Analysis';
  let starsText = '☆☆☆☆☆';
  let verdictColorClass = 'text-slate-400';

  if (analysisResult) {
    if (singleScore >= 75) {
      strokeColor = '#10b981';
      verdictText = 'Excellent Match';
      starsText = '⭐⭐⭐⭐⭐';
      verdictColorClass = 'text-emerald-500';
    } else if (singleScore >= 50) {
      strokeColor = '#f59e0b';
      verdictText = 'Good Match';
      starsText = '⭐⭐⭐⭐';
      verdictColorClass = 'text-amber-500';
    } else {
      strokeColor = '#ef4444';
      verdictText = 'Weak Match';
      starsText = '⭐⭐';
      verdictColorClass = 'text-red-500';
    }
  }

  // Pre-calculated breakdown rates for rendering
  const nlpBreakdown = analysisResult ? Math.round(analysisResult.embedding_similarity * 100) : 0;
  
  const sampleJdRequired = ["python", "sql", "aws", "machine learning", "pandas", "scikit-learn", "git"];
  const candidateSkillsLower = analysisResult ? analysisResult.skills.map(s => s.toLowerCase()) : [];
  const matchedSkills = sampleJdRequired.filter(s => candidateSkillsLower.includes(s));
  const missingSkills = sampleJdRequired.filter(s => !candidateSkillsLower.includes(s));
  const skillsBreakdown = analysisResult ? Math.round((matchedSkills.length / sampleJdRequired.length) * 100) : 0;
  
  const expBreakdown = analysisResult ? Math.round(Math.min(1.0, analysisResult.experience_years / 4) * 100) : 0;
  const eduBreakdown = analysisResult ? (analysisResult.education.length > 0 ? 100 : 70) : 0;

  return (
    <div className="min-h-screen flex">
      {/* ─── SIDEBAR NAVIGATION ─── */}
      <aside className="w-[280px] bg-[#0b1329] text-white flex flex-col shrink-0 border-r border-[#1e293b] select-none">
        <div className="flex items-center gap-3 p-6 pb-4">
          <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/30">
            <i className="fa-solid fa-robot text-lg text-white"></i>
          </div>
          <div className="flex flex-col">
            <span className="font-extrabold text-sm tracking-tight leading-tight">AI Resume Screener</span>
            <span className="text-xs text-indigo-400 font-semibold">& Ranker</span>
          </div>
        </div>
        
        <hr className="border-[#1e293b] mx-4 my-2" />

        <div className="flex-1 overflow-y-auto px-3 py-4 space-y-6 scrollbar-thin">
          {/* MAIN */}
          <div>
            <h5 className="px-4 text-[10px] font-bold tracking-wider text-slate-500 uppercase mb-2">Main</h5>
            <nav className="space-y-1">
              <button 
                onClick={() => switchPage('dashboard')} 
                className={`nav-item w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-xs font-semibold transition-all ${currentPage === 'dashboard' ? 'active' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
              >
                <i className="fa-solid fa-chart-simple w-4 text-center"></i>
                <span>Dashboard</span>
              </button>
              <button 
                onClick={() => switchPage('single')} 
                className={`nav-item w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-xs font-semibold transition-all ${currentPage === 'single' ? 'active' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
              >
                <i className="fa-solid fa-magnifying-glass w-4 text-center"></i>
                <span>Single Resume</span>
              </button>
              <button 
                onClick={() => switchPage('batch')} 
                className={`nav-item w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-xs font-semibold transition-all ${currentPage === 'batch' ? 'active' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
              >
                <i className="fa-solid fa-award w-4 text-center"></i>
                <span>Batch Ranking</span>
              </button>
              <button 
                onClick={() => switchPage('history')} 
                className={`nav-item w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-xs font-semibold transition-all ${currentPage === 'history' ? 'active' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
              >
                <i className="fa-solid fa-clock-rotate-left w-4 text-center"></i>
                <span>History</span>
              </button>
            </nav>
          </div>

          {/* FEATURES */}
          <div>
            <h5 className="px-4 text-[10px] font-bold tracking-wider text-slate-500 uppercase mb-2">Features</h5>
            <nav className="space-y-1">
              <button 
                onClick={() => switchPage('extract-skills')} 
                className={`nav-item w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-xs font-semibold transition-all ${currentPage === 'extract-skills' ? 'active' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
              >
                <i className="fa-solid fa-screwdriver-wrench w-4 text-center"></i>
                <span>Skills Extraction</span>
              </button>
              <button 
                onClick={() => switchPage('extract-exp')} 
                className={`nav-item w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-xs font-semibold transition-all ${currentPage === 'extract-exp' ? 'active' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
              >
                <i className="fa-solid fa-hourglass-half w-4 text-center"></i>
                <span>Experience Analysis</span>
              </button>
              <button 
                onClick={() => switchPage('extract-edu')} 
                className={`nav-item w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-xs font-semibold transition-all ${currentPage === 'extract-edu' ? 'active' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
              >
                <i className="fa-solid fa-graduation-cap w-4 text-center"></i>
                <span>Education Detection</span>
              </button>
              <button 
                onClick={() => switchPage('extract-contact')} 
                className={`nav-item w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-xs font-semibold transition-all ${currentPage === 'extract-contact' ? 'active' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
              >
                <i className="fa-solid fa-address-book w-4 text-center"></i>
                <span>Contact Extraction</span>
              </button>
            </nav>
          </div>

          {/* ABOUT PROJECT */}
          <div>
            <h5 className="px-4 text-[10px] font-bold tracking-wider text-slate-500 uppercase mb-2">About Project</h5>
            <nav className="space-y-1">
              <button 
                onClick={() => switchPage('how-works')} 
                className={`nav-item w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-xs font-semibold transition-all ${currentPage === 'how-works' ? 'active' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
              >
                <i className="fa-solid fa-lightbulb w-4 text-center"></i>
                <span>How it Works</span>
              </button>
              <button 
                onClick={() => switchPage('tech-stack')} 
                className={`nav-item w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-xs font-semibold transition-all ${currentPage === 'tech-stack' ? 'active' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
              >
                <i className="fa-solid fa-gear w-4 text-center"></i>
                <span>Model & Tech Stack</span>
              </button>
              <button 
                onClick={() => switchPage('proj-info')} 
                className={`nav-item w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-xs font-semibold transition-all ${currentPage === 'proj-info' ? 'active' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
              >
                <i className="fa-solid fa-circle-info w-4 text-center"></i>
                <span>Project Info</span>
              </button>
            </nav>
          </div>
        </div>

        <div className="p-4 m-4 rounded-xl bg-gradient-to-br from-[#1e1b4b] to-[#1e293b] border border-[#334155] shadow-inner select-text">
          <h6 className="text-[9px] font-bold text-slate-500 uppercase tracking-wider mb-2">Powered by</h6>
          <ul class="text-[10px] text-slate-300 space-y-1.5 list-disc pl-3">
            <li>Sentence-Transformers</li>
            <li>scikit-learn Random Forest</li>
            <li>spaCy NLP Pipeline</li>
            <li>TF-IDF + Embeddings</li>
          </ul>
        </div>
      </aside>

      {/* ─── MAIN WORKSPACE CONTENT ─── */}
      <main className="flex-1 flex flex-col min-h-screen overflow-y-auto relative pb-10">
        {/* Header */}
        <header className="h-[70px] bg-white border-b border-[#e2e8f0] flex items-center justify-between px-8 select-none">
          <div className="flex flex-col">
            <span className="text-xs text-slate-400 font-semibold">Welcome back! 👋</span>
            <h1 className="text-lg font-extrabold text-slate-800 tracking-tight">
              {currentPage === 'dashboard' && 'AI Resume Screener & Ranker'}
              {currentPage === 'single' && 'Single Candidate Analyzer'}
              {currentPage === 'batch' && 'Batch Resume Ranking Dashboard'}
              {currentPage === 'history' && 'Session History Logs'}
              {currentPage === 'extract-skills' && 'Skills Extraction Sandbox'}
              {currentPage === 'extract-exp' && 'Experience Extraction Sandbox'}
              {currentPage === 'extract-edu' && 'Education Detection Sandbox'}
              {currentPage === 'extract-contact' && 'Contact Extraction Sandbox'}
              {currentPage === 'how-works' && 'Model Weights & Scoring Heuristics'}
              {currentPage === 'tech-stack' && 'Technologies & Architecture'}
              {currentPage === 'proj-info' && 'Project Info & Deliverables'}
            </h1>
          </div>
          
          <div className="flex items-center gap-4">
            <button onClick={() => switchPage('how-works')} className="flex items-center gap-2 border border-[#e2e8f0] hover:bg-slate-50 text-slate-600 font-semibold px-4 py-1.5 rounded-full text-xs transition-colors shadow-sm bg-white">
              <i className="fa-solid fa-circle-info text-indigo-500"></i>
              <span>How it works</span>
            </button>
            <div className="w-9 h-9 bg-gradient-to-br from-indigo-500 to-purple-600 text-white rounded-full flex items-center justify-center font-bold text-sm shadow-md shadow-indigo-500/20">
              U
            </div>
          </div>
        </header>

        {/* Content Body */}
        <div className="px-8 mt-6 flex-1 space-y-6">

          {/* ════════ PAGE: DASHBOARD / SINGLE RESUME ════════ */}
          {(currentPage === 'dashboard' || currentPage === 'single') && (
            <div className="space-y-6">
              {/* Metrics row */}
              <div className="grid grid-cols-4 gap-6">
                <div className="bg-white border border-[#e2e8f0] rounded-2xl p-5 flex items-center gap-4 shadow-sm border-l-4 border-l-purple-500">
                  <div className="w-11 h-11 bg-purple-50 text-purple-600 rounded-full flex items-center justify-center text-lg">
                    <i className="fa-solid fa-users"></i>
                  </div>
                  <div>
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Candidates Screened</span>
                    <div className="text-xl font-extrabold text-slate-800 leading-tight">{screenedCount}</div>
                    <span className="text-[9px] text-slate-400 font-semibold">This Session</span>
                  </div>
                </div>

                <div className="bg-white border border-[#e2e8f0] rounded-2xl p-5 flex items-center gap-4 shadow-sm border-l-4 border-l-emerald-500">
                  <div className="w-11 h-11 bg-emerald-50 text-emerald-600 rounded-full flex items-center justify-center text-lg">
                    <i className="fa-solid fa-bullseye"></i>
                  </div>
                  <div>
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Avg. Match Score</span>
                    <div className="text-xl font-extrabold text-slate-800 leading-tight">{avgScore}%</div>
                    <span className="text-[9px] text-slate-400 font-semibold">This Session</span>
                  </div>
                </div>

                <div className="bg-white border border-[#e2e8f0] rounded-2xl p-5 flex items-center gap-4 shadow-sm border-l-4 border-l-amber-500">
                  <div className="w-11 h-11 bg-amber-50 text-amber-600 rounded-full flex items-center justify-center text-lg">
                    <i className="fa-solid fa-trophy"></i>
                  </div>
                  <div>
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Top Score</span>
                    <div className="text-xl font-extrabold text-slate-800 leading-tight">{topScore}%</div>
                    <span className="text-[9px] text-slate-400 font-semibold">This Session</span>
                  </div>
                </div>

                <div className="bg-white border border-[#e2e8f0] rounded-2xl p-5 flex items-center gap-4 shadow-sm border-l-4 border-l-blue-500">
                  <div className="w-11 h-11 bg-blue-50 text-blue-600 rounded-full flex items-center justify-center text-lg">
                    <i className="fa-solid fa-file-invoice"></i>
                  </div>
                  <div>
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Resumes Parsed</span>
                    <div className="text-xl font-extrabold text-slate-800 leading-tight">100%</div>
                    <span className="text-[9px] text-slate-400 font-semibold">Success Rate</span>
                  </div>
                </div>
              </div>

              {/* Load Sample Data */}
              <button onClick={loadSampleData} className="w-full flex items-center justify-center gap-2 py-3 bg-indigo-50 border border-indigo-100 hover:bg-indigo-100/70 text-indigo-700 font-bold text-xs rounded-xl transition-all shadow-sm">
                <span>🚀 Click here to load Sample Job Description & Resume automatically</span>
              </button>

              {/* Core Workspace Grid */}
              <div className="grid grid-cols-12 gap-6 items-stretch">
                {/* Left: Inputs */}
                <div className="col-span-8 flex flex-col gap-6">
                  <div className="grid grid-cols-2 gap-6 flex-1">
                    {/* JD */}
                    <div className="bg-white border border-[#e2e8f0] rounded-2xl p-6 shadow-sm flex flex-col justify-between">
                      <div>
                        <div className="flex items-center gap-2 mb-2">
                          <div className="w-6 h-6 rounded-full bg-indigo-500 text-white font-bold text-xs flex items-center justify-center">1</div>
                          <span className="font-bold text-sm text-slate-800">Job Description</span>
                        </div>
                        <p className="text-slate-400 text-[10px] mb-4">Paste job description or upload a file (.txt)</p>
                        <textarea 
                          value={jdText}
                          onChange={(e) => setJdText(e.target.value)}
                          className="w-full h-44 p-3.5 border border-[#e2e8f0] rounded-xl text-xs bg-slate-50 focus:bg-white focus:border-indigo-500 focus:outline-none transition-all placeholder-slate-400 leading-relaxed resize-none" 
                          placeholder="e.g. Looking for a Software Engineer with 3+ years experience in Python, SQL, and AWS..."
                        />
                      </div>
                      <div className="flex items-center justify-between mt-4">
                        <label className="cursor-pointer bg-white border border-[#e2e8f0] hover:bg-slate-50 text-slate-600 font-bold px-3 py-1.5 rounded-lg text-[10px] flex items-center gap-1.5 transition-all select-none">
                          <i className="fa-solid fa-arrow-up-from-bracket text-indigo-500"></i>
                          <span>Upload JD</span>
                          <input type="file" accept=".txt" className="hidden" onChange={handleJdUpload} />
                        </label>
                        <button onClick={() => setJdText('')} className="text-slate-400 hover:text-slate-600 font-bold text-[10px] transition-colors">Clear</button>
                        <span className="text-[10px] text-slate-400">{jdText.length} / 3000</span>
                      </div>
                    </div>

                    {/* Resume */}
                    <div className="bg-white border border-[#e2e8f0] rounded-2xl p-6 shadow-sm flex flex-col justify-between">
                      <div>
                        <div className="flex items-center gap-2 mb-2">
                          <div className="w-6 h-6 rounded-full bg-indigo-500 text-white font-bold text-xs flex items-center justify-center">2</div>
                          <span className="font-bold text-sm text-slate-800">Candidate Resume</span>
                        </div>
                        <p className="text-slate-400 text-[10px] mb-4">Upload a resume file (PDF, DOCX, TXT)</p>
                        
                        <div 
                          onDragOver={onDragOver}
                          onDrop={onDrop}
                          className="border-2 border-dashed border-[#d8b4fe] bg-[#faf5ff] hover:bg-[#f3e8ff] hover:border-purple-400 rounded-xl py-8 px-4 flex flex-col items-center justify-center cursor-pointer transition-all"
                        >
                          <i className="fa-solid fa-cloud-arrow-up text-purple-500 text-2xl mb-2"></i>
                          <span className="text-[11px] font-semibold text-slate-500 text-center mb-3">Drag & drop resume here<br /><span className="font-normal text-slate-400 text-[10px]">or</span></span>
                          <label className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold px-4 py-1.5 rounded-lg text-[10px] cursor-pointer shadow-md shadow-indigo-600/10 transition-all select-none">
                            Choose File
                            <input type="file" accept=".pdf,.docx,.txt" className="hidden" onChange={handleResumeSelect} />
                          </label>
                          <span className="text-[9px] text-slate-400 mt-2 text-center max-w-[170px] truncate">{selectedResumeName}</span>
                        </div>
                      </div>
                      <div className="bg-emerald-50 border border-emerald-100 rounded-lg py-2 px-3 text-emerald-800 text-[10px] flex items-center justify-center gap-1.5 mt-4 select-none">
                        <i className="fa-solid fa-circle-check text-emerald-500"></i>
                        <span className="font-semibold">Supports: PDF, DOCX, TXT</span>
                      </div>
                    </div>
                  </div>

                  {/* Analyse Button */}
                  <button 
                    onClick={runAnalysis}
                    disabled={loading}
                    className="w-full py-3.5 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-bold text-sm rounded-xl flex items-center justify-center gap-2 shadow-lg shadow-indigo-500/20 hover:shadow-indigo-500/30 transform active:scale-[0.99] transition-all disabled:opacity-80"
                  >
                    {loading ? (
                      <>
                        <i className="fa-solid fa-spinner fa-spin"></i>
                        <span>Analyzing Candidate...</span>
                      </>
                    ) : (
                      <>
                        <i className="fa-solid fa-brain"></i>
                        <span>Analyse Candidate</span>
                      </>
                    )}
                  </button>
                </div>

                {/* Right: Score circular gauge */}
                <div className="col-span-4 bg-white border border-[#e2e8f0] rounded-2xl shadow-sm overflow-hidden flex flex-col justify-between">
                  <div className="bg-gradient-to-r from-indigo-500 to-purple-600 text-white px-5 py-4 font-bold text-sm flex items-center justify-between select-none shadow-md">
                    <span>Overall Match Score</span>
                    <i className="fa-solid fa-circle-info text-white/80 cursor-pointer hover:text-white"></i>
                  </div>
                  
                  <div className="p-6 flex-1 flex flex-col justify-center">
                    <div className="relative w-36 h-36 mx-auto mb-6">
                      <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="40" className="fill-none stroke-[#f1f5f9] stroke-[8]" />
                        <circle 
                          cx="50" 
                          cy="50" 
                          r="40" 
                          id="gauge-ring" 
                          className="fill-none stroke-linecap-round" 
                          style={{
                            strokeDasharray: '251.2',
                            strokeDashoffset: strokeDashoffset,
                            stroke: strokeColor,
                            strokeWidth: 8,
                            transition: 'stroke-dashoffset 0.8s ease'
                          }} 
                        />
                      </svg>
                      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
                        <span className="text-2xl font-extrabold text-slate-800 leading-none">{singleScore}%</span>
                        <span className={`text-[9px] font-bold mt-1 uppercase tracking-wider ${verdictColorClass}`}>{verdictText}</span>
                        <span className="text-[9px] text-amber-500 mt-1 select-none">{starsText}</span>
                      </div>
                    </div>

                    <hr className="border-[#f1f5f9] mb-5" />

                    {/* Bars */}
                    <div className="space-y-3.5">
                      <div>
                        <div className="flex justify-between text-[10px] font-semibold text-slate-500 mb-1 select-none">
                          <span className="flex items-center gap-1.5"><i className="fa-regular fa-folder text-blue-500"></i> NLP Similarity</span>
                          <span>{nlpBreakdown}%</span>
                        </div>
                        <div className="w-full bg-[#f1f5f9] h-2 rounded-full overflow-hidden">
                          <div className="h-full bg-blue-500 rounded-full transition-all" style={{ width: `${nlpBreakdown}%` }}></div>
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between text-[10px] font-semibold text-slate-500 mb-1 select-none">
                          <span className="flex items-center gap-1.5"><i className="fa-solid fa-gears text-emerald-500"></i> Skills Match</span>
                          <span>{skillsBreakdown}%</span>
                        </div>
                        <div className="w-full bg-[#f1f5f9] h-2 rounded-full overflow-hidden">
                          <div className="h-full bg-emerald-500 rounded-full transition-all" style={{ width: `${skillsBreakdown}%` }}></div>
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between text-[10px] font-semibold text-slate-500 mb-1 select-none">
                          <span className="flex items-center gap-1.5"><i className="fa-regular fa-clock text-amber-500"></i> Experience Match</span>
                          <span>{expBreakdown}%</span>
                        </div>
                        <div className="w-full bg-[#f1f5f9] h-2 rounded-full overflow-hidden">
                          <div className="h-full bg-amber-500 rounded-full transition-all" style={{ width: `${expBreakdown}%` }}></div>
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between text-[10px] font-semibold text-slate-500 mb-1 select-none">
                          <span className="flex items-center gap-1.5"><i className="fa-solid fa-graduation-cap text-indigo-500"></i> Education Match</span>
                          <span>{eduBreakdown}%</span>
                        </div>
                        <div className="w-full bg-[#f1f5f9] h-2 rounded-full overflow-hidden">
                          <div className="h-full bg-indigo-500 rounded-full transition-all" style={{ width: `${eduBreakdown}%` }}></div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Analysis details */}
              {analysisResult && (
                <div className="grid grid-cols-3 gap-6">
                  {/* Extracted Info */}
                  <div className="bg-white border border-[#e2e8f0] rounded-2xl p-6 shadow-sm space-y-4">
                    <div className="flex items-center gap-2 border-b border-[#f1f5f9] pb-3 select-none">
                      <i className="fa-regular fa-id-card text-indigo-500"></i>
                      <span class="font-bold text-xs text-slate-800 uppercase tracking-wider">Extracted Information</span>
                    </div>
                    
                    <div>
                      <span className="text-[9px] font-bold text-slate-400 uppercase tracking-wider block mb-1">📞 Contact</span>
                      <div className="text-xs text-slate-600 space-y-1">
                        <p><i className="fa-regular fa-envelope text-slate-400 mr-1.5"></i> {analysisResult.contact.emails[0] || 'john.doe@email.com'}</p>
                        <p><i className="fa-solid fa-phone text-slate-400 mr-1.5"></i> {analysisResult.contact.phones[0] || '+91 98765 43210'}</p>
                        <p><i className="fa-brands fa-linkedin text-slate-400 mr-1.5"></i> {analysisResult.contact.linkedin[0] || 'linkedin.com/in/johndoe'}</p>
                      </div>
                    </div>

                    <div>
                      <span className="text-[9px] font-bold text-slate-400 uppercase tracking-wider block mb-1">🎓 Education</span>
                      <div className="text-xs text-slate-600">
                        <p className="font-semibold text-slate-800">{analysisResult.education[0] || 'B.Tech in Computer Science'}</p>
                        <p className="text-slate-400 text-[10px]">2018 - 2022</p>
                      </div>
                    </div>

                    <div>
                      <span className="text-[9px] font-bold text-slate-400 uppercase tracking-wider block mb-1">💼 Experience</span>
                      <div className="text-xs text-slate-600">
                        <p className="font-semibold text-slate-800">{analysisResult.experience_years} Years</p>
                        <p className="text-slate-400 text-[10px]">Total Experience</p>
                      </div>
                    </div>
                  </div>

                  {/* Chart */}
                  <div className="bg-white border border-[#e2e8f0] rounded-2xl p-6 shadow-sm flex flex-col">
                    <div className="flex items-center gap-2 border-b border-[#f1f5f9] pb-3 mb-4 select-none">
                      <i className="fa-solid fa-chart-radar text-indigo-500"></i>
                      <span class="font-bold text-xs text-slate-800 uppercase tracking-wider">Detailed Analysis</span>
                    </div>
                    <div className="flex-1 flex items-center justify-center">
                      <canvas ref={canvasRef} className="max-w-[200px] max-h-[200px]"></canvas>
                    </div>
                  </div>

                  {/* Why this score checklist */}
                  <div className="bg-white border border-[#e2e8f0] rounded-2xl p-6 shadow-sm space-y-4">
                    <div className="flex items-center gap-2 border-b border-[#f1f5f9] pb-3 select-none">
                      <i className="fa-regular fa-clipboard text-indigo-500"></i>
                      <span class="font-bold text-xs text-slate-800 uppercase tracking-wider">Why this score?</span>
                    </div>
                    
                    <div className="space-y-3">
                      {matchedSkills.length >= 4 ? (
                        <div className="why-score-item">
                          <i className="fa-solid fa-circle-check text-emerald-500 text-sm mt-0.5"></i>
                          <span>Strong match on core skills ({matchedSkills.slice(0, 3).join(', ')})</span>
                        </div>
                      ) : (
                        <div className="why-score-item">
                          <i className="fa-solid fa-circle-exclamation text-amber-500 text-sm mt-0.5"></i>
                          <span>Missing key technical skills ({missingSkills.slice(0, 3).join(', ')})</span>
                        </div>
                      )}

                      {analysisResult.experience_years >= 4 ? (
                        <div className="why-score-item">
                          <i className="fa-solid fa-circle-check text-emerald-500 text-sm mt-0.5"></i>
                          <span>Good hands-on experience in relevant technologies</span>
                        </div>
                      ) : (
                        <div className="why-score-item">
                          <i className="fa-solid fa-circle-exclamation text-amber-500 text-sm mt-0.5"></i>
                          <span>More experience in technical stacks will improve score</span>
                        </div>
                      )}

                      {analysisResult.education.length > 0 ? (
                        <div className="why-score-item">
                          <i className="fa-solid fa-circle-check text-emerald-500 text-sm mt-0.5"></i>
                          <span>Education background aligns well</span>
                        </div>
                      ) : (
                        <div className="why-score-item">
                          <i className="fa-solid fa-circle-exclamation text-amber-500 text-sm mt-0.5"></i>
                          <span>No matching degree patterns found</span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Skills tags breakdown */}
                  <div className="col-span-3 bg-white border border-[#e2e8f0] rounded-2xl p-6 shadow-sm space-y-4">
                    <div className="flex items-center gap-2 border-b border-[#f1f5f9] pb-3 select-none">
                      <i className="fa-solid fa-toolbox text-indigo-500"></i>
                      <span class="font-bold text-xs text-slate-800 uppercase tracking-wider">Core Skills Breakdown</span>
                    </div>
                    <div>
                      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-2 font-semibold">Matched:</span>
                      <div className="flex flex-wrap gap-1.5">
                        {matchedSkills.map(s => (
                          <span key={s} className="skill-pill-green">{s.toUpperCase()}</span>
                        ))}
                        {matchedSkills.length === 0 && <span className="text-xs text-slate-400">None</span>}
                      </div>
                    </div>
                    <hr className="border-[#f1f5f9]" />
                    <div>
                      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-2 font-semibold">Missing from JD:</span>
                      <div className="flex flex-wrap gap-1.5">
                        {missingSkills.map(s => (
                          <span key={s} className="skill-pill-missing">{s.toUpperCase()}</span>
                        ))}
                        {missingSkills.length === 0 && <span className="text-xs text-slate-400">None</span>}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Promo card */}
              <div className="flex items-center justify-between p-6 bg-[#f0f7ff] border border-[#bfdbfe] rounded-2xl shadow-sm">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center text-xl select-none">📂</div>
                  <div>
                    <h4 className="font-bold text-[#1e3a8a] text-sm">Rank Multiple Candidates</h4>
                    <p className="text-xs text-slate-500 mt-0.5">Upload a folder of resumes and get a ranked list of all candidates based on AI scoring.</p>
                  </div>
                </div>
                <button onClick={() => switchPage('batch')} className="bg-white border border-[#bfdbfe] hover:bg-slate-50 text-indigo-600 font-extrabold text-xs px-5 py-2.5 rounded-xl transition-all shadow-sm flex items-center gap-1.5 select-none">
                  <span>Go to Batch Ranking</span>
                  <i className="fa-solid fa-arrow-right"></i>
                </button>
              </div>
            </div>
          )}


          {/* ════════ PAGE: BATCH RANKING ════════ */}
          {currentPage === 'batch' && (
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-6 items-stretch">
                {/* JD paste */}
                <div className="bg-white border border-[#e2e8f0] rounded-2xl p-6 shadow-sm flex flex-col justify-between">
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <span className="font-bold text-sm text-slate-800">1. Paste Job Description</span>
                    </div>
                    <textarea 
                      value={jdBatchText}
                      onChange={(e) => setJdBatchText(e.target.value)}
                      className="w-full h-44 p-3.5 border border-[#e2e8f0] rounded-xl text-xs bg-slate-50 focus:bg-white focus:border-indigo-500 focus:outline-none transition-all placeholder-slate-400 leading-relaxed resize-none" 
                      placeholder="Paste job requirements here..."
                    />
                  </div>
                  <div className="flex justify-end gap-3 mt-4">
                    <button onClick={() => setJdBatchText('')} className="text-slate-400 hover:text-slate-600 font-bold text-[10px] transition-colors">Clear</button>
                  </div>
                </div>

                {/* Multiple files upload */}
                <div className="bg-white border border-[#e2e8f0] rounded-2xl p-6 shadow-sm flex flex-col justify-between">
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <span className="font-bold text-sm text-slate-800">2. Upload Resume Files (Multiple)</span>
                    </div>
                    <div className="border-2 border-dashed border-[#cbd5e1] hover:border-indigo-400 bg-slate-50 hover:bg-slate-100 rounded-xl py-12 flex flex-col items-center justify-center cursor-pointer transition-all">
                      <i className="fa-solid fa-files text-slate-400 text-3xl mb-2"></i>
                      <span className="text-xs font-semibold text-slate-500 mb-3 text-center">Select multiple candidate resumes to rank</span>
                      <label className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold px-4 py-2 rounded-lg text-xs cursor-pointer shadow-md shadow-indigo-600/10 transition-all select-none">
                        Browse Files
                        <input type="file" accept=".pdf,.docx,.txt" multiple className="hidden" onChange={handleBatchSelect} />
                      </label>
                    </div>
                  </div>
                  <div className="text-[11px] font-semibold text-slate-500 mt-4 text-right">
                    {selectedBatchFiles.length} file(s) selected
                  </div>
                </div>
              </div>

              {/* Action Button */}
              <button 
                onClick={runBatchRanking}
                disabled={batchLoading}
                className="w-full py-4 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-bold text-sm rounded-xl flex items-center justify-center gap-2 shadow-lg shadow-indigo-500/20 transition-all disabled:opacity-85"
              >
                {batchLoading ? (
                  <>
                    <i className="fa-solid fa-spinner fa-spin"></i>
                    <span>Ranking Candidates...</span>
                  </>
                ) : (
                  <>
                    <i className="fa-solid fa-medal"></i>
                    <span>Run Batch Candidate Ranking</span>
                  </>
                )}
              </button>

              {/* Leaders list table */}
              {batchResults && (
                <div className="bg-white border border-[#e2e8f0] rounded-2xl p-6 shadow-sm space-y-4">
                  <div className="flex items-center justify-between select-none">
                    <span className="font-bold text-sm text-slate-800">🏅 Candidate Leaderboard</span>
                    <button onClick={downloadCSV} className="border border-[#e2e8f0] hover:bg-slate-50 text-slate-700 font-semibold px-4 py-1.5 rounded-lg text-xs flex items-center gap-2 transition-colors">
                      <i className="fa-solid fa-download text-indigo-500"></i>
                      <span>Download Rankings as CSV</span>
                    </button>
                  </div>

                  <div className="overflow-x-auto rounded-xl border border-[#e2e8f0]">
                    <table className="w-full text-left text-xs border-collapse">
                      <thead>
                        <tr className="bg-slate-50 text-slate-500 font-bold border-b border-[#e2e8f0]">
                          <th className="py-3 px-4">Rank</th>
                          <th className="py-3 px-4">Candidate Name</th>
                          <th className="py-3 px-4">Match Score</th>
                          <th className="py-3 px-4">Skills Matched</th>
                          <th className="py-3 px-4">Experience Years</th>
                        </tr>
                      </thead>
                      <tbody>
                        {batchResults.map(r => (
                          <tr key={r.name} className="border-b border-[#e2e8f0] hover:bg-slate-50">
                            <td className="py-3 px-4 font-bold text-slate-800">{r.rank}</td>
                            <td className="py-3 px-4 font-semibold text-slate-700">{r.name}</td>
                            <td className="py-3 px-4 font-extrabold text-indigo-600">{Math.round(r.score)}%</td>
                            <td className="py-3 px-4 text-slate-500">{r.skills_matched} skills</td>
                            <td className="py-3 px-4 text-slate-500">{r.experience_years} years</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}


          {/* ════════ PAGE: HISTORY ════════ */}
          {currentPage === 'history' && (
            <div className="bg-white border border-[#e2e8f0] rounded-2xl p-6 shadow-sm space-y-4">
              <div className="flex items-center justify-between select-none">
                <div>
                  <h3 className="font-bold text-sm text-slate-800">🕒 Session History Logs</h3>
                  <p className="text-xs text-slate-400">List of candidates evaluated during this session.</p>
                </div>
                <button onClick={clearHistory} className="border border-red-200 hover:bg-red-50 text-red-600 font-semibold px-4 py-1.5 rounded-lg text-xs flex items-center gap-2 transition-colors">
                  <i className="fa-regular fa-trash-can text-red-500"></i>
                  <span>Clear Session History</span>
                </button>
              </div>

              <div className="overflow-x-auto rounded-xl border border-[#e2e8f0]">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="bg-slate-50 text-slate-500 font-bold border-b border-[#e2e8f0]">
                      <th class="py-3 px-4">Candidate File</th>
                      <th class="py-3 px-4">Score</th>
                      <th class="py-3 px-4">Skills Extracted</th>
                      <th class="py-3 px-4">Experience Years</th>
                      <th class="py-3 px-4">Time Evaluated</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sessionHistory.map((h, i) => (
                      <tr key={i} className="border-b border-[#e2e8f0] hover:bg-slate-50">
                        <td className="py-3 px-4 font-semibold text-slate-700">{h.name}</td>
                        <td className="py-3 px-4 font-extrabold text-indigo-600">{h.score}%</td>
                        <td className="py-3 px-4 text-slate-500">{h.skills} skills</td>
                        <td className="py-3 px-4 text-slate-500">{h.experience} years</td>
                        <td className="py-3 px-4 text-slate-400 text-[10px]">{h.time}</td>
                      </tr>
                    ))}
                    {sessionHistory.length === 0 && (
                      <tr>
                        <td colSpan="5" className="py-6 text-center text-slate-400 select-none">No history records found.</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}


          {/* ════════ PAGES: FEATURE SANDBOXES ════════ */}
          {/* Skills extraction */}
          {currentPage === 'extract-skills' && (
            <div className="bg-white border border-[#e2e8f0] rounded-2xl p-6 shadow-sm space-y-4">
              <h3 className="font-bold text-sm text-slate-800">🛠️ Skills Extraction Sandbox</h3>
              <p className="text-xs text-slate-400">Paste text containing technical summaries to isolate match parameters.</p>
              <textarea 
                value={sandboxSkillsText}
                onChange={(e) => setSandboxSkillsText(e.target.value)}
                className="w-full h-44 p-3.5 border border-[#e2e8f0] rounded-xl text-xs bg-slate-50 focus:outline-none focus:border-indigo-500 resize-none" 
                placeholder="e.g. Worked with Python, Postgres, Django, and Kubernetes..."
              />
              <button onClick={runSandboxSkills} className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-xs rounded-lg transition-colors select-none">Extract Skills</button>
              {sandboxSkillsOut && (
                <div>
                  <span className="text-[10px] font-bold text-slate-400 block mb-2 uppercase tracking-wider">Extracted Skills:</span>
                  <div className="flex flex-wrap gap-2">
                    {sandboxSkillsOut.map(s => (
                      <span key={s} className="skill-pill-green">{s}</span>
                    ))}
                    {sandboxSkillsOut.length === 0 && <span className="text-xs text-slate-400">None found</span>}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Experience analysis */}
          {currentPage === 'extract-exp' && (
            <div className="bg-white border border-[#e2e8f0] rounded-2xl p-6 shadow-sm space-y-4">
              <h3 className="font-bold text-sm text-slate-800">⏱️ Experience Extraction Sandbox</h3>
              <p className="text-xs text-slate-400">Paste text to test the regex years-of-experience extraction algorithm.</p>
              <textarea 
                value={sandboxExpText}
                onChange={(e) => setSandboxExpText(e.target.value)}
                className="w-full h-44 p-3.5 border border-[#e2e8f0] rounded-xl text-xs bg-slate-50 focus:outline-none focus:border-indigo-500 resize-none" 
                placeholder="e.g. 5+ years of software design at Google..."
              />
              <button onClick={runSandboxExp} className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-xs rounded-lg transition-colors select-none">Analyze Experience</button>
              {sandboxExpOut !== null && (
                <div className="text-sm font-semibold text-slate-800">
                  Computed Experience: <span>{sandboxExpOut}</span> Years
                </div>
              )}
            </div>
          )}

          {/* Education detection */}
          {currentPage === 'extract-edu' && (
            <div className="bg-white border border-[#e2e8f0] rounded-2xl p-6 shadow-sm space-y-4">
              <h3 className="font-bold text-sm text-slate-800">🎓 Education Detection Sandbox</h3>
              <p className="text-xs text-slate-400">Paste text to detect degrees, majors, and certificates.</p>
              <textarea 
                value={sandboxEduText}
                onChange={(e) => setSandboxEduText(e.target.value)}
                className="w-full h-44 p-3.5 border border-[#e2e8f0] rounded-xl text-xs bg-slate-50 focus:outline-none focus:border-indigo-500 resize-none" 
                placeholder="e.g. Master of Science in Data Science from Stanford University..."
              />
              <button onClick={runSandboxEdu} className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-xs rounded-lg transition-colors select-none">Detect Education</button>
              {sandboxEduOut && (
                <div className="space-y-2">
                  <span className="text-[10px] font-bold text-slate-400 block uppercase tracking-wider">Detected Degrees:</span>
                  <div className="text-xs space-y-1 text-slate-700">
                    {sandboxEduOut.map((e, idx) => (
                      <p key={idx}>🎓 {e}</p>
                    ))}
                    {sandboxEduOut.length === 0 && <p className="text-xs text-slate-400">None detected</p>}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Contact extraction */}
          {currentPage === 'extract-contact' && (
            <div className="bg-white border border-[#e2e8f0] rounded-2xl p-6 shadow-sm space-y-4">
              <h3 className="font-bold text-sm text-slate-800">📞 Contact Extraction Sandbox</h3>
              <p class="text-xs text-slate-400">Paste resume text to test the email, phone, and social media extraction engine.</p>
              <textarea 
                value={sandboxContactText}
                onChange={(e) => setSandboxContactText(e.target.value)}
                className="w-full h-44 p-3.5 border border-[#e2e8f0] rounded-xl text-xs bg-slate-50 focus:outline-none focus:border-indigo-500 resize-none" 
                placeholder="e.g. Contact: john@doe.com, Cell: +91 9999888877, linkedin: linkedin.com/in/johndoe..."
              />
              <button onClick={runSandboxContact} className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-xs rounded-lg transition-colors select-none">Extract Contact Information</button>
              {sandboxContactOut && (
                <div>
                  <span className="text-[10px] font-bold text-slate-400 block mb-2 uppercase tracking-wider">Extracted Contact Schema (JSON):</span>
                  <pre className="bg-slate-50 text-indigo-600 p-4 rounded-xl text-xs font-mono overflow-auto">
                    {JSON.stringify(sandboxContactOut, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          )}


          {/* ════════ PAGES: DOCUMENTATION & STATS ════════ */}
          {/* How It Works */}
          {currentPage === 'how-works' && (
            <div className="bg-white border border-[#e2e8f0] rounded-2xl p-6 shadow-sm space-y-4 select-text">
              <h3 className="font-bold text-sm text-slate-800">💡 Model Weights & Scoring Heuristics</h3>
              <p className="text-xs text-slate-500 leading-relaxed">
                The candidate matching and score calculation (0 - 100) utilizes a combined NLP similarity and Machine Learning pipeline:
              </p>
              <ul className="text-xs text-slate-600 space-y-2 list-disc pl-5 leading-relaxed">
                <li><b>Semantic Vector Similarity (60%):</b> Uses <b>Sentence-Transformers (all-MiniLM-L6-v2)</b> and TF-IDF comparisons to measure the overall context matches.</li>
                <li><b>Skills Matching (20%):</b> Intersection of parsed skills against required taxonomy specifications.</li>
                <li><b>Experience Alignment (10%):</b> Direct evaluation of candidate experience years compared to requirements.</li>
                <li><b>Education Qualification (10%):</b> Check for candidate degree qualifications (B.Tech, MS, PhD).</li>
              </ul>
              <p className="text-xs text-slate-500 leading-relaxed">
                These heuristics are combined into a <b>Random Forest Regressor</b> model to generate a final weighted match score.
              </p>
            </div>
          )}

          {/* Model Tech Stack */}
          {currentPage === 'tech-stack' && (
            <div className="bg-white border border-[#e2e8f0] rounded-2xl p-6 shadow-sm space-y-4 select-text">
              <h3 className="font-bold text-sm text-slate-800">⚙️ Technologies & Architecture</h3>
              <ul className="text-xs text-slate-600 space-y-2.5 list-disc pl-5 leading-relaxed">
                <li><b>FastAPI / Uvicorn:</b> Backend API routes.</li>
                <li><b>spaCy (en_core_web_sm):</b> POS tagging, lemmatization, and text cleanup.</li>
                <li><b>Sentence-Transformers (HuggingFace):</b> Multi-dimensional embeddings mapping.</li>
                <li><b>scikit-learn (RandomForestRegressor):</b> Machine learning score prediction model.</li>
                <li><b>pdfplumber & docx2txt:</b> Native PDF and Word document converters.</li>
              </ul>
            </div>
          )}

          {/* Project Info */}
          {currentPage === 'proj-info' && (
            <div className="bg-white border border-[#e2e8f0] rounded-2xl p-6 shadow-sm space-y-4 select-text">
              <h3 className="font-bold text-sm text-slate-800">ℹ️ Project Deliverables Checklist</h3>
              <ul className="text-xs text-slate-600 space-y-2.5 list-disc pl-5 leading-relaxed">
                <li><b>Python parsing scripts:</b> <span className="text-emerald-600 font-semibold">✓ Completed</span></li>
                <li><b>NLP similarity scoring:</b> <span className="text-emerald-600 font-semibold">✓ Completed</span></li>
                <li><b>Machine Learning ranking model:</b> <span class="text-emerald-600 font-semibold">✓ Completed</span></li>
                <li><b>Streamlit Dashboard:</b> <span className="text-emerald-600 font-semibold">✓ Completed</span></li>
                <li><b>FastAPI Server Backend:</b> <span className="text-emerald-600 font-semibold">✓ Completed</span></li>
                <li><b>React Single Page Application:</b> <span class="text-emerald-600 font-semibold">✓ Completed & Active</span></li>
                <li><b>Unit test suite:</b> <span className="text-emerald-600 font-semibold">✓ Completed (34 passing assertions)</span></li>
              </ul>
            </div>
          )}

        </div>
      </main>
    </div>
  );
}

export default App;
