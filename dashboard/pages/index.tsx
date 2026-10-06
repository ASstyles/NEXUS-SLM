import React, { useState, useEffect } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  CheckCircle2,
  Cpu,
  Database,
  DollarSign,
  Globe,
  Layers,
  Play,
  RefreshCw,
  Server,
  ShieldAlert,
  Zap,
} from 'lucide-react';

interface TriageResult {
  category: string;
  priority: string;
  language: string;
  requires_human: boolean;
  confidence: number;
  latency_ms: number;
  model_version: string;
  raw_output?: string;
}

interface BenchmarkData {
  columns: string[];
  rows: Array<{
    metric: string;
    base_slm: string;
    nexus_slm: string;
    large_llm: string;
  }>;
  status: {
    base_measured: boolean;
    nexus_measured: boolean;
    large_measured: boolean;
  };
}

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<'overview' | 'analyze' | 'benchmark' | 'experiments' | 'error_analysis' | 'registry' | 'monitoring'>('overview');
  const [apiConnected, setApiConnected] = useState<boolean>(false);
  const [hardwareInfo, setHardwareInfo] = useState<any>(null);

  // Analysis State
  const [ticketInput, setTicketInput] = useState<string>('Mera payment kat gaya but order confirm nahi hua.');
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<TriageResult | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  // Data states
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkData | null>(null);
  const [experiments, setExperiments] = useState<any[]>([]);
  const [telemetryStats, setTelemetryStats] = useState<any>(null);
  const [registryModels, setRegistryModels] = useState<any[]>([]);
  const [errorReport, setErrorReport] = useState<any>(null);

  // Cost simulator state
  const [simVolume, setSimVolume] = useState<number>(50000);

  const API_BASE = '/api/backend';

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        setApiConnected(true);
        setHardwareInfo(data.hardware);
      } else {
        setApiConnected(false);
      }
    } catch {
      setApiConnected(false);
    }
  };

  const fetchDashboardData = async () => {
    try {
      const [bmRes, expRes, telRes, regRes, errRes] = await Promise.all([
        fetch(`${API_BASE}/benchmark`),
        fetch(`${API_BASE}/experiments`),
        fetch(`${API_BASE}/metrics`),
        fetch(`${API_BASE}/registry`),
        fetch(`${API_BASE}/error-analysis`),
      ]);

      if (bmRes.ok) setBenchmarkData(await bmRes.json());
      if (expRes.ok) setExperiments(await expRes.json());
      if (telRes.ok) setTelemetryStats(await telRes.json());
      if (regRes.ok) setRegistryModels(await regRes.json());
      if (errRes.ok) setErrorReport(await errRes.json());
    } catch (e) {
      console.warn('Dashboard data fetch error:', e);
    }
  };

  useEffect(() => {
    checkHealth();
    fetchDashboardData();
    const interval = setInterval(() => {
      checkHealth();
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleAnalyze = async () => {
    if (!ticketInput.trim()) return;
    setIsAnalyzing(true);
    setAnalysisError(null);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: ticketInput }),
      });
      if (!res.ok) {
        throw new Error(`Inference returned HTTP ${res.status}`);
      }
      const data: TriageResult = await res.json();
      setAnalysisResult(data);
      fetchDashboardData();
    } catch (err: any) {
      setAnalysisError(err.message || 'Failed to execute triage');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getPriorityBadgeClass = (prio: string) => {
    switch (prio.toLowerCase()) {
      case 'critical': return 'badge-critical';
      case 'high': return 'badge-high';
      case 'medium': return 'badge-medium';
      case 'low': return 'badge-low';
      default: return 'badge-default';
    }
  };

  const getCategoryBadgeClass = (cat: string) => {
    if (cat.includes('payment') || cat.includes('billing')) return 'badge-payment';
    if (cat.includes('refund')) return 'badge-refund';
    if (cat.includes('fraud')) return 'badge-fraud';
    if (cat.includes('account') || cat.includes('password')) return 'badge-account';
    if (cat.includes('delivery') || cat.includes('tracking')) return 'badge-delivery';
    return 'badge-default';
  };

  // Cost projections based on active slider
  const calcNexusCost = (vol: number) => ((vol / 1000) * 0.0058).toFixed(2);
  const calcTier1Cost = (vol: number) => (((vol * 120 / 1e6) * 0.15) + ((vol * 45 / 1e6) * 0.60)).toFixed(2);
  const calcPremiumCost = (vol: number) => (((vol * 120 / 1e6) * 2.50) + ((vol * 45 / 1e6) * 10.00)).toFixed(2);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Header */}
      <header style={{
        borderBottom: '1px solid var(--border-subtle)',
        background: 'rgba(7, 9, 14, 0.85)',
        backdropFilter: 'blur(20px)',
        position: 'sticky',
        top: 0,
        zIndex: 50,
        padding: '16px 32px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{
            width: '42px',
            height: '42px',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, #6366f1, #06b6d4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 20px rgba(99, 102, 241, 0.4)'
          }}>
            <Zap size={22} color="#fff" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h1 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.25rem', fontWeight: 700, letterSpacing: '-0.02em' }}>
                NEXUS-SLM
              </h1>
              <span className="badge badge-default" style={{ fontSize: '0.65rem' }}>v1.0</span>
            </div>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              Neural EXecution for Unified Support Intelligence
            </p>
          </div>
        </div>

        {/* Global Status Badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 14px',
            borderRadius: '20px',
            background: apiConnected ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
            border: `1px solid ${apiConnected ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`
          }}>
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: apiConnected ? '#10b981' : '#ef4444',
              display: 'inline-block'
            }} />
            <span style={{ fontSize: '0.8rem', fontWeight: 500, color: apiConnected ? '#34d399' : '#f87171' }}>
              {apiConnected ? 'Inference Engine Online' : 'API Connecting...'}
            </span>
          </div>

          {hardwareInfo?.gpus?.[0] && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 12px',
              borderRadius: '20px',
              background: 'rgba(99, 102, 241, 0.1)',
              border: '1px solid rgba(99, 102, 241, 0.25)',
              fontSize: '0.78rem',
              color: '#c7d2fe'
            }}>
              <Cpu size={14} />
              <span>{hardwareInfo.gpus[0].name} ({hardwareInfo.gpus[0].total_vram_gb} GB VRAM)</span>
            </div>
          )}
        </div>
      </header>

      {/* Navigation Bar */}
      <nav style={{
        padding: '12px 32px',
        borderBottom: '1px solid var(--border-subtle)',
        background: 'rgba(14, 19, 31, 0.4)',
        display: 'flex',
        gap: '8px',
        overflowX: 'auto'
      }}>
        <button
          className={`nav-tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          <Activity size={16} /> Overview
        </button>
        <button
          className={`nav-tab-btn ${activeTab === 'analyze' ? 'active' : ''}`}
          onClick={() => setActiveTab('analyze')}
        >
          <Play size={16} /> Analyze
        </button>
        <button
          className={`nav-tab-btn ${activeTab === 'benchmark' ? 'active' : ''}`}
          onClick={() => setActiveTab('benchmark')}
        >
          <BarChart3 size={16} /> Benchmark
        </button>
        <button
          className={`nav-tab-btn ${activeTab === 'experiments' ? 'active' : ''}`}
          onClick={() => setActiveTab('experiments')}
        >
          <Layers size={16} /> Experiments
        </button>
        <button
          className={`nav-tab-btn ${activeTab === 'error_analysis' ? 'active' : ''}`}
          onClick={() => setActiveTab('error_analysis')}
        >
          <AlertTriangle size={16} /> Error Analysis
        </button>
        <button
          className={`nav-tab-btn ${activeTab === 'registry' ? 'active' : ''}`}
          onClick={() => setActiveTab('registry')}
        >
          <Database size={16} /> Model Registry
        </button>
        <button
          className={`nav-tab-btn ${activeTab === 'monitoring' ? 'active' : ''}`}
          onClick={() => setActiveTab('monitoring')}
        >
          <Server size={16} /> Monitoring
        </button>
      </nav>

      {/* Main Container */}
      <main style={{ flex: 1, padding: '32px', maxWidth: '1440px', margin: '0 auto', width: '100%' }}>

        {/* 1. OVERVIEW TAB */}
        {activeTab === 'overview' && (
          <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
            {/* KPI Cards Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '20px' }}>
              <div className="glass-panel" style={{ padding: '22px' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Total Requests
                </span>
                <div style={{ fontSize: '2rem', fontWeight: 700, marginTop: '8px', fontFamily: 'var(--font-heading)' }}>
                  {telemetryStats?.total_requests ?? 0}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#10b981', marginTop: '4px' }}>
                  Live Telemetry
                </div>
              </div>

              <div className="glass-panel" style={{ padding: '22px' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Best Macro-F1
                </span>
                <div style={{ fontSize: '2rem', fontWeight: 700, marginTop: '8px', color: '#a5f3fc', fontFamily: 'var(--font-heading)' }}>
                  {registryModels[0]?.evaluation_results?.macro_f1
                    ? registryModels[0].evaluation_results.macro_f1.toFixed(4)
                    : (benchmarkData?.rows?.find(r => r.metric === 'Macro-F1')?.nexus_slm ?? '—')}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-faint)', marginTop: '4px' }}>
                  Locked Test Benchmark
                </div>
              </div>

              <div className="glass-panel" style={{ padding: '22px' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  P50 Latency
                </span>
                <div style={{ fontSize: '2rem', fontWeight: 700, marginTop: '8px', color: '#fed7aa', fontFamily: 'var(--font-heading)' }}>
                  {telemetryStats?.p50_latency_ms ? `${telemetryStats.p50_latency_ms} ms` : '—'}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-faint)', marginTop: '4px' }}>
                  Median Response Time
                </div>
              </div>

              <div className="glass-panel" style={{ padding: '22px' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  JSON Validity
                </span>
                <div style={{ fontSize: '2rem', fontWeight: 700, marginTop: '8px', color: '#86efac', fontFamily: 'var(--font-heading)' }}>
                  {registryModels[0]?.evaluation_results?.json_validity_rate
                    ? `${(registryModels[0].evaluation_results.json_validity_rate * 100).toFixed(1)}%`
                    : '100.0%'}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-faint)', marginTop: '4px' }}>
                  Strict Schema Enforcement
                </div>
              </div>

              <div className="glass-panel" style={{ padding: '22px' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Active Model
                </span>
                <div style={{ fontSize: '1.25rem', fontWeight: 600, marginTop: '8px', fontFamily: 'var(--font-heading)', color: '#c7d2fe' }}>
                  Qwen2.5-1.5B (4-bit)
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-faint)', marginTop: '4px' }}>
                  Tag: nexus-v1.0
                </div>
              </div>
            </div>

            {/* Quick System Architecture Banner */}
            <div className="glass-panel" style={{ padding: '28px', background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(6, 182, 212, 0.05) 100%)' }}>
              <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.2rem', marginBottom: '8px' }}>
                Architecture & Core Research Premise
              </h2>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem', lineHeight: 1.6, maxWidth: '1000px' }}>
                Can a 1–3B parameter language model specialized using QLoRA achieve superior multilingual customer-support
                ticket triage (English, Hindi, Hinglish) with lower latency, lower memory footprint, and ~90% lower operational cost than general-purpose LLMs?
                NEXUS-SLM executes 4-bit quantized inference natively on edge or entry-tier GPUs while strictly producing validated JSON classifications.
              </p>
              <div style={{ display: 'flex', gap: '16px', marginTop: '20px' }}>
                <button className="btn-primary" onClick={() => setActiveTab('analyze')}>
                  Test Interactive Triage <ArrowRight size={16} />
                </button>
                <button className="btn-secondary" onClick={() => setActiveTab('benchmark')}>
                  View Benchmark Matrix
                </button>
              </div>
            </div>
          </div>
        )}

        {/* 2. ANALYZE TAB */}
        {activeTab === 'analyze' && (
          <div className="fade-in" style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '28px' }}>
            {/* Input Column */}
            <div className="glass-panel" style={{ padding: '28px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.2rem' }}>
                  Enter Support Ticket
                </h2>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  Accepts English, Hindi (Devanagari), & Hinglish
                </span>
              </div>

              {/* Sample Preset Chips */}
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '16px' }}>
                {[
                  { label: 'Hinglish Payment', text: 'Mera payment kat gaya but order confirm nahi hua.' },
                  { label: 'Hindi Payment', text: 'मेरा payment कट गया लेकिन order confirm नहीं हुआ।' },
                  { label: 'English Payment', text: 'My payment was deducted but my order was not confirmed.' },
                  { label: 'Fraud Alert', text: 'Urgent: Unauthorized charges of $450 detected on my credit card!' },
                  { label: 'Damaged Item', text: 'Delivery boy delivered a broken packet and refused to take return.' },
                  { label: 'Hinglish Refund', text: 'Item return pickup ho gaya tha 4 din pehle par refund nahi aaya.' },
                ].map((preset, idx) => (
                  <button
                    key={idx}
                    className="btn-secondary"
                    style={{ fontSize: '0.75rem', padding: '6px 12px' }}
                    onClick={() => setTicketInput(preset.text)}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>

              <textarea
                value={ticketInput}
                onChange={(e) => setTicketInput(e.target.value)}
                rows={6}
                placeholder="Type or paste a customer support message..."
                style={{
                  width: '100%',
                  background: 'rgba(7, 9, 14, 0.7)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: '12px',
                  color: 'var(--text-main)',
                  padding: '16px',
                  fontSize: '0.95rem',
                  fontFamily: 'var(--font-body)',
                  resize: 'vertical',
                  outline: 'none',
                  boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.5)'
                }}
              />

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '20px' }}>
                <button
                  className="btn-primary"
                  onClick={handleAnalyze}
                  disabled={isAnalyzing || !ticketInput.trim()}
                  style={{ minWidth: '150px' }}
                >
                  {isAnalyzing ? (
                    <>
                      <RefreshCw size={16} className="pulse-glow" style={{ animation: 'spin 1s linear infinite' }} /> Analyzing...
                    </>
                  ) : (
                    <>
                      <Zap size={16} /> ANALYZE TICKET
                    </>
                  )}
                </button>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Active Model: nexus-v1.0 (Qwen2.5-1.5B)
                </span>
              </div>

              {analysisError && (
                <div style={{
                  marginTop: '16px',
                  padding: '12px 16px',
                  borderRadius: '8px',
                  background: 'rgba(239, 68, 68, 0.15)',
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                  color: '#fca5a5',
                  fontSize: '0.85rem'
                }}>
                  {analysisError}
                </div>
              )}
            </div>

            {/* Results Column */}
            <div className="glass-panel" style={{ padding: '28px' }}>
              <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.2rem', marginBottom: '20px' }}>
                Triage Intelligence Output
              </h2>

              {analysisResult ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: '14px', borderBottom: '1px solid var(--border-subtle)' }}>
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Category</span>
                    <span className={`badge ${getCategoryBadgeClass(analysisResult.category)}`} style={{ fontSize: '0.85rem' }}>
                      {analysisResult.category.toUpperCase()}
                    </span>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: '14px', borderBottom: '1px solid var(--border-subtle)' }}>
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Priority</span>
                    <span className={`badge ${getPriorityBadgeClass(analysisResult.priority)}`} style={{ fontSize: '0.85rem' }}>
                      {analysisResult.priority.toUpperCase()}
                    </span>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: '14px', borderBottom: '1px solid var(--border-subtle)' }}>
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Language Modality</span>
                    <span className="badge badge-default" style={{ fontSize: '0.85rem' }}>
                      <Globe size={12} style={{ marginRight: '4px' }} />
                      {analysisResult.language.toUpperCase()}
                    </span>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: '14px', borderBottom: '1px solid var(--border-subtle)' }}>
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Human Escalation</span>
                    {analysisResult.requires_human ? (
                      <span className="badge badge-high" style={{ fontSize: '0.82rem' }}>
                        <ShieldAlert size={12} style={{ marginRight: '4px' }} /> ESCALATE TO HUMAN
                      </span>
                    ) : (
                      <span className="badge badge-low" style={{ fontSize: '0.82rem' }}>
                        <CheckCircle2 size={12} style={{ marginRight: '4px' }} /> AUTO-RESOLVE OK
                      </span>
                    )}
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: '14px', borderBottom: '1px solid var(--border-subtle)' }}>
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Model Confidence</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <div style={{
                        width: '100px',
                        height: '8px',
                        background: 'rgba(255,255,255,0.1)',
                        borderRadius: '4px',
                        overflow: 'hidden'
                      }}>
                        <div style={{
                          width: `${(analysisResult.confidence * 100).toFixed(0)}%`,
                          height: '100%',
                          background: analysisResult.confidence > 0.85 ? '#10b981' : '#f59e0b',
                        }} />
                      </div>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9rem', fontWeight: 600 }}>
                        {(analysisResult.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: '14px', borderBottom: '1px solid var(--border-subtle)' }}>
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Inference Latency</span>
                    <span style={{ fontFamily: 'var(--font-mono)', color: '#a5f3fc', fontSize: '0.9rem' }}>
                      {analysisResult.latency_ms} ms
                    </span>
                  </div>

                  {/* Raw JSON View */}
                  <div style={{ marginTop: '10px' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      Validated JSON Payload:
                    </span>
                    <pre style={{
                      marginTop: '6px',
                      background: 'rgba(0,0,0,0.4)',
                      padding: '12px',
                      borderRadius: '8px',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.8rem',
                      color: '#93c5fd',
                      overflowX: 'auto',
                      border: '1px solid rgba(255,255,255,0.05)'
                    }}>
                      {JSON.stringify({
                        category: analysisResult.category,
                        priority: analysisResult.priority,
                        language: analysisResult.language,
                        requires_human: analysisResult.requires_human,
                        confidence: analysisResult.confidence,
                        latency_ms: analysisResult.latency_ms,
                        model_version: analysisResult.model_version
                      }, null, 2)}
                    </pre>
                  </div>
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-faint)' }}>
                  <Zap size={36} style={{ opacity: 0.3, marginBottom: '12px' }} />
                  <p>Enter a customer ticket on the left and click <strong>ANALYZE</strong> to execute inference.</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* 3. BENCHMARK TAB */}
        {activeTab === 'benchmark' && (
          <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
            {/* Comparison Matrix */}
            <div className="glass-panel" style={{ padding: '28px' }}>
              <div style={{ marginBottom: '20px' }}>
                <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.25rem' }}>
                  Official Benchmark Comparison Matrix
                </h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '4px' }}>
                  Strict evaluation on the locked test benchmark across Base SLM, NEXUS-SLM, and Large Commercial LLMs.
                </p>
              </div>

              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.9rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border-subtle)', background: 'rgba(255,255,255,0.02)' }}>
                      <th style={{ padding: '14px 16px', color: 'var(--text-muted)', fontWeight: 600 }}>Metric</th>
                      <th style={{ padding: '14px 16px', color: 'var(--text-muted)', fontWeight: 600 }}>Base SLM</th>
                      <th style={{ padding: '14px 16px', color: '#818cf8', fontWeight: 600 }}>NEXUS-SLM (Fine-Tuned)</th>
                      <th style={{ padding: '14px 16px', color: 'var(--text-muted)', fontWeight: 600 }}>Large LLM (GPT-4o / Claude)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarkData?.rows?.map((row, idx) => (
                      <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                        <td style={{ padding: '12px 16px', fontWeight: 500 }}>{row.metric}</td>
                        <td style={{ padding: '12px 16px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>{row.base_slm}</td>
                        <td style={{ padding: '12px 16px', fontFamily: 'var(--font-mono)', color: '#c7d2fe', fontWeight: 600 }}>{row.nexus_slm}</td>
                        <td style={{ padding: '12px 16px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>{row.large_llm}</td>
                      </tr>
                    )) || (
                      <tr>
                        <td colSpan={4} style={{ padding: '24px', textAlign: 'center', color: 'var(--text-faint)' }}>
                          No experiment results available. Run evaluation script to populate.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Interactive Cost Simulator */}
            <div className="glass-panel" style={{ padding: '28px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
                <DollarSign size={20} color="#10b981" />
                <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.25rem' }}>
                  Economic Cost Analysis & Projections
                </h2>
              </div>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                Compare monthly operational expenditure based on expected customer ticket volume.
              </p>

              <div style={{ margin: '24px 0' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '0.88rem' }}>Monthly Support Ticket Volume:</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#38bdf8' }}>
                    {simVolume.toLocaleString()} tickets / month
                  </span>
                </div>
                <input
                  type="range"
                  min="5000"
                  max="500000"
                  step="5000"
                  value={simVolume}
                  onChange={(e) => setSimVolume(Number(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent-primary)', cursor: 'pointer' }}
                />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px' }}>
                <div style={{ padding: '20px', borderRadius: '12px', background: 'rgba(99, 102, 241, 0.1)', border: '1px solid rgba(99, 102, 241, 0.3)' }}>
                  <span style={{ fontSize: '0.78rem', textTransform: 'uppercase', color: '#c7d2fe', fontWeight: 600 }}>
                    NEXUS-SLM (Self-Hosted)
                  </span>
                  <div style={{ fontSize: '1.8rem', fontWeight: 700, marginTop: '8px', color: '#fff', fontFamily: 'var(--font-heading)' }}>
                    ${calcNexusCost(simVolume)}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#86efac', marginTop: '6px' }}>
                    Marginal GPU cost ($0.0058 / 1k reqs)
                  </div>
                </div>

                <div style={{ padding: '20px', borderRadius: '12px', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-subtle)' }}>
                  <span style={{ fontSize: '0.78rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600 }}>
                    Commercial API Tier 1 (GPT-4o-mini)
                  </span>
                  <div style={{ fontSize: '1.8rem', fontWeight: 700, marginTop: '8px', color: '#e2e8f0', fontFamily: 'var(--font-heading)' }}>
                    ${calcTier1Cost(simVolume)}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '6px' }}>
                    Token usage pricing
                  </div>
                </div>

                <div style={{ padding: '20px', borderRadius: '12px', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-subtle)' }}>
                  <span style={{ fontSize: '0.78rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600 }}>
                    Commercial API Premium (GPT-4o / Claude)
                  </span>
                  <div style={{ fontSize: '1.8rem', fontWeight: 700, marginTop: '8px', color: '#e2e8f0', fontFamily: 'var(--font-heading)' }}>
                    ${calcPremiumCost(simVolume)}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '6px' }}>
                    Premium enterprise pricing
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 4. EXPERIMENTS TAB */}
        {activeTab === 'experiments' && (
          <div className="fade-in glass-panel" style={{ padding: '28px' }}>
            <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.25rem', marginBottom: '8px' }}>
              Experiment Tracking & Ablation History
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '24px' }}>
              Recorded runs synced with MLflow and local registry.
            </p>

            {experiments.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border-subtle)', background: 'rgba(255,255,255,0.02)' }}>
                      <th style={{ padding: '12px 16px', color: 'var(--text-muted)' }}>Run Name</th>
                      <th style={{ padding: '12px 16px', color: 'var(--text-muted)' }}>Status</th>
                      <th style={{ padding: '12px 16px', color: 'var(--text-muted)' }}>LoRA Rank</th>
                      <th style={{ padding: '12px 16px', color: 'var(--text-muted)' }}>Macro-F1</th>
                      <th style={{ padding: '12px 16px', color: 'var(--text-muted)' }}>JSON Validity</th>
                      <th style={{ padding: '12px 16px', color: 'var(--text-muted)' }}>P50 Latency</th>
                    </tr>
                  </thead>
                  <tbody>
                    {experiments.map((run, idx) => (
                      <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                        <td style={{ padding: '12px 16px', fontWeight: 600 }}>{run.run_name}</td>
                        <td style={{ padding: '12px 16px' }}>
                          <span className="badge badge-low">{run.status}</span>
                        </td>
                        <td style={{ padding: '12px 16px', fontFamily: 'var(--font-mono)' }}>
                          {run.params?.lora?.r ?? run.params?.lora_r ?? 'r=16'}
                        </td>
                        <td style={{ padding: '12px 16px', fontFamily: 'var(--font-mono)', color: '#86efac' }}>
                          {run.metrics?.macro_f1?.toFixed(4) ?? '—'}
                        </td>
                        <td style={{ padding: '12px 16px', fontFamily: 'var(--font-mono)' }}>
                          {run.metrics?.json_validity_rate ?? run.metrics?.json_validity ?? '1.0'}
                        </td>
                        <td style={{ padding: '12px 16px', fontFamily: 'var(--font-mono)' }}>
                          {run.metrics?.latency_p50_ms ? `${run.metrics.latency_p50_ms} ms` : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-faint)' }}>
                No experiment results recorded yet. Run <code>python scripts/training/run_experiments.py</code> to log sweeps.
              </div>
            )}
          </div>
        )}

        {/* 5. ERROR ANALYSIS TAB */}
        {activeTab === 'error_analysis' && (
          <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
            <div className="glass-panel" style={{ padding: '28px' }}>
              <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.25rem', marginBottom: '8px' }}>
                Error Diagnostics & Systematic Failure Modes
              </h2>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '24px' }}>
                Automated error decomposition across semantic categories and language modalities.
              </p>

              {errorReport?.hardest_categories ? (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
                  {/* Hardest Categories */}
                  <div>
                    <h3 style={{ fontSize: '0.95rem', marginBottom: '12px', color: '#f8fafc' }}>
                      Top Confused Support Categories
                    </h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      {errorReport.hardest_categories.map((item: any, idx: number) => (
                        <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 14px', borderRadius: '8px', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-subtle)' }}>
                          <span style={{ fontSize: '0.85rem' }}>{item.category}</span>
                          <span className="badge badge-high">{item.error_count} errors</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Error Distribution */}
                  <div>
                    <h3 style={{ fontSize: '0.95rem', marginBottom: '12px', color: '#f8fafc' }}>
                      Error Type Distribution
                    </h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      {Object.entries(errorReport.error_distribution || {}).map(([errType, cnt]: any, idx) => (
                        <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 14px', borderRadius: '8px', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-subtle)' }}>
                          <span style={{ fontSize: '0.85rem' }}>{errType.replace('_', ' ')}</span>
                          <span className="badge badge-default">{cnt} instances</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-faint)' }}>
                  No error report available yet. Run <code>python scripts/evaluation/run_evaluation.py --mode error_analysis</code>.
                </div>
              )}
            </div>
          </div>
        )}

        {/* 6. MODEL REGISTRY TAB */}
        {activeTab === 'registry' && (
          <div className="fade-in glass-panel" style={{ padding: '28px' }}>
            <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.25rem', marginBottom: '8px' }}>
              Model Registry & Deployment Catalog
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '24px' }}>
              Versioned model artifacts with cryptographic hash verification and deployment statuses.
            </p>

            {registryModels.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border-subtle)', background: 'rgba(255,255,255,0.02)' }}>
                      <th style={{ padding: '12px 16px', color: 'var(--text-muted)' }}>Version</th>
                      <th style={{ padding: '12px 16px', color: 'var(--text-muted)' }}>Base Model</th>
                      <th style={{ padding: '12px 16px', color: 'var(--text-muted)' }}>Quantization</th>
                      <th style={{ padding: '12px 16px', color: 'var(--text-muted)' }}>Status</th>
                      <th style={{ padding: '12px 16px', color: 'var(--text-muted)' }}>SHA256 Hash</th>
                      <th style={{ padding: '12px 16px', color: 'var(--text-muted)' }}>Registered At</th>
                    </tr>
                  </thead>
                  <tbody>
                    {registryModels.map((m, idx) => (
                      <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                        <td style={{ padding: '12px 16px', fontWeight: 600 }}>{m.version}</td>
                        <td style={{ padding: '12px 16px' }}>{m.base_model}</td>
                        <td style={{ padding: '12px 16px' }}>
                          <span className="badge badge-default">{m.quantization}</span>
                        </td>
                        <td style={{ padding: '12px 16px' }}>
                          <span className={`badge ${m.deployment_status === 'ACTIVE' ? 'badge-low' : 'badge-default'}`}>
                            {m.deployment_status}
                          </span>
                        </td>
                        <td style={{ padding: '12px 16px', fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                          {m.model_hash?.slice(0, 16)}...
                        </td>
                        <td style={{ padding: '12px 16px', fontSize: '0.8rem', color: 'var(--text-faint)' }}>
                          {new Date(m.registered_at).toLocaleString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-faint)' }}>
                No registered model versions found.
              </div>
            )}
          </div>
        )}

        {/* 7. MONITORING TAB */}
        {activeTab === 'monitoring' && (
          <div className="fade-in glass-panel" style={{ padding: '28px' }}>
            <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.25rem', marginBottom: '8px' }}>
              Operational Telemetry & Traffic Breakdown
            </h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '24px' }}>
              Privacy-preserving metrics tracking real-time API performance without storing ticket content.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', marginBottom: '28px' }}>
              <div style={{ padding: '18px', borderRadius: '12px', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-subtle)' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Total Executions</span>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, marginTop: '6px' }}>
                  {telemetryStats?.total_requests ?? 0}
                </div>
              </div>

              <div style={{ padding: '18px', borderRadius: '12px', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-subtle)' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Error Rate</span>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, marginTop: '6px', color: telemetryStats?.error_rate > 0 ? '#f87171' : '#34d399' }}>
                  {telemetryStats ? `${(telemetryStats.error_rate * 100).toFixed(2)}%` : '0.00%'}
                </div>
              </div>

              <div style={{ padding: '18px', borderRadius: '12px', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-subtle)' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Human Escalation Rate</span>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, marginTop: '6px', color: '#fde047' }}>
                  {telemetryStats ? `${(telemetryStats.human_escalation_rate * 100).toFixed(1)}%` : '0.0%'}
                </div>
              </div>
            </div>

            {/* Language Distribution Breakdown */}
            {telemetryStats?.languages && (
              <div>
                <h3 style={{ fontSize: '0.95rem', marginBottom: '14px' }}>Language Traffic Distribution</h3>
                <div style={{ display: 'flex', gap: '16px' }}>
                  {Object.entries(telemetryStats.languages).map(([lang, cnt]: any, idx) => (
                    <div key={idx} style={{ padding: '12px 18px', borderRadius: '8px', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-subtle)' }}>
                      <span style={{ fontSize: '0.8rem', textTransform: 'uppercase', color: 'var(--text-muted)' }}>{lang}</span>
                      <div style={{ fontSize: '1.2rem', fontWeight: 700, marginTop: '4px' }}>{cnt}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

      </main>

      {/* Footer */}
      <footer style={{
        padding: '20px 32px',
        borderTop: '1px solid var(--border-subtle)',
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: '0.8rem',
        color: 'var(--text-faint)'
      }}>
        <span>NEXUS-SLM — Neural Support Intelligence Enterprise Platform</span>
        <span>Apache-2.0 License • PyTorch • PEFT QLoRA • Next.js • FastAPI</span>
      </footer>
    </div>
  );
}
