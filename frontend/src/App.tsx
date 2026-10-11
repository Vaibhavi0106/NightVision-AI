import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { Dashboard } from './pages/Dashboard';
import { ImageEnhancer } from './pages/ImageEnhancer';
import { VideoEnhancer } from './pages/VideoEnhancer';
import { BatchProcessing } from './pages/BatchProcessing';
import { PerformanceLab } from './pages/PerformanceLab';
import { AboutProject } from './pages/AboutProject';
import { HealthResponse } from './types';
import { fetchHealthStatus } from './api/client';

export function App() {
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [healthData, setHealthData] = useState<HealthResponse | undefined>(undefined);
  const [selectedFile, setSelectedFile] = useState<File | undefined>(undefined);

  const loadHealth = async () => {
    try {
      const data = await fetchHealthStatus();
      setHealthData(data);
    } catch (err) {
      console.error('Failed to load health status:', err);
    }
  };

  useEffect(() => {
    loadHealth();
    const interval = setInterval(loadHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleNavigate = (tab: string, file?: File) => {
    if (file) {
      setSelectedFile(file);
    }
    setActiveTab(tab);
  };

  const handleLoadSample = () => {
    setSelectedFile(undefined);
    setActiveTab('image');
  };

  return (
    <div className="min-h-screen bg-[#fff7f9] text-slate-800 flex flex-col font-sans">
      <Navbar
        activeTab={activeTab}
        setActiveTab={(tab) => {
          setSelectedFile(undefined);
          setActiveTab(tab);
        }}
        hardware={healthData?.hardware}
      />

      <main className="flex-1">
        {activeTab === 'dashboard' && (
          <Dashboard
            health={healthData}
            onNavigate={handleNavigate}
            onLoadSample={handleLoadSample}
          />
        )}

        {activeTab === 'image' && (
          <ImageEnhancer
            initialFile={selectedFile}
            hardware={healthData?.hardware}
          />
        )}

        {activeTab === 'video' && (
          <VideoEnhancer
            initialFile={selectedFile}
            hardware={healthData?.hardware}
          />
        )}

        {activeTab === 'batch' && (
          <BatchProcessing hardware={healthData?.hardware} />
        )}

        {activeTab === 'performance' && (
          <PerformanceLab hardware={healthData?.hardware} />
        )}

        {activeTab === 'about' && (
          <AboutProject hardware={healthData?.hardware} />
        )}
      </main>

      <footer className="border-t border-pink-200 bg-white py-6 px-6 text-center text-xs text-slate-500 font-mono shadow-inner">
        NightVision AI &copy; 2026 — GPU Accelerated Low-Light Photo & Video Enhancement Platform | Powered by NVIDIA CUDA & FastAPI
      </footer>
    </div>
  );
}

export default App;
