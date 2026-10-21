import React from 'react';
import { Eye, LayoutDashboard, Image as ImageIcon, Film, Layers, Cpu, Info, Sparkles } from 'lucide-react';
import { HardwareStatus } from '../types';
import { BackendStatusBadge } from './BackendStatusBadge';

interface Props {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  hardware?: HardwareStatus;
}

export const Navbar: React.FC<Props> = ({ activeTab, setActiveTab, hardware }) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'image', label: 'Image Enhancer', icon: ImageIcon },
    { id: 'video', label: 'Video Enhancer', icon: Film },
    { id: 'batch', label: 'Batch Processing', icon: Layers },
    { id: 'performance', label: 'Performance Lab', icon: Cpu },
    { id: 'about', label: 'About Project', icon: Info },
  ];

  return (
    <header className="sticky top-0 z-50 bg-gradient-to-r from-pink-600 via-rose-500 to-pink-600 text-white shadow-xl border-b-4 border-pink-700">
      <div className="w-full px-4 sm:px-8 lg:px-12 py-3.5 flex items-center justify-between">
        {/* Brand Logo */}
        <div 
          onClick={() => setActiveTab('dashboard')}
          className="flex items-center gap-3.5 cursor-pointer group select-none"
        >
          <div className="w-11 h-11 rounded-2xl bg-white p-0.5 shadow-lg group-hover:scale-105 transition-all">
            <div className="w-full h-full bg-pink-50 rounded-xl flex items-center justify-center">
              <Eye className="w-6 h-6 text-pink-600 group-hover:rotate-12 transition-transform" />
            </div>
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <span className="font-black text-2xl tracking-tight text-white drop-shadow-sm">
                NightVision<span className="text-pink-200">AI</span>
              </span>
              <span className="inline-flex items-center gap-1 text-[10px] font-mono font-black uppercase tracking-wider text-pink-700 bg-white px-2.5 py-0.5 rounded-full shadow">
                <Sparkles className="w-2.5 h-2.5 text-pink-600" /> HOT PINK GPU
              </span>
            </div>
            <p className="text-[11px] text-pink-100 font-bold tracking-wide">GPU Low-Light Enhancement Studio</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1.5 bg-pink-800/40 p-1.5 rounded-2xl border border-pink-400/50 backdrop-blur-md">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-black transition-all duration-200 ${
                  isActive
                    ? 'bg-white text-pink-700 shadow-lg scale-105'
                    : 'text-pink-100 hover:text-white hover:bg-pink-700/50'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-pink-600' : 'text-pink-200'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Hardware Status Badge */}
        <div className="hidden lg:block">
          <BackendStatusBadge hardware={hardware} />
        </div>
      </div>
    </header>
  );
};
