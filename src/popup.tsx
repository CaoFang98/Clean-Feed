import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { PluginSettings, DEFAULT_SETTINGS } from './core/types';

const styles = {
  container: {
    width: '320px',
    minHeight: '480px',
    background: 'linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%)',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  } as React.CSSProperties,
  header: {
    background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
    padding: '16px',
    color: 'white',
  } as React.CSSProperties,
  headerContent: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  } as React.CSSProperties,
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  } as React.CSSProperties,
  logoIcon: {
    width: '40px',
    height: '40px',
    background: 'rgba(255,255,255,0.2)',
    backdropFilter: 'blur(8px)',
    borderRadius: '12px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '20px',
    fontWeight: 'bold',
  } as React.CSSProperties,
  title: {
    fontSize: '18px',
    fontWeight: 'bold',
    margin: 0,
  } as React.CSSProperties,
  subtitle: {
    fontSize: '12px',
    opacity: 0.8,
    margin: '2px 0 0 0',
  } as React.CSSProperties,
  toggle: {
    position: 'relative' as const,
    width: '56px',
    height: '28px',
    borderRadius: '14px',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
  } as React.CSSProperties,
  toggleKnob: {
    position: 'absolute' as const,
    top: '4px',
    width: '20px',
    height: '20px',
    background: 'white',
    borderRadius: '50%',
    boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
    transition: 'all 0.3s ease',
  } as React.CSSProperties,
  tabs: {
    padding: '16px 16px 0 16px',
  } as React.CSSProperties,
  tabsContainer: {
    display: 'flex',
    gap: '4px',
    background: 'rgba(0,0,0,0.05)',
    padding: '4px',
    borderRadius: '12px',
  } as React.CSSProperties,
  tab: {
    flex: 1,
    padding: '8px 12px',
    border: 'none',
    borderRadius: '8px',
    fontSize: '13px',
    fontWeight: '500' as const,
    cursor: 'pointer',
    background: 'transparent',
    color: '#64748b',
    transition: 'all 0.2s ease',
  } as React.CSSProperties,
  tabActive: {
    background: 'white',
    color: '#4f46e5',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
  } as React.CSSProperties,
  content: {
    padding: '16px',
  } as React.CSSProperties,
  card: {
    background: 'white',
    borderRadius: '12px',
    padding: '16px',
    marginBottom: '12px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
    border: '1px solid #f1f5f9',
    transition: 'all 0.2s ease',
  } as React.CSSProperties,
  cardHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '12px',
  } as React.CSSProperties,
  cardTitle: {
    fontSize: '14px',
    fontWeight: '600' as const,
    color: '#1e293b',
    margin: 0,
  } as React.CSSProperties,
  cardDesc: {
    fontSize: '12px',
    color: '#64748b',
    margin: '4px 0 0 0',
  } as React.CSSProperties,
  sectionTitle: {
    fontSize: '13px',
    fontWeight: '600' as const,
    color: '#1e293b',
    marginBottom: '12px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  } as React.CSSProperties,
  dot: {
    width: '8px',
    height: '8px',
    background: '#4f46e5',
    borderRadius: '50%',
  } as React.CSSProperties,
  sliderContainer: {
    marginTop: '12px',
    paddingTop: '12px',
    borderTop: '1px solid #f1f5f9',
  } as React.CSSProperties,
  sliderLabel: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '8px',
  } as React.CSSProperties,
  sliderText: {
    fontSize: '12px',
    color: '#64748b',
    fontWeight: '500' as const,
  } as React.CSSProperties,
  sliderValue: {
    fontSize: '12px',
    fontWeight: 'bold',
    color: '#4f46e5',
    background: '#eef2ff',
    padding: '4px 8px',
    borderRadius: '6px',
  } as React.CSSProperties,
  slider: {
    width: '100%',
    height: '8px',
    borderRadius: '4px',
    background: '#e2e8f0',
    outline: 'none',
    cursor: 'pointer',
  } as React.CSSProperties,
  footer: {
    padding: '16px',
    borderTop: '1px solid #e2e8f0',
    background: 'linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%)',
  } as React.CSSProperties,
  footerContent: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px',
    fontSize: '12px',
    color: '#64748b',
  } as React.CSSProperties,
  footerLogo: {
    width: '20px',
    height: '20px',
    background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
    borderRadius: '6px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '10px',
    fontWeight: 'bold',
    color: 'white',
  } as React.CSSProperties,
  statusDot: {
    width: '8px',
    height: '8px',
    background: '#22c55e',
    borderRadius: '50%',
    animation: 'pulse 2s infinite',
  } as React.CSSProperties,
};

function Popup() {
  const [settings, setSettings] = useState<PluginSettings>(DEFAULT_SETTINGS);
  const [activeTab, setActiveTab] = useState<'general' | 'platforms' | 'detectors'>('general');

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const data = await chrome.storage.local.get('settings');
      if (data.settings) {
        setSettings({ ...DEFAULT_SETTINGS, ...data.settings });
      }
    } catch (error) {
      console.error('加载设置失败:', error);
    }
  };

  const saveSettings = async (newSettings: PluginSettings) => {
    setSettings(newSettings);
    try {
      await chrome.storage.local.set({ settings: newSettings });
    } catch (error) {
      console.error('保存设置失败:', error);
    }
  };

  const toggleEnabled = () => {
    saveSettings({ ...settings, enabled: !settings.enabled });
  };

  const togglePlatform = (platformId: string) => {
    const newPlatforms = {
      ...settings.platforms,
      [platformId]: {
        ...settings.platforms[platformId],
        enabled: !settings.platforms[platformId].enabled,
      },
    };
    saveSettings({ ...settings, platforms: newPlatforms });
  };

  const toggleDetector = (detectorId: string) => {
    const newDetectors = {
      ...settings.detectors,
      [detectorId]: {
        ...settings.detectors[detectorId],
        enabled: !settings.detectors[detectorId].enabled,
      },
    };
    saveSettings({ ...settings, detectors: newDetectors });
  };

  const updateDetectorThreshold = (detectorId: string, threshold: number) => {
    const newDetectors = {
      ...settings.detectors,
      [detectorId]: {
        ...settings.detectors[detectorId],
        threshold,
      },
    };
    saveSettings({ ...settings, detectors: newDetectors });
  };

  const toggleEffect = (effectId: string) => {
    const newEffects = {
      ...settings.effects,
      [effectId]: {
        ...settings.effects[effectId],
        enabled: !settings.effects[effectId].enabled,
      },
    };
    saveSettings({ ...settings, effects: newEffects });
  };

  const ToggleSwitch = ({ enabled, onChange }: { enabled: boolean; onChange: () => void }) => (
    <button
      onClick={onChange}
      style={{
        ...styles.toggle,
        background: enabled 
          ? 'linear-gradient(135deg, #4ade80 0%, #22c55e 100%)'
          : 'linear-gradient(135deg, #cbd5e1 0%, #94a3b8 100%)',
      }}
    >
      <div style={{
        ...styles.toggleKnob,
        left: enabled ? '32px' : '4px',
      }} />
    </button>
  );

  return (
    <div style={styles.container}>
      {/* 头部 */}
      <div style={styles.header}>
        <div style={styles.headerContent}>
          <div style={styles.logo}>
            <div style={styles.logoIcon}>CF</div>
            <div>
              <h1 style={styles.title}>CleanFeed</h1>
              <p style={styles.subtitle}>内容智能过滤</p>
            </div>
          </div>
          <ToggleSwitch enabled={settings.enabled} onChange={toggleEnabled} />
        </div>
      </div>

      {/* Tab 导航 */}
      <div style={styles.tabs}>
        <div style={styles.tabsContainer}>
          {(['general', 'platforms', 'detectors'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                ...styles.tab,
                ...(activeTab === tab ? styles.tabActive : {}),
              }}
            >
              {tab === 'general' ? '通用' : tab === 'platforms' ? '平台' : '检测器'}
            </button>
          ))}
        </div>
      </div>

      {/* 通用设置 */}
      {activeTab === 'general' && (
        <div style={styles.content}>
          <div style={styles.card}>
            <h3 style={styles.sectionTitle}>
              <span style={styles.dot}></span>
              效果设置
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {Object.entries(settings.effects).map(([id, effect]) => (
                <div key={id} style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '12px',
                  background: 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)',
                  borderRadius: '10px',
                  border: '1px solid #f1f5f9',
                }}>
                  <span style={{ fontSize: '14px', color: '#334155', fontWeight: '500' }}>
                    {effect.name}
                  </span>
                  <ToggleSwitch enabled={effect.enabled} onChange={() => toggleEffect(id)} />
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* 平台设置 */}
      {activeTab === 'platforms' && (
        <div style={styles.content}>
          {Object.entries(settings.platforms).map(([id, platform]) => (
            <div key={id} style={styles.card}>
              <div style={styles.cardHeader}>
                <div>
                  <h4 style={styles.cardTitle}>{platform.name}</h4>
                  <p style={styles.cardDesc}>
                    检测器: {platform.detectors.join(', ')}
                  </p>
                </div>
                <ToggleSwitch enabled={platform.enabled} onChange={() => togglePlatform(id)} />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* 检测器设置 */}
      {activeTab === 'detectors' && (
        <div style={styles.content}>
          {Object.entries(settings.detectors).map(([id, detector]) => (
            <div key={id} style={styles.card}>
              <div style={styles.cardHeader}>
                <div>
                  <h4 style={styles.cardTitle}>{detector.name}</h4>
                  <p style={styles.cardDesc}>{detector.description}</p>
                </div>
                <ToggleSwitch enabled={detector.enabled} onChange={() => toggleDetector(id)} />
              </div>
              {detector.enabled && (
                <div style={styles.sliderContainer}>
                  <div style={styles.sliderLabel}>
                    <span style={styles.sliderText}>检测阈值</span>
                    <span style={styles.sliderValue}>
                      {(detector.threshold * 100).toFixed(0)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={detector.threshold}
                    onChange={(e) => updateDetectorThreshold(id, parseFloat(e.target.value))}
                    style={styles.slider}
                  />
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* 底部信息 */}
      <div style={styles.footer}>
        <div style={styles.footerContent}>
          <div style={styles.footerLogo}>CF</div>
          <span>CleanFeed v1.0.0</span>
          <span style={{ color: '#cbd5e1' }}>•</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#22c55e', fontWeight: '500' }}>
            <span style={styles.statusDot}></span>
            运行中
          </span>
        </div>
      </div>
    </div>
  );
}

const root = createRoot(document.getElementById('root')!);
root.render(<Popup />);
