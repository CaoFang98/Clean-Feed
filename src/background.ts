import { PluginSettings, DEFAULT_SETTINGS } from './core/types';

// 初始化存储
chrome.runtime.onInstalled.addListener(async () => {
  console.log('[CleanFeed] 🚀 CleanFeed 已安装');
  
  // 初始化设置
  const existingSettings = await chrome.storage.local.get('settings');
  if (!existingSettings.settings) {
    await chrome.storage.local.set({ settings: DEFAULT_SETTINGS });
  }
});

// 监听标签页更新，在支持的平台页面上显示图标
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    const supportedPlatforms = ['xiaohongshu.com', 'zhihu.com'];
    const isSupported = supportedPlatforms.some(p => tab.url!.includes(p));
    
    if (isSupported) {
      chrome.action.enable(tabId);
    }
  }
});

console.log('[CleanFeed] ✅ CleanFeed 后台服务已启动');
