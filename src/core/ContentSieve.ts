import { PlatformAdapter } from '../platforms/PlatformAdapter';
import { XiaohongshuAdapter } from '../platforms/XiaohongshuAdapter';
import { ZhihuAdapter } from '../platforms/ZhihuAdapter';
import { Detector } from '../detectors/Detector';
import { AIDetector } from '../detectors/AIDetector';
import { LowQualityDetector } from '../detectors/LowQualityDetector';
import { BackendDetector } from '../detectors/BackendDetector';
import { Effect } from '../effects/Effect';
import { BlurEffect } from '../effects/BlurEffect';
import { LabelEffect } from '../effects/LabelEffect';
import { 
  PluginSettings, 
  DEFAULT_SETTINGS, 
  DetectionResult, 
  Post 
} from './types';

export class ContentSieve {
  private static instance: ContentSieve;
  
  private settings: PluginSettings = DEFAULT_SETTINGS;
  private platforms: Map<string, PlatformAdapter> = new Map();
  private detectors: Map<string, Detector> = new Map();
  private effects: Map<string, Effect> = new Map();
  private currentPlatform: PlatformAdapter | null = null;
  
  private mutationObserver: MutationObserver | null = null;
  private isProcessing = false;
  private debounceTimer: ReturnType<typeof setTimeout> | null = null;

  public static getInstance(): ContentSieve {
    if (!ContentSieve.instance) {
      ContentSieve.instance = new ContentSieve();
    }
    return ContentSieve.instance;
  }

  constructor() {
    this.registerPlatforms();
    this.registerDetectors();
    this.registerEffects();
  }

  private registerPlatforms() {
    this.platforms.set('xiaohongshu', XiaohongshuAdapter.getInstance());
    this.platforms.set('zhihu', ZhihuAdapter.getInstance());
  }

  private registerDetectors() {
    this.detectors.set('ai-content', BackendDetector.getInstance('ai_generated'));
    this.detectors.set('low-quality', BackendDetector.getInstance('low_quality'));
  }

  private registerEffects() {
    this.effects.set('blur', BlurEffect.getInstance());
    this.effects.set('label', LabelEffect.getInstance());
  }

  async init() {
    console.log('[CleanFeed] 🚀 正在初始化...');
    
    // 加载设置
    await this.loadSettings();
    
    // 识别当前平台
    this.detectCurrentPlatform();
    
    if (!this.currentPlatform) {
      console.log('[CleanFeed] ⏭️ 当前页面不在支持的平台上');
      return;
    }
    
    console.log(`[CleanFeed] 🎯 检测到平台: ${this.currentPlatform.name}`);
    
    // 监听设置变化
    this.listenToSettings();
    
    // 初始扫描
    this.scanAndDetect();
    
    // 设置监听，页面内容变化时重新扫描
    this.setupObserver();
  }

  private async loadSettings() {
    try {
      const data = await chrome.storage.local.get('settings');
      if (data.settings) {
        this.settings = { ...DEFAULT_SETTINGS, ...data.settings };
        console.log('[CleanFeed] ⚙️ 设置加载成功:', this.settings);
      }
    } catch (error) {
      console.warn('[CleanFeed] ⚠️ 加载设置失败:', error);
    }
  }

  private listenToSettings() {
    chrome.storage.onChanged.addListener((changes) => {
      if (changes.settings) {
        this.settings = { ...DEFAULT_SETTINGS, ...changes.settings.newValue };
        console.log('[CleanFeed] 🔄 设置已更新:', this.settings);
        this.reapplyEffects();
      }
    });
  }

  private detectCurrentPlatform() {
    const url = window.location.href;
    for (const [id, platform] of this.platforms) {
      if (platform.matchUrl(url)) {
        this.currentPlatform = platform;
        break;
      }
    }
  }

  private setupObserver() {
    this.mutationObserver = new MutationObserver(() => {
      this.debounceScan();
    });

    this.mutationObserver.observe(document.body, {
      childList: true,
      subtree: true,
    });
    console.log('[CleanFeed] 👀 MutationObserver 已设置');
  }

  private debounceScan() {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }
    this.debounceTimer = setTimeout(() => {
      this.scanAndDetect();
    }, 1000);
  }

  private async scanAndDetect() {
    if (!this.settings.enabled || this.isProcessing || !this.currentPlatform) {
      return;
    }

    const platformConfig = this.settings.platforms[this.currentPlatform.id];
    if (!platformConfig?.enabled) {
      console.log(`[CleanFeed] ⏭️ 平台 ${this.currentPlatform.name} 已禁用`);
      return;
    }

    this.isProcessing = true;
    console.log('[CleanFeed] 🔍 开始扫描...');

    try {
      const posts = this.currentPlatform.getPosts();
      console.log(`[CleanFeed] 📝 找到 ${posts.length} 个帖子`);
      
      for (const post of posts) {
        // 检查是否已检测过
        const hasAllResults = platformConfig.detectors.every(id => 
          post.detectionResults?.has(id)
        );
        
        if (hasAllResults) {
          continue;
        }

        // 运行所有配置的检测器
        const results: DetectionResult[] = [];
        for (const detectorId of platformConfig.detectors) {
          const detector = this.detectors.get(detectorId);
          const detectorConfig = this.settings.detectors[detectorId];
          
          if (!detector || !detectorConfig?.enabled) {
            continue;
          }

          const result = await detector.detect(post.text);
          post.detectionResults = post.detectionResults || new Map();
          post.detectionResults.set(detectorId, result);
          results.push(result);

          // 如果触发过滤，应用效果
          if (result.shouldFilter && result.confidence >= detectorConfig.threshold) {
            this.applyEffects(post, result, platformConfig.effects);
          }
        }
      }
      
      console.log(`[CleanFeed] ✅ 扫描完成，共处理 ${posts.length} 个帖子`);
    } catch (error) {
      console.error('[CleanFeed] ❌ 检测过程出错:', error);
    } finally {
      this.isProcessing = false;
    }
  }

  private applyEffects(post: Post, result: DetectionResult, effectIds: string[]) {
    for (const effectId of effectIds) {
      const effect = this.effects.get(effectId);
      const effectConfig = this.settings.effects[effectId];
      
      if (!effect || !effectConfig?.enabled) {
        continue;
      }

      effect.apply(
        this.currentPlatform!.getContainer(post.element), 
        result, 
        effectConfig.options
      );
    }
  }

  private reapplyEffects() {
    console.log('[CleanFeed] 🔄 重新应用效果');
    // 这里可以实现效果的重新应用
  }

  public getSettings(): PluginSettings {
    return this.settings;
  }

  public async updateSettings(settings: Partial<PluginSettings>) {
    this.settings = { ...this.settings, ...settings };
    await chrome.storage.local.set({ settings: this.settings });
  }
}
