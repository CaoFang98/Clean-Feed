// 通用类型定义

export interface Post {
  id: string;
  element: HTMLElement;
  text: string;
  metadata?: Record<string, any>;
  detectionResults?: Map<string, DetectionResult>;
}

export interface DetectionResult {
  detectorId: string;
  shouldFilter: boolean;
  confidence: number;
  metadata?: any;
}

export interface PlatformConfig {
  id: string;
  name: string;
  urlPattern: string;
  detectors: string[];
  effects: string[];
  enabled: boolean;
}

export interface DetectorConfig {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  threshold: number;
}

export interface EffectConfig {
  id: string;
  name: string;
  enabled: boolean;
  options?: Record<string, any>;
}

export interface PluginSettings {
  enabled: boolean;
  platforms: Record<string, PlatformConfig>;
  detectors: Record<string, DetectorConfig>;
  effects: Record<string, EffectConfig>;
}

export const DEFAULT_PLATFORM_CONFIGS: Record<string, PlatformConfig> = {
  xiaohongshu: {
    id: 'xiaohongshu',
    name: '小红书',
    urlPattern: 'https://www.xiaohongshu.com/*',
    detectors: ['ai-content'],
    effects: ['blur', 'label'],
    enabled: true,
  },
  zhihu: {
    id: 'zhihu',
    name: '知乎',
    urlPattern: 'https://www.zhihu.com/*',
    detectors: ['low-quality'],
    effects: ['blur', 'label'],
    enabled: true,
  },
};

export const DEFAULT_DETECTOR_CONFIGS: Record<string, DetectorConfig> = {
  'ai-content': {
    id: 'ai-content',
    name: 'AI内容检测',
    description: '检测AI生成的内容',
    enabled: true,
    threshold: 0.7,
  },
  'low-quality': {
    id: 'low-quality',
    name: '低质量内容检测',
    description: '检测低质量/水帖内容',
    enabled: true,
    threshold: 0.6,
  },
};

export const DEFAULT_EFFECT_CONFIGS: Record<string, EffectConfig> = {
  blur: {
    id: 'blur',
    name: '虚化效果',
    enabled: true,
    options: { intensity: 8 },
  },
  label: {
    id: 'label',
    name: '标签提示',
    enabled: true,
  },
  hide: {
    id: 'hide',
    name: '完全隐藏',
    enabled: false,
  },
};

export const DEFAULT_SETTINGS: PluginSettings = {
  enabled: true,
  platforms: DEFAULT_PLATFORM_CONFIGS,
  detectors: DEFAULT_DETECTOR_CONFIGS,
  effects: DEFAULT_EFFECT_CONFIGS,
};
