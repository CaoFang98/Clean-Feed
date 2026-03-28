import { Effect } from './Effect';
import { DetectionResult } from '../core/types';

export class BlurEffect extends Effect {
  readonly id = 'blur';
  readonly name = '虚化效果';

  private static instance: BlurEffect;

  public static getInstance(): BlurEffect {
    if (!BlurEffect.instance) {
      BlurEffect.instance = new BlurEffect();
    }
    return BlurEffect.instance;
  }

  apply(element: HTMLElement, result: DetectionResult, options?: Record<string, any>): void {
    const intensity = options?.intensity || 8;
    
    // 创建包装器（如果不存在）
    let wrapper = element.parentElement;
    if (!wrapper?.classList.contains('cf-wrapper')) {
      wrapper = document.createElement('div');
      wrapper.className = 'cf-wrapper';
      element.parentNode?.insertBefore(wrapper, element);
      wrapper.appendChild(element);
    }

    // 应用虚化
    element.classList.add('cf-blur');
    element.style.filter = `blur(${intensity}px)`;

    console.log(`[CleanFeed] 🌫️ 应用虚化效果，强度=${intensity}px`);
  }

  remove(element: HTMLElement): void {
    element.classList.remove('cf-blur');
    element.style.filter = 'none';
  }
}
