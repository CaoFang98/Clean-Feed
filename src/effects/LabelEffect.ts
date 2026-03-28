import { Effect } from './Effect';
import { DetectionResult } from '../core/types';

export class LabelEffect extends Effect {
  readonly id = 'label';
  readonly name = '标签提示';

  private static instance: LabelEffect;

  public static getInstance(): LabelEffect {
    if (!LabelEffect.instance) {
      LabelEffect.instance = new LabelEffect();
    }
    return LabelEffect.instance;
  }

  apply(element: HTMLElement, result: DetectionResult, options?: Record<string, any>): void {
    // 先找到或创建包装器
    let wrapper = element.parentElement;
    if (!wrapper?.classList.contains('cf-wrapper')) {
      wrapper = document.createElement('div');
      wrapper.className = 'cf-wrapper';
      element.parentNode?.insertBefore(wrapper, element);
      wrapper.appendChild(element);
    }

    // 检查是否已有标签
    if (wrapper.querySelector('.cf-label')) {
      return;
    }

    // 创建标签
    const label = document.createElement('div');
    label.className = 'cf-label';
    
    let labelText = '检测到过滤内容';
    if (result.detectorId === 'ai-content') {
      labelText = '🤖 AI生成内容';
    } else if (result.detectorId === 'low-quality') {
      labelText = '⚠️ 低质量内容';
    }

    label.innerHTML = `
      <div class="cf-label-text">${labelText}</div>
      <div class="cf-label-confidence">置信度: ${(result.confidence * 100).toFixed(1)}%</div>
    `;

    wrapper.insertBefore(label, wrapper.firstChild);

    console.log(`[CleanFeed] 🏷️ 应用标签效果`);
  }

  remove(element: HTMLElement): void {
    const wrapper = element.parentElement;
    if (wrapper?.classList.contains('cf-wrapper')) {
      const label = wrapper.querySelector('.cf-label');
      label?.remove();
    }
  }
}
