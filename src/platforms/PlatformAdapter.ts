import { Post } from '../core/types';

export abstract class PlatformAdapter {
  abstract readonly id: string;
  abstract readonly name: string;

  /**
   * 检查当前URL是否匹配该平台
   */
  abstract matchUrl(url: string): boolean;

  /**
   * 获取页面上所有帖子
   */
  abstract getPosts(): Post[];

  /**
   * 从帖子元素中提取文本内容
   */
  abstract extractText(element: HTMLElement): string;

  /**
   * 获取帖子的容器元素（用于应用效果）
   */
  abstract getContainer(element: HTMLElement): HTMLElement;

  /**
   * 生成帖子唯一ID
   */
  protected generateId(element: HTMLElement): string {
    return `${this.id}-${Math.random().toString(36).substr(2, 9)}`;
  }
}
