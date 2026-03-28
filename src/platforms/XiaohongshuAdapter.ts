import { PlatformAdapter } from './PlatformAdapter';
import { Post } from '../core/types';

export class XiaohongshuAdapter extends PlatformAdapter {
  readonly id = 'xiaohongshu';
  readonly name = '小红书';

  private static instance: XiaohongshuAdapter;

  public static getInstance(): XiaohongshuAdapter {
    if (!XiaohongshuAdapter.instance) {
      XiaohongshuAdapter.instance = new XiaohongshuAdapter();
    }
    return XiaohongshuAdapter.instance;
  }

  matchUrl(url: string): boolean {
    return url.includes('xiaohongshu.com');
  }

  getPosts(): Post[] {
    const posts: Post[] = [];
    
    // 探索页/推荐页卡片
    const exploreCards = document.querySelectorAll('.feeds-page .note-item, .feeds-container .note-item, [data-v-0d54d578], [class*="note-item"], [class*="explore-feed"]');
    
    // 搜索页卡片
    const searchCards = document.querySelectorAll('.search-page .note-item, .search-result .note-item');
    
    // 通用 fallback: 找所有可能是帖子卡片的元素
    const allCards = document.querySelectorAll('[class*="card"], [class*="feed"], [class*="note"], [class*="post"], section');
    
    console.log(`[CleanFeed] 📊 选择器统计: explore=${exploreCards.length}, search=${searchCards.length}, all=${allCards.length}`);
    
    // 合并所有找到的元素
    const elements = new Set<Element>();
    exploreCards.forEach(el => elements.add(el));
    searchCards.forEach(el => elements.add(el));
    allCards.forEach(el => elements.add(el));
    
    console.log(`[CleanFeed] 📝 去重后找到 ${elements.size} 个潜在帖子元素`);

    elements.forEach((element, index) => {
      const html = element.innerHTML.toLowerCase();
      
      // 必须包含文本内容
      const text = this.extractText(element as HTMLElement);
      if (text.length < 10) {
        return;
      }
      
      // 必须看起来像一个帖子卡片
      const hasPostSignals = (
        html.includes('点赞') || html.includes('like') ||
        html.includes('收藏') || html.includes('collect') ||
        html.includes('评论') || html.includes('comment') ||
        html.includes('分享') || html.includes('share') ||
        html.includes('作者') || html.includes('author') ||
        (text.length > 20 && element.querySelectorAll('img').length > 0)
      );
      
      if (!hasPostSignals) {
        return;
      }
      
      // 检查是否已经被处理过
      if ((element as HTMLElement).dataset.feedProcessed === 'true') {
        return;
      }
      
      // 标记为已处理
      (element as HTMLElement).dataset.feedProcessed = 'true';
      
      const post: Post = {
        id: this.generateId(element as HTMLElement),
        element: element as HTMLElement,
        text,
        detectionResults: new Map(),
      };
      
      posts.push(post);
    });

    console.log(`[CleanFeed] ✅ 筛选出 ${posts.length} 个有效帖子`);
    return posts;
  }

  extractText(element: HTMLElement): string {
    // 获取所有文本节点
    const textParts: string[] = [];
    
    const extract = (node: Node) => {
      if (node.nodeType === Node.TEXT_NODE) {
        const text = node.textContent?.trim() || '';
        if (text.length > 0) {
          textParts.push(text);
        }
      } else if (node.nodeType === Node.ELEMENT_NODE) {
        const elem = node as Element;
        // 跳过 script, style, svg 等标签
        const tagName = elem.tagName.toLowerCase();
        if (tagName === 'script' || tagName === 'style' || tagName === 'svg' || tagName === 'noscript') {
          return;
        }
        // 递归处理子节点
        elem.childNodes.forEach(extract);
      }
    };
    
    element.childNodes.forEach(extract);
    
    // 合并文本，用空格分隔
    return textParts.join(' ').replace(/\s+/g, ' ').trim();
  }

  getContainer(element: HTMLElement): HTMLElement {
    return element;
  }

  isDetailPage(): boolean {
    return window.location.pathname.includes('/explore/') || 
           window.location.search.includes('noteId');
  }

  extractDetailContent(): string {
    const selectors = [
      '.note-content',
      '.detail-content',
      '[class*="content"] article',
      '[class*="note"] p',
      'main section'
    ];
    
    for (const selector of selectors) {
      const element = document.querySelector(selector);
      if (element) {
        const text = this.extractText(element as HTMLElement);
        if (text.length > 50) {
          return text;
        }
      }
    }
    
    return this.extractText(document.body);
  }
}
