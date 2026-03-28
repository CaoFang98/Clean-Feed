import './styles.css';
import { ContentSieve } from './core/ContentSieve';

// 启动 CleanFeed
console.log('[CleanFeed] 🚀 正在启动 CleanFeed...');
const sieve = ContentSieve.getInstance();
sieve.init();
