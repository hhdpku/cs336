import regex as re
from collections import Counter
from collections.abc import Iterable, Iterator
import heapq
from tqdm import tqdm

# 文档提供的 GPT-2 正则表达式
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pre_tokenize(text):
    return [match.group() for match in re.finditer(PAT, text)]

class MergeElement:
    def __init__(self, pair, freq):
        self.pair = pair
        self.freq = freq

    def __lt__(self, other):
        # 1. 首先比较频率（频率高的排在“前面”，即值更小）
        if self.freq != other.freq:
            return self.freq > other.freq  # 注意这里是 >，因为我们要模拟大根堆
        
        # 2. 频率相同时，比较词典序（词典序大的排在“前面”）
        # 根据作业：max([("A", "B"), ("BA", "A")]) -> ("BA", "A")
        return self.pair > other.pair

    def __repr__(self):
        return f"Element(pair={self.pair}, freq={self.freq})"
    
class BPETokenizer:
    def __init__(self, vocab_size, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []

        self.vocab = {}          # bytes -> token_id
        self.inv_vocab = {}      # token_id -> bytes
        self.merges = []         # [(bytes, bytes), ...]
        
        # 内部加速缓存
        self._merge_ranks = None
        self.encode_cache = {}

    @property
    def vocab_to_id(self):
        """兼容性属性，指向 self.vocab"""
        return self.vocab

    @property
    def merge_ranks(self):
        """延迟初始化优先级字典，加速编码过程"""
        if self._merge_ranks is None:
            self._merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        return self._merge_ranks

    def init_vocab(self):
        """初始化基础词表：特殊令牌 + 256个原始字节"""
        idx = 0
        # 优先添加特殊令牌
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            if b not in self.vocab:
                self.vocab[b] = idx
                self.inv_vocab[idx] = b
                idx += 1

        # 添加基础字节
        for i in range(256):
            b = bytes([i])
            if b not in self.vocab:
                self.vocab[b] = idx
                self.inv_vocab[idx] = b
                idx += 1

    def _update_pair_counts(self, old_tuple, new_tuple, count, pair_freqs):
        affected_pairs = set()

        # 1. old word 中的 pair 被移除
        for i in range(len(old_tuple) - 1):
            pair = (old_tuple[i], old_tuple[i + 1])
            pair_freqs[pair] -= count
            affected_pairs.add(pair)

            if pair_freqs[pair] <= 0:
                del pair_freqs[pair]

        # 2. new word 中的新 pair 被加入
        for i in range(len(new_tuple) - 1):
            pair = (new_tuple[i], new_tuple[i + 1])
            pair_freqs[pair] += count
            affected_pairs.add(pair)

        return affected_pairs

    def train(self, corpus: str):
        self.init_vocab()

        # 1. 预处理：保护特殊令牌，不参与合并
        if self.special_tokens:
            special_pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
            parts = re.split(special_pattern, corpus)
        else:
            parts = [corpus]
        
        # 2. 统计唯一单词及其频率
        word_freqs = Counter()
        for part in parts:
            if part in self.special_tokens or not part:
                continue
            for tok in pre_tokenize(part):
                byte_tuple = tuple(bytes([b]) for b in tok.encode("utf-8"))
                word_freqs[byte_tuple] += 1

        # 3. 统计初始字节对频率
        pair_freqs = Counter()
        for word_tuple, count in word_freqs.items():
            for i in range(len(word_tuple) - 1):
                pair_freqs[(word_tuple[i], word_tuple[i+1])] += count
        
        heap = []
        for pair, freq in pair_freqs.items():
            heapq.heappush(heap, MergeElement(pair, freq))

        # 4. 迭代合并
        while len(self.vocab) < self.vocab_size:
            if not pair_freqs:
                break

            # 选最高频对，若频率相同选词典序最大的 (Tie-breaking)
            while heap:
                top = heapq.heappop(heap)
                pair = top.pair
                freq = top.freq

                # Lazy 校验：跳过过期 pair
                if pair not in pair_freqs:
                    continue
                if pair_freqs[pair] != freq:
                    continue
                best_pair = pair
                break
            else:
                break
            new_token = best_pair[0] + best_pair[1]
            
            self.merges.append(best_pair)
            idx = len(self.vocab)
            self.vocab[new_token] = idx
            self.inv_vocab[idx] = new_token

            # 5. 增量更新受合并影响的词
            new_word_freqs = Counter()
            for word_tuple, count in word_freqs.items():
                if best_pair[0] in word_tuple and best_pair[1] in word_tuple:
                    new_word_tuple = self._apply_merge(list(word_tuple), best_pair)
                    new_word_tuple = tuple(new_word_tuple)
                    affected_pairs = self._update_pair_counts(word_tuple, new_word_tuple, count, pair_freqs)
                    for p in affected_pairs:
                        if p in pair_freqs:
                            heapq.heappush(heap, MergeElement(p, pair_freqs[p]))
                    new_word_freqs[new_word_tuple] += count
                else:
                    new_word_freqs[word_tuple] += count
            word_freqs = new_word_freqs

        
        # 训练结束后重置缓存
        self._merge_ranks = None

    def train_from_stats(self, word_freqs: Counter):
        self.init_vocab()
        
        # 1. 建立初始 pair 统计 和 反向索引
        pair_freqs = Counter()
        pair_to_words = {} # pair -> set(word_tuple)
        
        for word_tuple, count in word_freqs.items():
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i+1])
                pair_freqs[pair] += count
                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add(word_tuple)

        # 2. 初始化大根堆
        heap = [MergeElement(p, f) for p, f in pair_freqs.items() if f > 0]
        heapq.heapify(heap)

        # 3. 迭代合并循环
        while len(self.vocab) < self.vocab_size:
            if not heap: break
            top = heapq.heappop(heap)
            
            # 延迟删除校验
            if top.freq != pair_freqs.get(top.pair, 0): continue
            if top.freq == 0: break

            best_pair = top.pair
            new_token = best_pair[0] + best_pair[1]
            
            # --- 高效增量更新开始 ---
            # 仅遍历包含 best_pair 的词
            if best_pair in pair_to_words:
                # list() 是为了在循环中安全修改 set
                for word_tuple in list(pair_to_words[best_pair]):
                    count = word_freqs[word_tuple]
                    
                    # A. 清理旧词贡献：从所有受影响对的频率中减去该词的 count
                    for i in range(len(word_tuple) - 1):
                        p = (word_tuple[i], word_tuple[i+1])
                        pair_freqs[p] -= count
                        # 这里不需要立刻从 pair_to_words 移除，最后统一清理
                    
                    # B. 执行合并
                    new_word_tuple = tuple(self._apply_merge(list(word_tuple), best_pair))
                    
                    # C. 更新全局统计
                    del word_freqs[word_tuple]
                    word_freqs[new_word_tuple] += count
                    
                    # D. 添加新词贡献：更新新词产生的对频率，并推入堆，更新索引
                    for i in range(len(new_word_tuple) - 1):
                        p = (new_word_tuple[i], new_word_tuple[i+1])
                        pair_freqs[p] += count
                        if p not in pair_to_words:
                            pair_to_words[p] = set()
                        pair_to_words[p].add(new_word_tuple)
                        # 重新入堆
                        heapq.heappush(heap, MergeElement(p, pair_freqs[p]))
                
                # 清理已经失效的旧索引
                del pair_to_words[best_pair]

            # 记录 merge
            self.merges.append(best_pair)
            self.vocab[new_token] = len(self.vocab)
            self.inv_vocab[len(self.vocab)-1] = new_token


    def encode(self, text: str) -> list[int]:
        # 按长度降序排列特殊令牌以实现贪心匹配
        sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
        if self.special_tokens:
            special_pattern = "(" + "|".join(re.escape(t) for t in sorted_specials) + ")"
            parts = re.split(special_pattern, text)
        else:
            parts = [text]

        ids = []
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.vocab[part.encode("utf-8")])
            elif part:
                ids.extend(self._encode_text_block(part))
        return ids

    def _encode_text_block(self, text: str) -> list[int]:
        ids = []
        for tok in pre_tokenize(text):
            if tok in self.encode_cache:
                ids.extend(self.encode_cache[tok])
                continue
            word = [bytes([b]) for b in tok.encode("utf-8")]
            
            while len(word) > 1:
                # 获取当前序列中所有相邻对
                pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
                # 寻找在训练中最先出现的合并对
                best_pair = min(pairs, key=lambda p: self.merge_ranks.get(p, float('inf')))
                
                if best_pair not in self.merge_ranks:
                    break
                
                word = self._apply_merge(word, best_pair)
            tok_ids = [self.vocab[b] for b in word]
            self.encode_cache[tok] = tok_ids
            ids.extend(tok_ids)
    
        return ids

    def _apply_merge(self, word: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                new_word.append(word[i] + word[i+1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return new_word

    def decode(self, token_ids: list[int]) -> str:
        # 使用 errors='replace' 处理非法或不完整的字节序列 (U+FFFD)
        bs = b"".join(self.inv_vocab[i] for i in token_ids)
        return bs.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def save(self, vocab_filepath: str, merges_filepath: str):
        # 1. 保存 vocab：id \t repr(bytes)
        with open(vocab_filepath, "w", encoding="utf-8") as f:
            for token_id, token_bytes in sorted(self.inv_vocab.items()):
                f.write(f"{token_id}\t{repr(token_bytes)}\n")

        # 2. 保存 merges：repr(bytes1) repr(bytes2)
        with open(merges_filepath, "w", encoding="utf-8") as f:
            for b1, b2 in self.merges:
                f.write(f"{repr(b1)}\t{repr(b2)}\n")

    @classmethod
    def load(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = {}

        with open(vocab_filepath, "r", encoding="utf-8") as f:
            for line in f:
                idx_str, b_repr = line.rstrip().split("\t")
                token_bytes = eval(b_repr)
                assert isinstance(token_bytes, (bytes, bytearray))
                vocab[int(idx_str)] = token_bytes

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                b1_repr, b2_repr = line.rstrip().split("\t")
                b1 = eval(b1_repr)
                b2 = eval(b2_repr)
                merges.append((b1, b2))

        instance = cls(vocab_size=len(vocab), special_tokens=special_tokens)
        instance.inv_vocab = vocab
        instance.vocab = {v: k for k, v in vocab.items()}
        instance.merges = merges
        instance._merge_ranks = None

        return instance

    
