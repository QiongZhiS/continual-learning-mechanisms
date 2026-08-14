"""E2 记忆场（Titans 式惊喜写入）——调度层记忆组件（H1a · M2）

设计（AGI-实验方案.md §4 E2 + v0.23 S4 接口对齐）：
- 写入策略：surprise（开，Titans 式：只写意外样本）/ full（关，全量写），容量 C 固定
- 惊喜分数：逐样本 CE loss（stablemax_cross_entropy，与训练同 loss_fn）——预测误差 = 意外程度代理
- 开臂阈值：滑动窗口移动平均 + k·σ（意外 = 高于近期均值）；开臂淘汰 = 低 surprise 优先（保留高价值）
- 关臂淘汰 = FIFO（最老先淘汰）——两臂最终内容不同，容量相同，测"内容选择"的容量效率
- 读取：优先级采样（surprise 加权）/ 均匀采样——E3-D 夜间回放（S4: r=0.25）/ E6a 机制臂共用接口
"""
import numpy as np


class MemoryField:
    def __init__(self, capacity=2000, strategy="full", k=1.0, seed=0):
        assert strategy in ("surprise", "full"), f"未知策略: {strategy}"
        self.capacity = capacity
        self.strategy = strategy
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.entries = []        # (inputs[81], labels[81], ids, surprise, ts, domain)
        self.surprise_hist = []  # 滑动窗口（动态阈值，最近 1000 条）
        self.ts = 0

    # ---------------- 写入 ----------------
    def update(self, inputs, labels, ids, surprise, domain):
        """inputs/labels/ids: np arrays (B, 81)/(B, 81)/(B,)；surprise: (B,) float"""
        surprise = np.asarray(surprise, dtype=np.float64)
        if self.strategy == "surprise":
            if self.surprise_hist:
                hist = np.asarray(self.surprise_hist)
                thr = float(hist.mean() + self.k * hist.std())
            else:
                thr = float(surprise.mean() + self.k * surprise.std())
            keep = surprise >= thr
        else:
            keep = np.ones(len(surprise), dtype=bool)
        self.surprise_hist.extend(surprise.tolist())
        self.surprise_hist = self.surprise_hist[-1000:]
        for i in np.where(keep)[0]:
            self.entries.append((np.asarray(inputs[i]).copy(),
                                 np.asarray(labels[i]).copy(),
                                 int(ids[i]),
                                 float(surprise[i]),
                                 self.ts, domain))
            self.ts += 1
        self._trim()

    def _trim(self):
        if len(self.entries) <= self.capacity:
            return
        if self.strategy == "surprise":
            self.entries.sort(key=lambda e: e[3])      # 低 surprise 先淘汰 → 保留高价值
            self.entries = self.entries[-self.capacity:]
        else:
            self.entries = self.entries[-self.capacity:]  # FIFO：最老先淘汰

    # ---------------- 读取 ----------------
    def sample(self, n, mode="priority"):
        """priority: surprise 加权（softmax 权重）；uniform: 等概率。返回 (inputs, labels, ids)"""
        if not self.entries:
            return None
        idx = np.arange(len(self.entries))
        if mode == "priority":
            s = np.array([e[3] for e in self.entries], dtype=np.float64)
            w = np.exp((s - s.max()) / max(float(s.std()), 1e-6))
            w = w / w.sum()
            pick = self.rng.choice(idx, size=n, replace=True, p=w)
        else:
            pick = self.rng.choice(idx, size=n, replace=True)
        inputs = np.stack([self.entries[i][0] for i in pick])
        labels = np.stack([self.entries[i][1] for i in pick])
        ids = np.array([self.entries[i][2] for i in pick])
        return inputs, labels, ids

    # ---------------- 状态 ----------------
    def stats(self):
        s = np.array([e[3] for e in self.entries], dtype=np.float64) if self.entries else None
        return {
            "strategy": self.strategy, "capacity": self.capacity,
            "n": len(self.entries),
            "mean_surprise": round(float(s.mean()), 4) if s is not None else None,
            "domains": sorted({e[5] for e in self.entries}),
        }
