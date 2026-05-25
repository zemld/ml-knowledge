# FlashAttention

> IO-aware алгоритм точного вычисления attention, который ускоряет Transformer за счёт минимизации операций чтения/записи в медленную память GPU, не жертвуя точностью результата.

## Проблема наивной реализации

Стандартный scaled dot-product attention вычисляется в три этапа, каждый из которых — отдельный GPU-ядро:

1. $S = QK^\top / \sqrt{d_k}$ — записывается матрица $N \times N$ в HBM.
2. $P = \text{softmax}(S)$ — читается из HBM и снова записывается $N \times N$.
3. $O = PV$ — читается $P$ из HBM.

При длине последовательности $N$ и размерности головы $d$ суммарное число обращений к HBM составляет $O(N^2 + Nd)$. Для длинных последовательностей доминирует $N^2$-член — матрица внимания становится огромным узким местом памяти. При $N=8192$ она занимает ~256 МБ только для одной головы.

## Иерархия памяти GPU

| Уровень | Размер (A100) | Пропускная способность |
|---|---|---|
| **HBM** (глобальная память) | 40–80 ГБ | ~2 ТБ/с |
| **SRAM** (on-chip, shared memory) | ~192 КБ на SM | ~19 ТБ/с |

SRAM примерно в 10 раз быстрее HBM, но в 100 000 раз меньше. Наивный attention работает медленно не из-за нехватки вычислительной мощности, а потому что постоянно гоняет данные через медленный HBM. Операция memory-bound, а не compute-bound.

## Основная идея: блочное вычисление (tiling)

FlashAttention разбивает матрицы $Q$, $K$, $V$ на блоки размера $B$, которые умещаются в SRAM:

```
Для каждого блока K_j, V_j (из HBM → SRAM):
    Для каждого блока Q_i (из HBM → SRAM):
        Вычислить S_ij = Q_i K_j^T / sqrt(d_k)
        Обновить текущий выход O_i через online softmax
    Записать финальный O_i → HBM
```

Промежуточная матрица $N \times N$ никогда не материализуется в HBM — все промежуточные скоры живут только в SRAM. В HBM записывается лишь финальный выход $O$ размером $N \times d$.

## Online softmax trick

Стандартный softmax требует двух проходов: сначала найти максимум (для численной стабильности), затем вычислить нормировку по всей строке. При блочном вычислении вся строка недоступна сразу.

Решение — поддерживать бегущую статистику для каждой строки:
- $m_i$ — текущий максимум скоров (для стабильности)
- $\ell_i$ — текущая нормировочная сумма

При добавлении нового блока скоров $S_{\text{new}}$:

$$m_{\text{new}} = \max(m_{\text{old}},\ \max(S_{\text{new}}))$$

$$\ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \cdot \ell_{\text{old}} + \sum e^{S_{\text{new}} - m_{\text{new}}}$$

Накопленный выход корректируется аналогичным масштабирующим коэффициентом. По окончании всех блоков $O_i / \ell_i$ даёт точно тот же результат, что и обычный softmax по полной строке.

## Сложность IO-операций

| Реализация | Чтений/записей HBM |
|---|---|
| Стандартный attention | $O(N^2 + Nd)$ |
| FlashAttention | $O(N^2 d^2 / M)$ |

Здесь $M$ — размер SRAM, $d$ — размерность головы. При типичных значениях ($d = 64\text{–}128$, $M \approx 192\text{ КБ}$) знаменатель $M/d^2 \gg 1$, и FlashAttention делает существенно меньше HBM-обращений. Память для хранения attention-матрицы падает с $O(N^2)$ до $O(N)$ — хранятся только бегущие статистики $(m, \ell)$ и выход $O$.

## Обратный проход: recomputation

При обратном распространении наивный attention требует хранить матрицу $P$ размером $N \times N$. FlashAttention вместо этого сохраняет только $(m, \ell)$ — и при backward pass перевычисляет блоки $S$ и $P$ заново прямо из $Q, K, V$. Это стоит дополнительных вычислений, но экономит память и сокращает HBM-трафик.

## Версии

| Версия | Год | Ключевые улучшения |
|---|---|---|
| **FlashAttention-1** | 2022 | IO-aware tiling, online softmax, ядро без материализации $N \times N$ |
| **FlashAttention-2** | 2023 | Лучшее разбиение работы по варпам, параллелизм по измерению длины последовательности, поддержка MQA/GQA; ~2× быстрее FA1, 50–73% теор. пропускной способности A100 |
| **FlashAttention-3** | 2024 | Оптимизация под Hopper (H100): асинхронность через TMA, специализация варпов, перемежение matmul и softmax, low-precision (FP8) |

## Связанные темы

- [[neural-networks/transformer/attention-mechanism|Механизм внимания: Q, K, V]]
- [[neural-networks/transformer/overview|Transformer — обзор архитектуры]]

## Источники

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness — Dao et al., 2022](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning — Dao, 2023](https://arxiv.org/abs/2307.08691)
- [FlashAttention: IO-Aware Exact Attention — mbrenndoerfer.com](https://mbrenndoerfer.com/writing/flashattention-io-aware-exact-attention-long-context-language-models)
- [Why FlashAttention? — Medium](https://medium.com/@katherineolowookere/why-flashattention-4b0f6cca8653)
- [From Online Softmax to FlashAttention — UW CSE599M notes](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
