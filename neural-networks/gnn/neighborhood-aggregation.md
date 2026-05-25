# Neighborhood Aggregation (Message Passing) в GNN

> Универсальный алгоритм обучения представлений вершин: каждый узел итерационно собирает информацию от соседей, агрегирует её и обновляет своё состояние — так структура графа встраивается в числовые векторы.

## Основная идея

Цель та же, что у [[neural-networks/gnn/node2vec|node2vec]] — получить для каждой вершины вектор $\mathbf{h}_v$, отражающий её место в графе. Но подход принципиально другой: вместо случайных блужданий используется **прямое распространение признаков** через нейросеть.

На каждом слое каждый узел:
1. Получает сообщения от всех своих соседей.
2. Агрегирует их в одно число/вектор.
3. Объединяет агрегат со своим текущим состоянием и обновляет представление.

Это повторяется $K$ раз — по числу слоёв.

## Формальное описание

На слое $k$ для вершины $v$ с множеством соседей $\mathcal{N}(v)$:

$$\mathbf{m}_{\mathcal{N}(v)}^{(k)} = \text{AGGREGATE}^{(k)}\!\left(\left\{\mathbf{h}_u^{(k-1)} : u \in \mathcal{N}(v)\right\}\right)$$

$$\mathbf{h}_v^{(k)} = \text{COMBINE}^{(k)}\!\left(\mathbf{h}_v^{(k-1)},\; \mathbf{m}_{\mathcal{N}(v)}^{(k)}\right)$$

Начальное состояние: $\mathbf{h}_v^{(0)} = \mathbf{x}_v$ — исходные признаки вершины.

Финальный эмбеддинг после $K$ слоёв: $\mathbf{z}_v = \mathbf{h}_v^{(K)}$.

Обе функции — **AGGREGATE** и **COMBINE** — содержат обучаемые параметры. Градиент проходит через них во время обратного распространения.

## Рецептивное поле: что «видит» вершина

После $K$ слоёв агрегации вершина $v$ интегрирует информацию со всех вершин, находящихся не дальше $K$ переходов:

| Слоёв $K$ | Что охвачено |
|---|---|
| 1 | Только прямые соседи |
| 2 | Соседи соседей |
| $K$ | Весь $K$-hop окрестности |

На практике $K = 2{-}3$ — более глубокие сети страдают от **over-smoothing**: эмбеддинги всех вершин сходятся к единому вектору, теряя различимость.

## Функции AGGREGATE: варианты

AGGREGATE должна быть **инвариантной к перестановкам** — порядок соседей не определён.

**Mean (среднее)**

$$\mathbf{m} = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u$$

Нормализует по числу соседей. Теряет информацию о размере окрестности.

**Sum (сумма)**

$$\mathbf{m} = \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u$$

Сохраняет информацию о степени вершины. GIN (Graph Isomorphism Network) использует sum для максимальной выразительности.

**Max-pooling**

$$\mathbf{m}_j = \max_{u \in \mathcal{N}(v)} \mathbf{h}_{u,j}$$

Выбирает наиболее выраженный признак по каждой размерности. Хорошо выявляет экстремальные паттерны.

**Attention (GAT)**

$$\mathbf{m} = \sum_{u \in \mathcal{N}(v)} \alpha_{vu}\, \mathbf{h}_u, \quad \alpha_{vu} = \text{softmax}_u\!\left(a\!\left(\mathbf{W}\mathbf{h}_v,\, \mathbf{W}\mathbf{h}_u\right)\right)$$

Каждый сосед получает **обученный вес** $\alpha_{vu}$, зависящий от содержимого вершин. Анизотропная агрегация — не все соседи равнозначны.

## Три ключевые архитектуры

| Архитектура | Агрегация | Веса соседей | Масштабируемость |
|---|---|---|---|
| **GCN** | Нормированная сумма (mean по степеням) | Фиксированные (степень узла) | Низкая — нужен весь граф |
| **GraphSAGE** | Mean / Max / LSTM (настраивается) | Обучаемые | Высокая — сэмплирует подмножество соседей |
| **GAT** | Взвешенная сумма | Обучаемые через attention | Средняя — attention дорогой |

**GCN** — самый простой вариант. Базовый шаг обновления:

$$\mathbf{H}^{(k)} = \sigma\!\left(\tilde{D}^{-1/2}\tilde{A}\,\tilde{D}^{-1/2}\,\mathbf{H}^{(k-1)}\mathbf{W}^{(k)}\right)$$

где $\tilde{A} = A + I$ (матрица смежности с добавленными self-loop), $\tilde{D}$ — матрица степеней.

## Neighborhood Aggregation vs node2vec

| | Neighborhood Aggregation | node2vec |
|---|---|---|
| **Признаки вершин** | Использует напрямую | Игнорирует |
| **Подход** | Нейросеть, обратное распространение | Случайные блуждания + Skip-Gram |
| **Тип обучения** | Supervised / Semi-supervised | Unsupervised |
| **Новые вершины** | GraphSAGE — индуктивный | Трансдуктивный, нужно переобучение |
| **Точность с метками** | Выше (end-to-end) | Ниже |

## Связанные темы

- [[neural-networks/gnn/overview|Графовые нейросети (GNN)]]
- [[neural-networks/gnn/node2vec|node2vec]] — альтернативный подход через блуждания без признаков
- [[neural-networks/gnn/node-level-tasks|Node-level задачи]] — классификация вершин на основе эмбеддингов
- [[neural-networks/transformer/attention-mechanism|Механизм внимания]] — GAT использует ту же идею внимания, что Transformer

## Источники

- [Chapter 4: The Graph Neural Network Model — McGill (Hamilton)](https://cs.mcgill.ca/~wlh/comp766/files/chapter4_draft_mar29.pdf)
- [Message Passing in GNNs — Kumo.ai](https://kumo.ai/pyg/concepts/message-passing/)
- [Comparing GCN, GraphSAGE, and GAT — apxml.com](https://apxml.com/courses/introduction-to-graph-neural-networks/chapter-3-foundational-gnn-architectures/comparing-gcn-graphsage-gat)
- [A Gentle Introduction to Graph Neural Networks — Distill](https://distill.pub/2021/gnn-intro/)
