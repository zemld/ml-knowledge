# Графовые нейросети (GNN)

> Класс нейросетей, работающих с данными в виде графов — узлы хранят признаки объектов, рёбра описывают связи между ними. GNN обновляет представление каждого узла через агрегацию сообщений от соседей, что позволяет учитывать структуру графа при обучении.

## Что такое GNN

Большинство нейросетей ожидают данные в фиксированной сетке — изображение как матрица пикселей, текст как последовательность токенов. Но многие реальные данные имеют неравномерную структуру связей: молекулы, социальные сети, транзакционные графы, дорожные сети. GNN решают задачу обучения именно на таких данных.

Ключевая идея — **передача сообщений (message passing)**: на каждом слое каждый узел собирает информацию от своих соседей, агрегирует её (сумма, среднее, max) и обновляет своё представление. После нескольких слоёв узел «знает» о структуре своего окружения на глубину $k$ переходов.

## Уровни задач

GNN решают задачи трёх уровней:

- [[neural-networks/gnn/node-level-tasks|Node-level задачи]] — предсказание свойств отдельных вершин
- [[neural-networks/gnn/edge-level-tasks|Edge-level задачи]] — предсказание свойств или наличия рёбер
- [[neural-networks/gnn/graph-level-tasks|Graph-level задачи]] — предсказание свойств всего графа целиком

## Алгоритмы и методы

- [[neural-networks/gnn/neighborhood-aggregation|Neighborhood Aggregation (Message Passing)]] — основной механизм GNN: итерационная агрегация признаков соседей через AGGREGATE и COMBINE
- [[neural-networks/gnn/node2vec|node2vec]] — обучение эмбеддингов вершин через смещённые случайные блуждания и Skip-Gram

## Связанные темы

- [[neural-networks/overview|Нейросети]]
- [[neural-networks/transformer/overview|Transformer]] — механизм внимания можно рассматривать как GNN на полном графе

## Источники

- [A Gentle Introduction to Graph Neural Networks — Distill](https://distill.pub/2021/gnn-intro/)
- [Graph Neural Networks Guide — V7 Labs](https://www.v7labs.com/blog/graph-neural-networks-guide)
- [GNN with PyG — Towards Data Science](https://towardsdatascience.com/graph-neural-networks-with-pyg-on-node-classification-link-prediction-and-anomaly-detection-14aa38fe1275/)
