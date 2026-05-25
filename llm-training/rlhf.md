# RLHF: Reward Model и PPO

> RLHF (Reinforcement Learning from Human Feedback) — этап выравнивания, на котором SFT-модель дообучается через RL-цикл: сначала обучают «судью» (reward model) на человеческих предпочтениях, потом с его помощью улучшают политику модели через PPO.

## Два шага RLHF

### Шаг 1 — Обучение Reward Model (RM)

**Reward Model** — отдельная нейросеть, которая принимает пару (prompt, response) и выдаёт скалярную оценку: насколько хорош ответ.

Процесс сбора данных:
1. SFT-модель генерирует несколько вариантов ответа на один и тот же запрос.
2. Аннотаторы-люди ранжируют эти ответы от лучшего к худшему.
3. Из ранжирований формируются пары предпочтений $(y_w, y_l)$ — «выигравший» и «проигравший» ответы.

RM обучается предсказывать, какой из двух ответов предпочтёт человек, с помощью **Bradley-Terry модели**:

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\!\left(r(x, y_w) - r(x, y_l)\right)\right]$$

где $r(x, y)$ — скор reward model для запроса $x$ и ответа $y$, $\sigma$ — сигмоид.

### Шаг 2 — Fine-Tuning через PPO

Обученный RM используется как среда в RL-цикле. Языковая модель становится **политикой** (policy): она принимает промпт и генерирует ответ-действие. Награда — скор от RM.

Для стабилизации в цель добавляется **KL-штраф**: политика не должна сильно отклоняться от SFT-модели (иначе модель может «взломать» RM, генерируя бессмысленный, но высоко оцениваемый текст):

$$r_{\text{итого}}(x, y) = r_{\text{RM}}(x, y) - \beta \cdot \text{KL}\!\left[\pi_\theta(y \mid x) \,\|\, \pi_{\text{SFT}}(y \mid x)\right]$$

**PPO (Proximal Policy Optimization)** обновляет веса модели маленькими шагами, ограничивая изменение политики через clip:

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}\left[\min\!\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$

где $r_t(\theta) = \pi_\theta / \pi_{\text{old}}$ — отношение новой и старой политики, $\hat{A}_t$ — оценка преимущества.

## Четыре модели в PPO-цикле

RLHF с PPO требует одновременно держать в памяти четыре модели:
1. **Policy model** — обучаемая языковая модель.
2. **Value model** — оценщик ожидаемой будущей награды.
3. **Reward model** — замороженный «судья».
4. **Reference model** — замороженная SFT-модель для KL-штрафа.

Это делает RLHF вычислительно тяжёлым и чувствительным к гиперпараметрам.

## Ограничения RLHF

- Reward model может содержать ошибки и привести к **reward hacking** — модель научится обманывать RM вместо того, чтобы давать хорошие ответы.
- Нестабильность PPO требует тщательного подбора гиперпараметров.
- Дорогостоящий сбор предпочтений от людей.

## Связанные темы

- [[llm-training/overview|Пайплайн тренировки LLM]]
- [[llm-training/sft|Supervised Fine-Tuning (SFT)]]
- [[llm-training/dpo|DPO — прямая оптимизация предпочтений]]

## Источники

- [RLHF Explained (2026)](https://decodethefuture.org/en/rlhf-explained/)
- [Secrets of RLHF in LLMs Part I: PPO](https://ar5iv.labs.arxiv.org/html/2307.04964)
- [Complete Guide On Fine-Tuning LLMs using RLHF](https://www.labellerr.com/blog/reinforcement-learning-from-human-feedback/)
