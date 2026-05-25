# DPO — прямая оптимизация предпочтений

> DPO (Direct Preference Optimization) — упрощённая альтернатива RLHF: вместо отдельной reward model и RL-цикла задача выравнивания переформулируется как задача бинарной классификации и решается стандартным градиентным спуском.

## Ключевая идея

Авторы DPO (Rafailov et al., 2023) показали, что задача, которую RLHF решает итеративно через PPO, имеет **аналитическое решение** в замкнутой форме. Это позволяет напрямую вывести функцию потерь без reward model и RL-цикла.

Вместо вопроса «какой ответ максимизирует награду?» DPO задаёт вопрос «этот ответ предпочтительный или нет?» и оптимизирует это напрямую на парах предпочтений.

## Функция потерь

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

- $\pi_\theta$ — обучаемая модель.
- $\pi_{\text{ref}}$ — замороженная SFT-модель (опорная).
- $y_w$ — «выигравший» ответ, $y_l$ — «проигравший».
- $\beta$ — параметр, управляющий отклонением от опорной модели.

Логика: модель должна назначать относительно более высокую вероятность лучшему ответу, чем опорная модель, и более низкую — худшему.

## Что убирает DPO по сравнению с RLHF

| Компонент | RLHF + PPO | DPO |
|---|---|---|
| Reward model | нужна | не нужна |
| RL-цикл (PPO) | нужен | не нужен |
| Моделей в памяти | ~4 | ~2 |
| Стабильность | ниже | выше |
| Данные | онлайн-генерация возможна | статический датасет пар |

## Ограничения DPO

- Обучается на **статическом** датасете пар предпочтений — не может использовать данные, генерируемые онлайн.
- Более склонен к **overfitting** на маленьких наборах предпочтений.
- RLHF обучает обобщённую reward function и потенциально лучше обобщается на запросы, не вошедшие в датасет.

## Связанные темы

- [[llm-training/overview|Пайплайн тренировки LLM]]
- [[llm-training/rlhf|RLHF: Reward Model и PPO]]
- [[llm-training/sft|Supervised Fine-Tuning (SFT)]]

## Источники

- [DPO paper — Rafailov et al., 2023](https://arxiv.org/pdf/2305.18290)
- [Direct Preference Optimization — Deep (Learning) Focus](https://cameronrwolfe.substack.com/p/direct-preference-optimization)
- [Simplifying Alignment: From RLHF to DPO](https://huggingface.co/blog/ariG23498/rlhf-to-dpo)
