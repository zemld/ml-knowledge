# Пайплайн тренировки LLM

> Обучение большой языковой модели состоит из трёх последовательных этапов: предобучение на огромных корпусах текста, supervised fine-tuning для следования инструкциям и выравнивание по предпочтениям человека через RLHF или DPO.

## Зачем нужны три этапа

Каждый этап решает отдельную задачу:

1. **Предобучение** — дать модели широкие знания о языке, фактах и рассуждениях, обучая её предсказывать следующий токен на терабайтах текста.
2. **SFT** — научить модель следовать инструкциям, показывая ей примеры «запрос → хороший ответ».
3. **Выравнивание (RLHF / DPO)** — сделать так, чтобы модель давала ответы, которые люди считают полезными и безопасными.

Без предобучения у модели нет знаний. Без SFT она не понимает, что от неё хотят. Без выравнивания она может давать правдоподобные, но вредные или бессмысленные ответы.

## Разделы

- [[llm-training/pretraining|Предобучение (Pretraining)]] — данные, токенизация, цель обучения и масштаб
- [[llm-training/sft|Supervised Fine-Tuning (SFT)]] — обучение на парах «инструкция — ответ»
- [[llm-training/rlhf|RLHF: Reward Model и PPO]] — обучение с подкреплением на основе предпочтений
- [[llm-training/dpo|DPO — прямая оптимизация предпочтений]] — альтернатива RLHF без отдельной reward model
- [[llm-training/finetuning-methods|Методы файн-тьюнинга]] — Full FT, LoRA, QLoRA, Adapters, Prefix/Prompt Tuning
- [[llm-training/quantization|Квантизация моделей]] — INT8/INT4, scale+zero-point, GPTQ, GGUF, AWQ, NF4

## Связанные темы

- [[neural-networks/transformer/overview|Transformer — обзор архитектуры]]

## Источники

- [InstructGPT paper — Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)
- [Pretraining: Breaking Down the Modern LLM Training Pipeline](https://mlops.community/blog/pretraining-breaking-down-the-modern-llm-training-pipeline)
