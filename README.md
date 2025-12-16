Audio Compression with Fully Convolutional Autoencoders

Overview

The project trains и evaluates нейросетевой компрессор аудио. Звук превращается в комплексный STFT (лог-магнитуда + фаза как два канала), сжимается fully-convolutional автоэнкодером и восстанавливается обратно. Цель — изучать компрессию без меток и добиваться баланса между качеством (SNR/SI-SDR/LSD) и размером латента/блоба.

⸻

Project Structure

.
├── src/
│   ├── audio_io.py          # Загрузка/сохранение аудио
│   ├── stft_utils.py        # STFT/iSTFT, utils для магн/фазы
│   ├── dataset.py           # PyTorch Dataset: возвращает стек [лог-маг, фаза]
│   ├── model_fc_ae.py       # Fully-conv автоэнкодер с даун/ап-семплом и bottleneck-resblock
│   ├── train.py             # Обучение (комплексный спекстр., мел/волновые лоссы)
│   ├── inference.py         # compress/decompress с квантованием латента
│   ├── eval.py              # Оценка: SNR, SI-SDR, LSD, печать компрессии и размер блоба
│   ├── test_reconstruct.py  # Реконструкция из чекпоинта
│   ├── size_check.py        # Проверка форм/компрессии
│   └── numba_cache.py       # Опциональный кеш numba
├── data/                    # Тренировочные wav/flac
├── data_test/               # Тестовые wav/flac
├── results/                 # Чекпоинты и recon_*.wav
└── README.md

⸻

Model

- Вход: 2 канала (лог-магнитуда, фаза) комплексного STFT.
- Энкодер: несколько Conv2d со stride=2 (доп. даунсемпл для более компактного латента) и bottleneck-ResBlock.
- Декодер: зеркальный ConvTranspose2d, предсказывает оба канала (магнитуда и фаза), чтобы не зависеть от сохранённой фазы.
- Квантование: per-channel max-norm, int8/16 (по умолчанию 8 бит) без бит-пэкинга.

⸻

Training

Запуск:

python src/train.py

Основные настройки через env:
- MAX_TRAIN_FILES (default 20) — сколько файлов брать для обучения.
- MAX_AUDIO_SECONDS (default 8) — случайный кроп для ускорения (≤0 отключает).
- MODEL_CHANNELS (default 8) — ширина сети; меньше = сильнее компрессия.
- MEL_LOSS_WEIGHT (default 0.003) — вес мел-лосса (меньше = мягче, меньше “робота”).
- EPOCHS (default 80).

Лоссы: L1 по магн/фазе, L1 по waveform, мел-лосс на двух разрешениях.

⸻

Inference and Evaluation

- Реконструкция: python src/test_reconstruct.py (нужен results/model_last.pth).
- Оценка: python src/eval.py — сохраняет recon_*.wav, печатает
  компрессию (input/latent, размер блоба), SNR, SI-SDR, LSD.
- compress/decompress в `src/inference.py`: можно хранить фазу (store_phase=True) или полагаться на предсказанную фазу; quant_bits по умолчанию 8.

⸻

Quick smoke test
If you just want to verify the pipeline works:

python src/smoke_test.py

Берёт data_test/*.wav (или генерит 1s 440 Hz), гоняет два режима (почти lossless и деградированный), пишет recon/residual в results/.

⸻

Future directions

- Уйти в time-domain энкодер/декодер (Conv/TasNet-style) для лучшей перцепции при том же объёме.
- Полностью learned phase/complex decoding: улучшить фазовое моделирование без хранения фазы.
- Перцептивные метрики: добавить PESQ/STOI (для речи), multi-band LSD/SDR, MOS-proxy.
- Variational/VQ латент: дискретный кодбук или KL-регуляризация для более управляемой компрессии.
- Multi-resolution/learned filterbanks вместо фиксированной STFT.

⸻

Requirements
	•	Python 3.8+
	•	PyTorch
	•	NumPy
	•	SciPy
	•	librosa (optional)
	•	numba (optional)

pip install torch numpy scipy librosa numba

⸻

License

Intended for research and educational use.
