# образ на основе которого создаём контейнер
FROM python:3.10

# рабочая директория внутри проекта
WORKDIR /usr/src/lct_cv

# переменные окружения для python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# устанавливаем зависимости
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# копируем содержимое текущей папки в контейнер
COPY . .