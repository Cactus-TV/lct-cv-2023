# образ на основе которого создаём контейнер
FROM python:3.10

# рабочая директория внутри проекта
WORKDIR /usr/src/lct_cv

# переменные окружения для python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# устанавливаем зависимости
RUN pip install --upgrade pip
# RUN apt-get add --update --no-cache --virtual .tmp-build-deps \
#       gcc g++ cmake libc-dev linux-headers libblas liblapack-dev gfortran libatlas-base-dev
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# копируем содержимое текущей папки в контейнер
COPY . .