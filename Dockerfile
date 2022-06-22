# Сборка ---------------------------------------

# В качестве базового образа для сборки используем gcc:latest
FROM python:latest

# Установим рабочую директорию для сборки GoogleTest
ADD . /app
WORKDIR /build

# Скачаем все необходимые пакеты и выполним сборку GoogleTest
# Такая длинная команда обусловлена тем, что
# Docker на каждый RUN порождает отдельный слой,
# Влекущий за собой, в данном случае, ненужный оверхед
RUN ls
RUN pip install -r ../app/requirements.txt && \
    pip install pybuilder
RUN ls ../app
# USER sample
#RUN pyb --start-project
WORKDIR /app
RUN    pyb -v



# Скопируем директорию /src в контейнер



# Запуск ---------------------------------------

