FROM python:3.13-trixie

WORKDIR /app
ENV PYTHONPATH="/app/backend:${PYTHONPATH}"

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY . /app/backend

EXPOSE 8000

CMD ["python", "/app/backend/main.py"]