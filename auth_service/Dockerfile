FROM python:3.13-trixie

WORKDIR /app
ENV PYTHONPATH="/app:${PYTHONPATH}"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "auth_service.main:app", "--host", "0.0.0.0", "--port", "8001"]