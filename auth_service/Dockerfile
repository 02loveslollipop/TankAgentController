FROM python:3.13-trixie

WORKDIR /app
ENV PYTHONPATH="/app:${PYTHONPATH}"

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# Copy the repo context into /app/auth_service so the package name exists
COPY . /app/auth_service

EXPOSE 8001

CMD ["uvicorn", "auth_service.main:app", "--host", "0.0.0.0", "--port", "8001"]