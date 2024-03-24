FROM python:bullseye
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY src/* .
CMD ["python3", "-m", "uvicorn", "--host", "0.0.0.0", "main:app"]