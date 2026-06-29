# Confirmed running version: .venv/Scripts/python.exe --version -> 3.11.9
# (Glue jobs elsewhere in this repo pin python_version="3.9" -- that's an
# unrelated AWS Glue runtime constraint, not this app's target version.)
FROM python:3.11-slim

WORKDIR /app

COPY query/requirements.txt query/requirements.txt
RUN pip install --no-cache-dir -r query/requirements.txt

COPY query/ query/

EXPOSE 8000

CMD ["uvicorn", "query.server:app", "--host", "0.0.0.0", "--port", "8000"]
