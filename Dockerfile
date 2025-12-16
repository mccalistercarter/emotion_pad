# Use python 3.10, copy the code in, install dependencies, and run demo_script.py,

FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "demo_script.py"]