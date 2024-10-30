FROM python:3.11 AS production

# Set the working directory
WORKDIR /app

COPY requirements.txt .
COPY app.py .
COPY ./models ./models
COPY ./data ./data

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 7860

ENTRYPOINT ["python", "app.py"]