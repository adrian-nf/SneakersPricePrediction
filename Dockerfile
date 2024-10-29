FROM python:3.11 as production

# Set the working directory
WORKDIR /app

COPY requirements.txt .
COPY app.py .
COPY ./models ./models

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 7869

ENTRYPOINT ["python", "app.py"]