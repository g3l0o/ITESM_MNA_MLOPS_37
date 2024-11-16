FROM python:3.10.14

# Set the working directory
WORKDIR /mlops

# Copy the model and server code
COPY models/model.pkl /mlops/
COPY mlops/stages/predict.py /mlops/


# Install dependencies
RUN pip install fastapi uvicorn scikit-learn==1.3.2 pydantic

# Expose port 8000
EXPOSE 8000

# Run the application
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]