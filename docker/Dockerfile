FROM public.ecr.aws/lambda/python:3.10
RUN python3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN python3.10 -m pip install smart_open[s3]
COPY app.py ./
CMD ["app.lambda_handler"]