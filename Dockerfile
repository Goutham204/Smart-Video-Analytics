FROM public.ecr.aws/lambda/python:3.10

RUN yum install -y mesa-libGL && yum clean all

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ${LAMBDA_TASK_ROOT}/

RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" && \
    mv yolov8n.pt ${LAMBDA_TASK_ROOT}/models/yolov8n.pt

CMD ["main.handler"]