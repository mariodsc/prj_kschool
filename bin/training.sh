#!/bin/sh
# cloud.google.com/ai-platform/training/docs/runtime-version-list
# cloud.google.com/ai-platform/training/docs/machine-types

EPOCHS=30
BATCH_SIZE=1024

gcloud ai-platform jobs submit training mnist_iht_`date +"%s"` \
  --python-version 3.7 \
  --runtime-version 2.3 \
  --scale-tier BASIC \
  --package-path ./trainer \
  --module-name trainer.task \
  --region europe-west1 \
  --job-dir gs://mdesousac-mnist/tmp \
  -- \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --model-output-path gs://mdesousac-mnist/models

