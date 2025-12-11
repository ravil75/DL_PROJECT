#!/bin/bash

uvicorn demo.api.main:app --host 0.0.0.0 --port 8000 &

sleep 30

export API_URL=http://localhost:8000
streamlit run demo/streamlit_app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true