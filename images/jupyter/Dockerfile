FROM jupyter/base-notebook:python-3.11

# Install the dependencies
RUN pip install requests snowflake-snowpark-python[pandas]
RUN pip install pymilvus==2.4.0
RUN pip install onnxruntime
RUN pip install huggingface_hub
RUN pip install transformers

# Set the working directory
WORKDIR /workspace/

# Download the onnx model
RUN wget -O model.onnx  https://huggingface.co/GPTCache/paraphrase-albert-onnx/resolve/main/model.onnx?download=true

# Expose Jupyter Notebook port
EXPOSE 8888

# Copy the notebooks directory to the container's /app directory
RUN mkdir /workspace/.local /workspace/.cache && chmod 777 -R /workspace

# Copy the demo
COPY --chmod=777 TestMilvus.ipynb model.py ./

# Copy the tokenizer files
RUN mkdir /workspace/my_tokenizer_directory && chmod 777 -R /workspace/my_tokenizer_directory
COPY --chmod=777 ./my_tokenizer_directory/special_tokens_map.json ./my_tokenizer_directory/tokenizer_config.json ./my_tokenizer_directory/tokenizer.json ./my_tokenizer_directory

# Run Jupyter Notebook on container startup
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
