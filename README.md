# PDF Processor Docker Solution

This solution processes PDF files using NLP models (Sentence Transformers + T5) to extract and summarize key sections based on a given persona and task.

---

## 🐳 Docker Usage

### 🔧 1. Build the Docker Image

```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .


docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  mysolutionname:somerandomidentifier

.
├── app/
│   ├── main.py               # Main processing script
│   └── requirements.txt      # (Optional if pip install used directly in Dockerfile)
├── input/                    # Place your .pdf files here
├── output/                   # JSON summaries will be written here
├── Dockerfile
└── README.md

