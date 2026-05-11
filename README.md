# AgroLens-AI

AgroLens-AI provides an automated diagnostic tool to identify plant diseases from leaf images using Deep Learning and Computer Vision. The platform helps farmers and researchers detect diseases at an early stage, enabling precision treatment, reducing excessive pesticide usage, and minimizing crop yield loss.

# Getting Started

### Prerequisites

Before running the project, ensure the following tools are installed:

* Git
* Python 3.8 or higher
* pip package manager
* Virtual Environment (recommended)

### Installation

## 1. Clone the Repository

```bash
git clone https://github.com/code-with-bhumi/AgroLens-AI.git
cd AgroLens-AI
```

## 2. Create Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

# Backend Setup & Execution

### Navigate to Backend Folder

```bash
cd backend
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Backend Server

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

### API Access

After starting the server:

* API Base URL: `http://127.0.0.1:8000`
* Swagger Documentation: `http://127.0.0.1:8000/docs`

# Frontend Setup & Execution

### Navigate to Frontend Folder

```bash
cd frontend
```

### Run Frontend

Open the following file directly in your browser:

```bash
frontend/index.html
```

## License
Distributed under the MIT License.

**Maintained by:** [Vasi Khan](https://github.com/vasi2904k), [Bhumi Shah](https://github.com/code-with-bhumi), [Baviya](https://github.com/Baviyas)
